"""
Synthetic data generation via Bayesian Networks

Based on following paper

Zhang J, Cormode G, Procopiuc CM, Srivastava D, Xiao X.
PrivBayes: Private Data Release via Bayesian Networks. (2017)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from collections import namedtuple

import synthesis.synthesizers.utils as utils
from synthesis.synthesizers._base import BaseDPSynthesizer
from thomas.core.bayesiannetwork import BayesianNetwork
from diffprivlib.mechanisms import Exponential

AP_pair = namedtuple('AttributeParentPair', ['node', 'parents'])


class PrivBayes(BaseDPSynthesizer):
    """PrivBayes: Private Data Release via Bayesian Networks (Zhang et al 2017)

    Version:
    - vanilla encoding
    - mutual information as score function

    Extended:
    - Ability to initialize network

    Default hyperparameters set according to paper recommendations
    """

    def __init__(self, epsilon=1.0, theta_usefulness=4, epsilon_split=0.3,
                 score_function='R', network_init=None, verbose=True):
        super().__init__(epsilon=epsilon, verbose=verbose)
        self.theta_usefulness = theta_usefulness
        self.epsilon_split = epsilon_split  # also called Beta in paper
        self.score_function = score_function  # choose between 'R' and 'MI'
        self.network_init = network_init

    def fit(self, data):
        data = self._check_input_data(data)
        self._check_init_args()

        self._greedy_bayes(data)
        self._compute_conditional_distributions(data)
        self.model_ = BayesianNetwork.from_CPTs('PrivBayes', self.cpt_.values())
        return self

    def _check_init_args(self):
        super()._check_init_args()
        self._check_score_function()

    def _check_score_function(self):
        """Checks input score function and sets sensitivity"""
        if self.score_function.upper() not in ['R', 'MI']:
            raise ValueError("Score function must be 'R' or 'MI'")

        if self.score_function.upper() == 'R':
            self._score_sensitivity = (3 / self.n_records_fit_) + (2 / self.n_records_fit_**2)

        # note: for simplicity we assume that all AP_pairs are non-binary, which is the upperbound of MI sensitivity
        elif self.score_function.upper() == 'MI':
            self._score_sensitivity = (2 / self.n_records_fit_) * np.log((self.n_records_fit_ + 1) / 2) + \
                              (((self.n_records_fit_ - 1) / self.n_records_fit_) *
                               np.log((self.n_records_fit_ + 1) / (self.n_records_fit_ - 1)))

    def sample(self, n_records=None):
        self._check_is_fitted()
        n_records = n_records or self.n_records_fit_

        synth_data = self._generate_data(n_records)
        if self.verbose:
            print("\nSynthetic Data Generated\n")
        return synth_data

    def _greedy_bayes(self, data):
        nodes, nodes_selected = self._init_network(data)

        # normally len(nodes) - 1, unless user initialized part of the network
        self._n_nodes_dp_computed = len(nodes) - len(nodes_selected)

        for i in range(len(nodes_selected), len(nodes)):
            if self.verbose:
                print("{}/{} - Evaluating next node to add to network".format(i + 1, len(self.columns_)))

            nodes_remaining = nodes - nodes_selected

            # select ap_pair candidates
            ap_pairs = []
            for node in nodes_remaining:
                max_domain_size = self._max_domain_size(data, node)
                max_parent_sets = self._max_parent_sets(data, nodes_selected, max_domain_size)

                # empty set - domain size of node violates theta_usefulness
                if len(max_parent_sets) == 0:
                    ap_pairs.append(AP_pair(node, parents=None))
                # [empty set] - no parents found that meet domain size restrictions
                elif len(max_parent_sets) == 1 and len(max_parent_sets[0]) == 0:
                    ap_pairs.append(AP_pair(node, parents=None))
                else:
                    ap_pairs.extend([
                        AP_pair(node, parents=tuple(p)) for p in max_parent_sets
                    ])
            if self.verbose:
                print("Number of AttributeParentPair candidates: {}".format(len(ap_pairs)))
                print('Candidates: {}'.format(ap_pairs))

            scores = self._compute_scores(data, ap_pairs)
            sampled_pair = self._exponential_mechanism(ap_pairs, scores)

            if self.verbose:
                print("Selected node: '{}' - with parents: {}\n".format(sampled_pair.node, sampled_pair.parents))
            nodes_selected.add(sampled_pair.node)
            self.network_.append(sampled_pair)
        if self.verbose:
            print("Learned Network Structure\n")
        return self

    def _max_domain_size(self, data, node):
        """Computes the maximum domain size a node can have to satisfy theta-usefulness"""
        node_cardinality = utils.cardinality(data[node])
        max_domain_size = self.n_records_fit_ * (1 - self.epsilon_split) * self.epsilon / \
                          (2 * len(self.columns_) * self.theta_usefulness * node_cardinality)
        return max_domain_size

    def _max_parent_sets(self, data, v, max_domain_size):
        """Refer to algorithm 5 in paper - max parent set is 1) theta-useful and 2) maximal."""
        if max_domain_size < 1:
            return set()
        if len(v) == 0:
            return [set()]

        x = np.random.choice(tuple(v))
        x_domain_size = utils.cardinality(data[x])
        x = {x}

        v_without_x = v - x

        parent_sets1 = self._max_parent_sets(data, v_without_x, max_domain_size)
        parent_sets2 = self._max_parent_sets(data, v_without_x, max_domain_size / x_domain_size)

        for z in parent_sets2:
            if z in parent_sets1:
                parent_sets1.remove(z)
            parent_sets1.append(z.union(x))
        return parent_sets1

    def _init_network(self, X):
        self._binary_columns = [c for c in X.columns if X[c].unique().size <= 2]
        nodes = set(X.columns)

        if self.network_init is not None:
            nodes_selected = set(n.node for n in self.network_init)
            # print("Pre-defined network init: {}".format(self.network_))
            for i, pair in enumerate(self.network_init):
                if self.verbose:
                    print("{}/{} - init node {} - with parents: {}".format(i + 1, len(self.network_init),
                                                                           pair.node, pair.parents))
            self.network_ = self.network_init.copy()
            return nodes, nodes_selected

        # if set_network is not called we start with a random first node
        self.network_ = []
        nodes_selected = set()

        root = np.random.choice(tuple(nodes))
        self.network_.append(AP_pair(node=root, parents=None))
        nodes_selected.add(root)
        if self.verbose:
            print("1/{} - Root of network: {}\n".format(X.shape[1], root))
        return nodes, nodes_selected

    def set_network(self, network):
        assert [isinstance(n, AP_pair) for n in network], "input network does not consists of AP_pairs"
        self.network_init = network
        return self

    def _compute_scores(self, data, ap_pairs):
        """Compute score for all ap_pairs"""
        scores = np.empty(len(ap_pairs))

        for idx, pair in enumerate(ap_pairs):
            if pair.parents is None:
                scores[idx] = 0
            elif self.score_function == 'R':
                scores[idx] = self.r_score(data, pair.node, pair.parents)
            elif self.score_function == 'MI':
                scores[idx] = self.mi_score(data, pair.node, pair.parents)
        return scores

    def _exponential_mechanism(self, ap_pairs, scores):
        """select AP_pair with exponential mechanism"""
        local_epsilon = self.epsilon * self.epsilon_split / self._n_nodes_dp_computed
        dp_mech = Exponential(epsilon=local_epsilon, sensitivity=self._score_sensitivity,
                              utility=list(scores), candidates=ap_pairs)
        sampled_pair = dp_mech.randomise()
        return sampled_pair

    def _compute_conditional_distributions(self, data):
        self.cpt_ = dict()

        local_epsilon = self.epsilon * (1 - self.epsilon_split) / len(self.columns_)

        for idx, pair in enumerate(self.network_):
            if pair.parents is None:
                attributes = [pair.node]
            else:
                attributes = [*pair.parents, pair.node]

            cpt_size = utils.cardinality(data[attributes])
            if self.verbose:
                print('Learning conditional probabilities: {} - with parents {} '
                      '~ estimated size: {}'.format(pair.node, pair.parents, cpt_size))

            dp_cpt = utils.dp_conditional_distribution(data[attributes], epsilon=local_epsilon)
            self.cpt_[pair.node] = dp_cpt
        return self

    def _generate_data(self, n_records):
        synth_data = np.empty([n_records, len(self.columns_)], dtype=object)

        for i in range(n_records):
            if self.verbose:
                print('Number of records generated: {} / {}'.format(i + 1, n_records), end='\r')
            record = self._sample_record()
            synth_data[i] = list(record.values())

        # numpy.array to pandas.DataFrame with original column ordering
        synth_data = pd.DataFrame(synth_data, columns=[c.node for c in self.network_])[self.columns_]
        return synth_data

    def _sample_record(self):
        """samples a value column for column by conditioning for parents"""
        record = {}
        for col_idx, pair in enumerate(self.network_):
            node = self.model_[pair.node]
            node_cpt = node.cpt
            node_states = node.states

            if node.conditioning:
                parent_values = [record[p] for p in node.conditioning]
                node_probs = node_cpt[tuple(parent_values)]
            else:
                node_probs = node_cpt.values
            sampled_node_value = np.random.choice(node_states, p=node_probs)

            record[node.name] = sampled_node_value
        return record

    @staticmethod
    def mi_score(data, columns_a, columns_b):
        columns_a = utils._ensure_arg_is_list(columns_a)
        columns_b = utils._ensure_arg_is_list(columns_b)

        data_a = data[columns_a].squeeze()
        if len(columns_b) == 1:
            data_b = data[columns_b].squeeze()
        else:
            data_b = data.loc[:, columns_b].apply(lambda x: ' '.join(x.values), axis=1).squeeze()
        return mutual_info_score(data_a, data_b)

    @staticmethod
    def mi_score_thomas(data, columns_a, columns_b):
        columns_a = utils._ensure_arg_is_list(columns_a)
        columns_b = utils._ensure_arg_is_list(columns_b)

        prob_a = utils.compute_distribution(data[columns_a])
        prob_b = utils.compute_distribution(data[columns_b])
        prob_joint = utils.compute_distribution(data[columns_a + columns_b])

        # todo: pull-request thomas to add option to normalize to remove 0's
        # align
        prob_div = prob_joint / (prob_b * prob_a)
        prob_joint, prob_div = prob_joint.extend_and_reorder(prob_joint, prob_div)

        # remove zeros as this will result in issues with log
        prob_joint = prob_joint.values[prob_joint.values != 0]
        prob_div = prob_div.values[prob_div.values != 0]
        mi = np.sum(prob_joint * np.log(prob_div))
        # mi = np.sum(p_nodeparents.values * np.log(p_nodeparents / (p_parents * p_node)))
        return mi


    @staticmethod
    def r_score(data, columns_a, columns_b):
        """An alternative score function to mutual information with lower sensitivity - can be used on non-binary domains.
        Relies on the L1 distance from a joint distribution to a joint distributions that minimizes mutual information.
        Refer to Lemma 5.2
        """
        columns_a = utils._ensure_arg_is_list(columns_a)
        columns_b = utils._ensure_arg_is_list(columns_b)

        # compute distribution that minimizes mutual information
        prob_a = utils.compute_distribution(data[columns_a])
        prob_b = utils.compute_distribution(data[columns_b])
        prob_independent = prob_b * prob_a

        # compute joint distribution
        prob_joint = utils.joint_distribution(data[columns_a + columns_b])

        # substract not part of thomas - need to ensure alignment
        # todo: should be part of thomas - submit pull-request to thomas
        prob_joint, prob_independent = prob_joint.extend_and_reorder(prob_joint, prob_independent)
        l1_distance = 0.5 * np.sum(np.abs(prob_joint.values - prob_independent.values))
        return l1_distance




if __name__ == "__main__":
    data_path = '../../examples/data/original/adult.csv'
    data = pd.read_csv(data_path, delimiter=', ', engine='python')
    columns = ['age', 'sex', 'education', 'workclass', 'income']
    data = data.loc[:, columns]
    print(data.head())

    pb = PrivBayes(epsilon=0.1)
    pb.fit(data)
    df_synth = pb.sample(1000)
    pb.score(data, df_synth, score_dict=True)

    """test pb with init network"""
    # init_network = [AP_pair(node='age', parents=None),
    #                 AP_pair(node='education', parents=('age',)),
    #                 AP_pair(node='sex', parents=('age', 'education'))]
    #
    # pbinit = PrivBayes()
    # pbinit.set_network(init_network)
    # pbinit.fit(df)
    # df_synth_init = pbinit.sample(1000)


    """test scoring functions"""
    pair = pb.network_[3]
    pb.mi_score(data, pair.node, pair.parents)
    pb.mi_score_thomas(data, pair.node, pair.parents)
    pb.r_score(data, pair.node, pair.parents)