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

from synthesis.synthesizers._base import BaseDPSynthesizer
import synthesis.synthesizers.utils as utils
from thomas.core.bayesiannetwork import BayesianNetwork

NodeParentPair = namedtuple('NodeParentPair', ['node', 'parents'])


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
                 network_init=None, verbose=True):
        super().__init__(epsilon=epsilon, verbose=verbose)
        self.theta_usefulness = theta_usefulness
        self.epsilon_split = epsilon_split  # also called Beta in paper
        self.network_init = network_init

    def fit(self, data):
        self._check_init_args()
        data = self._check_input_data(data)

        self._greedy_bayes(data)
        self._compute_conditional_distributions(data)
        self.model_ = BayesianNetwork.from_CPTs('PrivBayes', self.cpt_.values())
        return self

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

            # select NodeParentPair candidates
            node_parent_pairs = []
            for node in nodes_remaining:
                max_domain_size = self._max_domain_size(data, node)
                max_parent_sets = self._max_parent_sets(data, nodes_selected, max_domain_size)

                # empty set - no parents found that meet domain size restrictions
                if len(max_parent_sets) == 1 and len(max_parent_sets[0]) == 0:
                    node_parent_pairs.append(NodeParentPair(node, parents=None))
                else:
                    node_parent_pairs.extend([
                        NodeParentPair(node, parents=tuple(p)) for p in max_parent_sets
                    ])
            if self.verbose:
                print("Number of NodeParentPair candidates: {}".format(len(node_parent_pairs)))
                print('Candidates: {}'.format(node_parent_pairs))

            scores = self._compute_scores(data, node_parent_pairs)
            sampled_pair = self._exponential_mechanism(data, node_parent_pairs, scores)

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
        max_domain_size = self.n_records_fit_ * (1 - self.epsilon_split) / \
                          (2 * len(self.columns_) * self.theta_usefulness * node_cardinality)
        return max_domain_size

    def _max_parent_sets(self, data, v, max_domain_size):
        """refer to algorithm 5 in paper
        max parent set is 1) theta-useful and 2) maximal."""
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
        self.network_.append(NodeParentPair(node=root, parents=None))
        nodes_selected.add(root)
        if self.verbose:
            print("1/{} - Root of network: {}\n".format(X.shape[1], root))
        return nodes, nodes_selected

    def set_network(self, network):
        assert [isinstance(n, NodeParentPair) for n in network], "input network does not consists of " \
                                                                 "NodeParentPairs"
        self.network_init = network
        return self

    def _compute_scores(self, X, node_parent_pairs):
        scores = np.empty(len(node_parent_pairs))
        for idx, pair in enumerate(node_parent_pairs):
            scores[idx] = self._compute_mutual_information(X, pair)
        return scores

    def _compute_mutual_information(self, X, pair):
        df_node = X[pair.node].values
        # if there aren't any parents that meet theta-usefulness requirement, there is no mutual information to compute
        if pair.parents is None:
            return 0
        if len(pair.parents) == 1:
            df_parent = X[pair.parents[0]].values
        else:
            df_parent = X.loc[:, pair.parents].apply(lambda x: ' '.join(x.values), axis=1).values
        return mutual_info_score(df_node, df_parent)

    def _exponential_mechanism(self, X, node_parent_pairs, scores):
        # todo check if dp correct -> e.g. 2*scaling?
        scaling_factors = self._compute_scaling_factor(X, node_parent_pairs)
        sampling_distribution = np.exp(scores / 2 * scaling_factors)
        normalized_sampling_distribution = sampling_distribution / sampling_distribution.sum()
        pair_idx = np.arange(len(node_parent_pairs))
        sampled_pair_idx = np.random.choice(pair_idx, p=normalized_sampling_distribution)
        sampled_pair = node_parent_pairs[sampled_pair_idx]
        return sampled_pair

    def _compute_scaling_factor(self, X, node_parent_pairs):
        n_records = self.n_records_fit_
        scaling_factors = np.empty(len(node_parent_pairs))

        for idx, pair in enumerate(node_parent_pairs):
            if pair.node in self._binary_columns or (pair.parents is not None and
                                                     len(pair.parents) == 1 and
                                                     pair.parents[0] in self._binary_columns):
                sensitivity = (np.log(n_records) / n_records) + \
                              (((n_records - 1) / n_records) * np.log(n_records / (n_records - 1)))
            else:
                sensitivity = (2 / n_records) * np.log((n_records + 1) / 2) + \
                              (((n_records - 1) / n_records) * np.log((n_records + 1) / (n_records - 1)))

            scaling_factors[idx] = self._n_nodes_dp_computed * sensitivity / (self.epsilon * self.epsilon_split)
        return scaling_factors
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

        # np to df with original column ordering
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


if __name__ == "__main__":
    data_path = '../../examples/data/original/adult.csv'
    df = pd.read_csv(data_path, delimiter=', ', engine='python')
    columns = ['age', 'sex', 'education', 'workclass', 'income']
    df = df.loc[:, columns]
    print(df.head())

    pb = PrivBayes()
    pb.fit(df)
    df_synth = pb.sample()
    pb.score(df, df_synth, score_dict=True)

    """test pb with init network"""
    init_network = [NodeParentPair(node='age', parents=None),
                    NodeParentPair(node='education', parents=('age',)),
                    NodeParentPair(node='sex', parents=('age', 'education'))]

    pbinit = PrivBayes()
    pbinit.set_network(init_network)
    pbinit.fit(df)
    df_synth_init = pbinit.sample(1000)
