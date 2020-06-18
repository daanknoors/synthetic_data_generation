"""
Synthetic Data Generation using a Bayesian Network

Based on following paper

Zhang J, Cormode G, Procopiuc CM, Srivastava D, Xiao X.
PrivBayes: Private Data Release via Bayesian Networks. (2017)
"""

import numpy as np
import pandas as pd
from pyhere import here
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.metrics import mutual_info_score
from itertools import combinations
from collections import namedtuple
from copy import deepcopy

import warnings
warnings.filterwarnings('ignore')

import synthesis.tools.dp_utils as dp_utils
from thomas.core.cpt import CPT
from thomas.core.factor import Factor
from thomas.core.bayesiannetwork import BayesianNetwork

NodeParentPair = namedtuple('NodeParentPair', ['node', 'parents'])
score_functions = {
    'mi': 'mutual_information',
    'F': 'F_score', # note requires binary encoded data
    'R': 'R_score'
}


class PrivBayes(BaseEstimator, TransformerMixin):
    """PrivBayes: generate data based on DP Bayesian Network"""

    def __init__(self, epsilon: float = 1.0, degree_network=None,
                 theta_usefulness=4, score_function='mi', random_state=None,
                 epsilon_split=None):
        self.epsilon = epsilon
        self.k = degree_network
        self.theta = theta_usefulness
        self.score_function = score_function.lower()
        self.random_state = random_state
        if epsilon_split is None:
            self.epsilon_split = [0.4, 0.6]
        else:
            if isinstance(epsilon_split, float):
                epsilon_split = [epsilon_split, 1-epsilon_split]
            self.epsilon_split = list(epsilon_split)

    def fit(self, X, y=None):
        self._n_records, self._n_columns = X.shape
        # X = X.loc[columns] if columns is not None else X
        self.random_state_ = check_random_state(self.random_state)
        # todo integrate random state
        self._greedy_bayes(X)
        self._compute_conditional_distributions(X)
        # todo specify name in init?
        self.bayesian_network_ = BayesianNetwork.from_CPTs('PrivBayes', self.cpt_.values())
        return self

    def transform(self, X, n_records=None):
        n_records = self._n_records if n_records is None else n_records

        Xt = self._generate_data(X, n_records)
        return Xt

    def _greedy_bayes(self, X):
        X = X.astype(str, copy=False)
        n_records, n_columns = X.shape

        if not self.k:
            self.k = self._compute_degree_network(n_records, n_columns)
        print("1/{} - Degree of network (k): {}\n".format(n_columns, self.k))

        nodes, network, nodes_selected = self._init_network(X)

        while len(nodes_selected) < len(nodes):
            print("{}/{} - Evaluating next node to add to network".format(i+1, n_columns))

            nodes_remaining = nodes - nodes_selected
            n_parents = min(self.k, len(nodes_selected))

            node_parent_pairs = [
                NodeParentPair(n, tuple(p)) for n in nodes_remaining
                for p in combinations(nodes_selected, n_parents)
            ]

            scores = self._compute_scores(X, node_parent_pairs)

            if self.epsilon:
                sampled_pair = self._exponential_mechanism(X, node_parent_pairs, scores)
            else:
                sampled_pair = node_parent_pairs.index(max(scores))
            print("Sampled node: {} - with parents: {}\n".format(sampled_pair.node, sampled_pair.parents))
            nodes_selected.add(sampled_pair.node)
            network.append(sampled_pair)
        print("Learned Bayesian Network\n")
        self.network_ = network
        return self

    def _init_network(self, X):

        if self.score_function == 'mi':
            # identify binary columns for calculating sensitivity
            self._binary_columns = [c for c in X.columns if X[c].unique().size <=2]
        elif self.score_function == 'f':
            # todo encode data -> or make assertion and let user binary encode before callling
            pass
        elif self.score_function == 'r':
            # todo
            pass
        else:
            raise ValueError("Score function unknown select from {mi, f, r} - see PrivBayes paper")
        nodes = set(X.columns)
        network = []
        nodes_selected = set()

        root = np.random.choice(tuple(nodes))
        network.append(NodeParentPair(node=root, parents=None))
        nodes_selected.add(root)
        print("Root of network: {}".format(root))

        return nodes, network, nodes_selected
                    
    def _compute_scores(self, X, node_parent_pairs):
        cached_scores = self._get_cached_scores(node_parent_pairs)
        # todo fix cache_scores
        scores = np.empty(len(node_parent_pairs))
        for idx, pair in enumerate(node_parent_pairs):
            if self.score_function == 'mi':
                scores[idx] = self._compute_mutual_information(X, pair)
        return scores


    def _get_cached_scores(self, node_parent_pairs):
        return []

    def _compute_mutual_information(self, X, pair):
        p_node = Factor(X.groupby(pair.node).size()).normalize()
        p_parents = Factor(X.groupby(list(pair.parents)).size()).normalize()
        p_nodeparents = Factor(X.groupby([*pair.parents, pair.node]).size()).normalize()

        # todo: have to get values from Factor: 'numpy.ndarray' object has no attribute '_data'
        mi = np.sum(p_nodeparents.values * np.log(p_nodeparents/(p_node*p_parents)))
        mi_old = self._compute_mutual_information_old(X, pair)
        if mi != mi_old:
            print('New mi: {} - old mi: {}'.format(mi, mi_old))
        return mi



    def _compute_mutual_information_old(self, X, pair):
        df_node = X[pair.node].values
        if len(pair.parents) == 1:
            df_parent = X[pair.parents[0]].values
        else:
            # todo find alternative method to combine parent cols
            df_parent = X.loc[:, pair.parents].apply(lambda x: ' '.join(x.values), axis=1).values
        return mutual_info_score(df_node, df_parent)

    def _exponential_mechanism(self, X, node_parent_pairs, scores):
        scaling_factors = self._compute_scaling_factor(X, node_parent_pairs)
        sampling_distribution = np.exp(scores / 2*scaling_factors)
        normalized_sampling_distribution = sampling_distribution / sampling_distribution.sum()
        pair_idx = np.arange(len(node_parent_pairs))
        sampled_pair_idx = np.random.choice(pair_idx, p=normalized_sampling_distribution)
        sampled_pair = node_parent_pairs[sampled_pair_idx]
        return sampled_pair

    def _compute_scaling_factor(self, X, node_parent_pairs):
        n_records, n_columns = X.shape
        scaling_factors = np.empty(len(node_parent_pairs))
        if self.score_function == 'mi':
            for idx, pair in enumerate(node_parent_pairs):
                if pair.node in self._binary_columns or \
                        (len(pair.parents) == 1 and pair.parents[0] in self._binary_columns):
                    sensitivity = (np.log(n_records) / n_records) + \
                                  (((n_records-1)/n_records) * np.log(n_records/(n_records-1)))
                else:
                    sensitivity = (2/n_records)*np.log((n_records+1)/2) + \
                                  (((n_records-1)/n_records) * np.log((n_records+1)/(n_records-1)))

                scaling_factors[idx] = (n_columns - 1) * sensitivity / (self.epsilon*self.epsilon_split[0])
        return scaling_factors


    def _compute_degree_network(self, n_records, n_columns):
        """
        Determine the degree of the network (k) by finding the lowest integer k possible that ensures that the defined
        level of theta-usefulness is met. This criterion measures the ratio of information over noise.
        Lemma 4.8 in the paper. 

        Note there are some inconsistencies between the original paper from 2014 and the updated version in 2017
        - avg_scale_info: full epsilon in paper 2014 | epsilon_2 in paper2017
        - avg_scale_noise: k + 3 in paper 2014 lemma 3 | k + 2 in paper 2017 lemma  4.8
        """
        k = n_columns - 1

        while k > 1:
            # avg_scale_info = self.epsilon * (1 - self.epsilon_split[0]) * n_records
            avg_scale_info = self.epsilon * self.epsilon_split[1] * n_records
            avg_scale_noise = (n_columns - k) * (2 ** (k + 2))
            if (avg_scale_info / avg_scale_noise) >= self.theta:
                break
            k -= 1
        return k

    def _compute_conditional_distributions(self, X):
        P = dict()

        # first materialize noisy distributions for nodes who have a equal number of parents to the degree k.
        # earlier nodes can be inferred from these distributions without adding extra noise
        for idx, pair in enumerate(self.network_[self.k:]):
            print(pair)
            local_epsilon = self.epsilon * self.epsilon_split[1] / (X.shape[1] - self.k)

            attributes = [*pair.parents, pair.node]
            dp_joint_distribution = dp_utils.dp_joint_distribution(X[attributes], epsilon=local_epsilon)
            # todo: without conversion to pd.series
            cpt = CPT(dp_joint_distribution, conditioned_variables=[pair.node]).normalize()
            P[pair.node] = cpt
            # todo: how to store cpt ins bayesian network?
            # retain noisy joint distribution from k+1 node to infer distributions parent nodes
            if idx == 0:
                infer_from_distribution = Factor(dp_joint_distribution)
                infer_from_distribution = infer_from_distribution.sum_out(pair.node)

        # for pair in self.network_[:self.k]:

        # go iteratively from node at k to root of network, sum out child nodes and get cpt.
        for pair in reversed(self.network_[:self.k]):
            # infer_from_distribution = infer_from_distribution.sum_out(pair.node)
            # conditioned_var = pair.parents[-1]
            cpt = CPT(infer_from_distribution, conditioned_variables=[pair.node]).normalize()
            P[pair.node] = cpt
            infer_from_distribution = infer_from_distribution.sum_out(pair.node)

        self.cpt_ = P
        return self

    def _generate_data(self, X, n_records):
        Xt = np.empty([n_records, X.shape[1]], dtype=object)

        for i in range(n_records):
            print('Number of records generated: {} / {}'.format(i, n_records), end='\r')
            record = self._sample_record()
            Xt[i] = list(record.values())

        # np to df with original column ordering
        df_synth = pd.DataFrame(Xt, columns=[c.node for c in self.network_])[X.columns]
        return df_synth

    def _sample_record(self):
        record = {}
        for col_idx, pair in enumerate(self.network_):
            node = self.bayesian_network_[pair.node]
            # todo filter cpt based on sampled parents

            # specify pre-sampled conditioning values
            node_cpt = node.cpt
            for parent in node.conditioning:
                parent_value = record[parent]
                node_cpt = node_cpt[parent_value]

                # try:
                #     node_cpt = node_cpt[parent_value]
                # except:
                #     print(record)
                #     print(node)
                #     print(node_cpt)
                #     print("parent: {} - {}".format(parent, parent_value))
                #     print('----')

            sampled_node_value = np.random.choice(node.states, p=node_cpt)

            record[node.name] = sampled_node_value

        return record


if __name__ == "__main__":
    data_path = here("examples/data/input/adult.csv")
    df = pd.read_csv(data_path, delimiter=', ').astype(str)
    columns = ['age', 'sex', 'education', 'workclass', 'income']
    df = df.loc[:, columns]
    print(df.head())

    # epsilon = float(np.inf)
    epsilon = 1
    pb = PrivBayes(epsilon=epsilon, degree_network=2)
    pb.fit(df)
    print(pb.network_)
    print("Succesfull")

    pb_copy = deepcopy(pb)

    df_synth = pb.transform(df)
    df_synth.head()