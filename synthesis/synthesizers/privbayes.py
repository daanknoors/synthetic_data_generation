"""
Synthetic data generation via Bayesian Networks

Based on following paper

Zhang J, Cormode G, Procopiuc CM, Srivastava D, Xiao X.
PrivBayes: Private Data Release via Bayesian Networks. (2017)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

import synthesis.synthesizers.utils as utils
from synthesis.synthesizers._base import BaseDPSynthesizer
from synthesis.transformers.discretization import GeneralizeContinuous, GeneralizeCategorical, HierarchicalEncoder
from thomas.core.bayesiannetwork import BayesianNetwork, DiscreteNetworkNode
from diffprivlib.mechanisms import Exponential

# APPair = namedtuple('AttributeParentPair', ['attribute', 'parents'])

class HierarchicalAttribute:
    """Contains hierarchical attribute information - does not contain data itself"""
    def __init__(self, data, height=1,  hierarchy_level=0, encoder=None):
        self.name = data.squeeze().name
        self.cardinality = data
        self.height = height
        self.hierarchy_level = hierarchy_level
        self.encoder = encoder

    @property
    def cardinality(self):
        return self._cardinality

    @cardinality.setter
    def cardinality(self, data):
        if not len(data.squeeze().shape) == 1:
            raise ValueError("Data can only be 1 dimensional")
        self._cardinality = [utils.cardinality(data)]

    @property
    def hierarchy_level(self):
        return self._hierarchy_level

    @hierarchy_level.setter
    def hierarchy_level(self, level):
        """set hierarchy_level"""
        assert 0 <= level <= self.height, "level must be between 0 and class height"
        self._hierarchy_level = level

    def add_encoder(self, encoder):
        assert isinstance(encoder, HierarchicalEncoder), 'Encoder must be of type HierarchicalEncoder'
        self.encoder = encoder
        self.height += len(encoder.encoding_cardinality)
        self._cardinality.extend(encoder.encoding_cardinality)

    def generalize(self, data):
        if not self.encoder:
            return data
        return self.encoder.transform(data, self.hierarchy_level)

    def __repr__(self):
        msg = f"HierarchicalAttribute(name='{self.name}', hierarchy_level={self.hierarchy_level})"
        return msg


class APPair:
    """AttributeParentPair - candidate pairs for constructing the network"""

    def __init__(self, attribute, parents):
        self.attribute = attribute
        self.parents = parents if not isinstance(parents, str) else (parents,)

    @property
    def attribute_name(self):
        """Get attribute name"""
        return self.attribute.name

    @property
    def parent_names(self):
        parent_names = [p.name for p in self.parents if p.name is not None]
        return parent_names

    @property
    def names(self):
        return self.parent_names + [self.attribute_name]

    @property
    def cardinality(self):
        """Total cardinality of attribute and parents"""
        attribute_cardinality = [self.attribute.cardinality[0]]
        parent_cardinality = [p.cardinality[p.hierarchy_level] for p in self.parents] if self.parents else [1]
        return np.product(attribute_cardinality + parent_cardinality)

    def process_data(self, data):
        """subset data by ap_pair attribute and parents and apply hierarchical encoding if needed"""
        if self.parents is None:
            return data[self.attribute.name]

        # subset and generalize
        subset_columns = self.names
        processed_data = data[subset_columns]
        for parent in self.parents:
            parent_data = processed_data[[parent.name]]
            # in case we process a single row, we need to reshape the data
            # if not parent_data.shape:
            #     parent_data = parent_data.reshape(-1, 1)
            generalized_parent = parent.generalize(parent_data).squeeze()

            # in case we process a single row, we need to index to get the actual value
            # if len(generalized_parent) == 1:
            #     generalized_parent = generalized_parent.values[0]

            processed_data[parent.name] = generalized_parent
        return processed_data

    def __repr__(self):
        attribute = self.attribute
        parents = self.parents
        if isinstance(self.attribute, HierarchicalAttribute):
            attribute = self.attribute.name
        if self.parents:
            parents = [(p.name, p.hierarchy_level) for p in self.parents]
        msg = f"APPair(attribute='{attribute}', parents={parents})"
        return msg


class PrivBayes(BaseDPSynthesizer):
    """PrivBayes: Private Data Release via Bayesian Networks (Zhang et al 2017)

    Version:
    - hierarchical encoding
    - R or mutual information as score function

    Extensions (optional):
    - add encoders
    - initialize network

    Default hyperparameters set according to paper recommendations
    """

    def __init__(self, epsilon=1.0, theta_usefulness=4, epsilon_split=0.3,
                 score_function='R', encoders=None,
                 network_init=None, verbose=True):
        super().__init__(epsilon=epsilon, verbose=verbose)
        self.theta_usefulness = theta_usefulness
        self.epsilon_split = epsilon_split  # also called Beta in paper
        self.score_function = score_function  # choose between 'R' and 'MI'
        # self.hierarchical_encoder = hierarchical_encoder # {column_name: {value: [list of generalized values], ...})
        self.encoders = encoders # dict of encoders {column_name: [Encoder1, ..., EncoderN]}
        self.network_init = network_init

    def fit(self, data):
        data = self._check_input_data(data)
        self._check_init_args()
        data = self._encode_data(data)

        self._greedy_bayes(data)
        self._compute_conditional_distributions(data)
        self.model_ = BayesianNetwork.from_CPTs('PrivBayes', self.cpt_.values())
        return self

    def _encode_data(self, data):
        """applies encoders if specified"""
        if not self.encoders:
            return data
        for column, encoder in self.encoders.items():
            encoder.fit(data[[column]])
            data[column] = encoder.transform(data[[column]])
        return data

    def _decode_data(self, data):
        """reverse encoding scheme if specified"""
        if not self.encoders:
            return data
        for column, encoder in self.encoders.items():
            if hasattr(encoder, 'enable_inverse') and encoder.enable_inverse is True:
                data[column] = encoder.inverse_transform(data[[column]])
        return data

    def _check_init_args(self):
        super()._check_init_args()
        self._check_score_function()

        if not 0 < self.epsilon_split <= 1:
            raise ValueError("Epsilon split must be a value between 0 and 1")

    def _check_score_function(self):
        """Checks input score function and sets sensitivity"""
        if self.score_function.upper() not in ['R', 'MI']:
            raise ValueError("Score function must be 'R' or 'MI'")

        if self.score_function.upper() == 'R':
            self._score_sensitivity = (3 / self.n_records_fit_) + (2 / self.n_records_fit_**2)

        # note: for simplicity we assume that all APPairs are non-binary, which is the upperbound of MI sensitivity
        elif self.score_function.upper() == 'MI':
            self._score_sensitivity = (2 / self.n_records_fit_) * np.log((self.n_records_fit_ + 1) / 2) + \
                              (((self.n_records_fit_ - 1) / self.n_records_fit_) *
                               np.log((self.n_records_fit_ + 1) / (self.n_records_fit_ - 1)))

    def sample(self, n_records=None):
        self._check_is_fitted()
        n_records = n_records or self.n_records_fit_

        synth_data = self._generate_data(n_records)
        synth_data = self._decode_data(synth_data)

        if self.verbose:
            print("\nSynthetic Data Generated\n")
        return synth_data

    def _greedy_bayes(self, data):
        attributes, attributes_selected = self._init_network(data)

        # normally len(attributes) - 1, unless user initialized part of the network
        self._n_attributes_dp_computed = len(attributes) - len(attributes_selected)


        for i in range(len(attributes_selected), len(attributes)):
            if self.verbose:
                print("{}/{} - Evaluating next attribute to add to network".format(i + 1, len(self.columns_)))

            attributes_remaining = attributes - attributes_selected

            # select APPair candidates
            ap_pairs = []
            for attribute in attributes_remaining:
                max_domain_size = self._max_domain_size(attribute)
                # max_parent_sets = self._max_parent_sets(data, attributes_selected, max_domain_size)
                max_parent_sets = self._max_parent_sets(data, attributes_selected, max_domain_size)

                # empty set - domain size of attribute violates theta_usefulness
                if len(max_parent_sets) == 0:
                    ap_pairs.append(APPair(attribute, parents=None))
                # [empty set] - no parents found that meet domain size restrictions
                elif len(max_parent_sets) == 1 and len(max_parent_sets[0]) == 0:
                    ap_pairs.append(APPair(attribute, parents=None))
                else:
                    ap_pairs.extend([
                        APPair(attribute, parents=tuple(p)) for p in max_parent_sets
                    ])
            if self.verbose:
                print("Number of AttributeParentPair candidates: {}".format(len(ap_pairs)))
                print('Candidates: {}'.format(ap_pairs))

            scores = self._compute_scores(data, ap_pairs)
            sampled_pair = self._exponential_mechanism(ap_pairs, scores)

            if self.verbose:
                print("Selected AttributeParentPair: '{}'\n".format(sampled_pair))
            attributes_selected.add(sampled_pair.attribute)
            self.network_.append(sampled_pair)
        if self.verbose:
            print("Learned Network Structure\n")
        return self

    def _max_domain_size(self, attribute):
        """Computes the maximum domain size a attribute can have to satisfy theta-usefulness"""
        attribute_cardinality = attribute.cardinality[0]
        max_domain_size = self.n_records_fit_ * (1 - self.epsilon_split) * self.epsilon / \
                          (2 * len(self.columns_) * self.theta_usefulness * attribute_cardinality)
        return max_domain_size

    # def _max_parent_sets(self, data, v, max_domain_size):
    #     """Refer to algorithm 5 in paper - max parent set is 1) theta-useful and 2) maximal."""
    #     if max_domain_size < 1:
    #         return set()
    #     if len(v) == 0:
    #         return [set()]
    #
    #     x = np.random.choice(tuple(v))
    #     x_domain_size = utils.cardinality(data[x])
    #     x = {x}
    #
    #     v_without_x = v - x
    #
    #     parent_sets1 = self._max_parent_sets(data, v_without_x, max_domain_size)
    #     parent_sets2 = self._max_parent_sets(data, v_without_x, max_domain_size / x_domain_size)
    #
    #     for z in parent_sets2:
    #         if z in parent_sets1:
    #             parent_sets1.remove(z)
    #         parent_sets1.append(z.union(x))
    #     return parent_sets1

    def _max_parent_sets(self, data, v, max_domain_size):
        """Refer to algorithm 6 in paper
        MaxParentSet is 1) theta-useful, 2) maximal, and 3) lowest generalization level"""
        if max_domain_size < 1:
            return set()
        if len(v) == 0:
            return [set()]

        x = np.random.choice(tuple(v))

        # x = {x}
        v_without_x = v - {x}
        s = []
        u = []

        for i in range(x.height):
            parent_sets1 = self._max_parent_sets(data, v_without_x, max_domain_size / x.cardinality[i])
            for z in parent_sets1:
                if z in u:
                    continue
                u.append(z)
                x.hierarchy_level = i
                s.append(z.union({x}))

        parent_sets2 = self._max_parent_sets(data, v_without_x, max_domain_size)
        for z in parent_sets2:
            if z in u:
                continue
            s.append(z)
        return s

    def _create_hierarchical_attributes(self, data):
        """Create hierarchical attributes and add hierarchical information if encoder is specified in init"""
        attributes = set()
        for a in data.columns:
            h_attribute = HierarchicalAttribute(data[a])

            # add hierarchical info if encoder is given in init
            if self.encoders and (a in self.encoders) and isinstance(self.encoders[a], HierarchicalEncoder):
                print(self.encoders[a])
                h_attribute.add_encoder(self.encoders[a])
            attributes.add(h_attribute)
        return attributes

    def _init_network(self, data):
        self._binary_columns = [c for c in data.columns if data[c].unique().size <= 2]

        attributes = self._create_hierarchical_attributes(data)

        #todo evaluate whether network init still works with or without hierarchical attributes
        if self.network_init is not None:
            attributes_selected = [pair.attribute for pair in self.network_init]
            # create hierarchical attributes from APPairs in network init
            if not all(isinstance(a, HierarchicalAttribute) for a in attributes_selected):
                attributes_selected = self._create_hierarchical_attributes(data[attributes_selected])

            for i, pair in enumerate(self.network_init):
                if self.verbose:
                    print("{}/{} - init attribute {} - with parents: {}".format(i + 1, len(self.network_init),
                                                                           pair.attribute, pair.parents))
            self.network_ = self.network_init.copy()
            return attributes, attributes_selected

        # if set_network is not called we start with a random first attribute
        self.network_ = []
        attributes_selected = set()

        root = np.random.choice(tuple(attributes))
        self.network_.append(APPair(attribute=root, parents=None))
        attributes_selected.add(root)
        if self.verbose:
            print("1/{} - Root of network: {}\n".format(data.shape[1], root))
        return attributes, attributes_selected

    def set_network(self, network):
        """define the network initialisation - attributes not part of network init will be DP computed"""
        assert [isinstance(n, APPair) for n in network], "Input network does not consists of APPairs"
        self.network_init = network
        return self

    def _compute_scores(self, data, APPairs):
        """Compute score for all APPairs"""
        scores = np.empty(len(APPairs))

        for idx, pair in enumerate(APPairs):
            if pair.parents is None:
                scores[idx] = 0
                break
            processed_data = pair.process_data(data)
            if self.score_function == 'R':
                scores[idx] = self.r_score(processed_data, pair.attribute_name, pair.parent_names)
            elif self.score_function == 'MI':
                scores[idx] = self.mi_score(processed_data, pair.attribute_name, pair.parent_names)
        return scores


    def _exponential_mechanism(self, ap_pairs, scores):
        """select APPair with exponential mechanism"""
        local_epsilon = self.epsilon * self.epsilon_split / self._n_attributes_dp_computed
        dp_mech = Exponential(epsilon=local_epsilon, sensitivity=self._score_sensitivity,
                              utility=list(scores), candidates=ap_pairs)
        sampled_pair = dp_mech.randomise()
        return sampled_pair

    def _compute_conditional_distributions(self, data):
        self.cpt_ = dict()

        local_epsilon = self.epsilon * (1 - self.epsilon_split) / len(self.columns_)

        for idx, pair in enumerate(self.network_):
            # if pair.parents is None:
            #     attributes = [pair.attribute]
            # else:
            #     attributes = [*pair.parents, pair.attribute]

            # cpt_size = utils.cardinality(data[attributes])

            cpt_size = pair.cardinality
            processed_data = pair.process_data(data)
            if self.verbose:
                print('Learning conditional probabilities: {} ~ estimated size: {}'.format(pair, cpt_size))

            dp_cpt = utils.dp_conditional_distribution(processed_data, epsilon=local_epsilon)
            self.cpt_[pair.attribute.name] = dp_cpt
        return self

    def _generate_data(self, n_records):
        # synth_data = np.empty([n_records, len(self.columns_)], dtype=object)
        synth_data = pd.DataFrame(index=range(n_records), columns=[c.attribute.name for c in self.network_])

        for i in range(n_records):
            if self.verbose:
                print('Number of records generated: {} / {}'.format(i + 1, n_records), end='\r')
            # record = self._sample_record()
            # synth_data[i] = list(record.values())

            # sample a value column for column by conditioning for parents
            for col_idx, pair in enumerate(self.network_):
                attribute_cpt = self.model_[pair.attribute_name].cpt
                attribute_states = self.model_[pair.attribute_name].states

                if pair.parents:
                    # process parents of the record being sampled to ensure correct generalization
                    record = synth_data.iloc[i]
                    processed_record = pair.process_data(record)
                    parent_values = [processed_record[p] for p in pair.parent_names]
                    try:
                        attribute_probs = attribute_cpt[tuple(parent_values)]
                    except:
                        print(parent_values)
                else:
                    attribute_probs = attribute_cpt.values
                sampled_attribute_value = np.random.choice(attribute_states, p=attribute_probs)

                synth_data.loc[i, pair.attribute_name] = sampled_attribute_value

        # numpy.array to pandas.DataFrame with original column ordering
        # synth_data = pd.DataFrame(synth_data, columns=[c.attribute for c in self.network_])[self.columns_]
        # convert to original column ordering
        synth_data = synth_data[self.columns_]
        return synth_data

    # def _sample_record(self):
    #     """samples a value column for column by conditioning for parents"""
    #     record = {}
    #     for col_idx, pair in enumerate(self.network_):
    #         attribute = self.model_[pair.attribute]
    #         attribute_cpt = attribute.cpt
    #         attribute_states = attribute.states
    #
    #         if attribute.conditioning:
    #             parent_values = [record[p] for p in attribute.conditioning]
    #             pair.process_data(record)
    #             attribute_probs = attribute_cpt[tuple(parent_values)]
    #         else:
    #             attribute_probs = attribute_cpt.values
    #         sampled_attribute_value = np.random.choice(attribute_states, p=attribute_probs)
    #
    #         record[attribute.name] = sampled_attribute_value
    #     return record

    @staticmethod
    def mi_score(data, column_names_a, column_names_b):
        column_names_a = utils._ensure_arg_is_list(column_names_a)
        column_names_b = utils._ensure_arg_is_list(column_names_b)

        data_a = data[column_names_a].squeeze()
        if len(column_names_b) == 1:
            data_b = data[column_names_b].squeeze()
        else:
            data_b = data.loc[:, column_names_b].astype(str).apply(lambda x: ' '.join(x.values), axis=1).squeeze()
        return mutual_info_score(data_a, data_b)

    @staticmethod
    def mi_score_thomas(data, column_names_a, colum_names_b):
        column_names_a = utils._ensure_arg_is_list(column_names_a)
        colum_names_b = utils._ensure_arg_is_list(colum_names_b)

        prob_a = utils.compute_distribution(data[column_names_a])
        prob_b = utils.compute_distribution(data[colum_names_b])
        prob_joint = utils.compute_distribution(data[column_names_a + colum_names_b])

        # todo: pull-request thomas to add option for normalizing with remove 0's
        # align
        prob_div = prob_joint / (prob_b * prob_a)
        prob_joint, prob_div = prob_joint.extend_and_reorder(prob_joint, prob_div)

        # remove zeros as this will result in issues with log
        prob_joint = prob_joint.values[prob_joint.values != 0]
        prob_div = prob_div.values[prob_div.values != 0]
        mi = np.sum(prob_joint * np.log(prob_div))
        # mi = np.sum(p_attributeparents.values * np.log(p_attributeparents / (p_parents * p_attribute)))
        return mi


    @staticmethod
    def r_score(data, column_names_a, column_names_b):
        """An alternative score function to mutual information with lower sensitivity - can be used on non-binary domains.
        Relies on the L1 distance from a joint distribution to a joint distributions that minimizes mutual information.
        Refer to Lemma 5.2
        """
        column_names_a = utils._ensure_arg_is_list(column_names_a)
        column_names_b = utils._ensure_arg_is_list(column_names_b)

        # compute distribution that minimizes mutual information
        prob_a = utils.compute_distribution(data[column_names_a])
        prob_b = utils.compute_distribution(data[column_names_b])
        prob_independent = prob_b * prob_a

        # compute joint distribution
        prob_joint = utils.joint_distribution(data[column_names_a + column_names_b])

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

    encoders = {
        'age': HierarchicalEncoder(encoders=[GeneralizeContinuous(n_bins=10)])
    }

    pb = PrivBayes(epsilon=0.1, encoders=encoders)
    pb.fit(data)
    df_synth = pb.sample(1000)
    pb.score(data, df_synth, score_dict=True)

    """test pb with init network"""
    # init_network = [APPair(attribute='age', parents=None),
    #                 APPair(attribute='education', parents=('age',)),
    #                 APPair(attribute='sex', parents=('age', 'education'))]
    #
    # pbinit = PrivBayes()
    # pbinit.set_network(init_network)
    # pbinit.fit(df)
    # df_synth_init = pbinit.sample(1000)


    """test scoring functions"""
    pair = pb.network_[3]
    pb.mi_score(data, pair.attribute_name, pair.parent_names)
    pb.mi_score_thomas(data, pair.attribute_name, pair.parent_names)
    pb.r_score(data, pair.attribute_name, pair.parent_names)



