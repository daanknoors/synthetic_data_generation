"""
A meta-algorithm for synthetic data generation that combines preprocessing and linked
synthesis methods in one class
"""

import numpy as np
import pandas as pd
from pyhere import here
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from itertools import combinations
from collections import namedtuple
from copy import deepcopy

import warnings
warnings.filterwarnings('ignore')

import synthesis.tools.dp_utils as dp_utils
import synthesis.tools.utils as utils
from synthesis.hist_synthesis import HistSynthesizer
from synthesis.bayes_synthesis import PrivBayes, PrivBayesFix, NodeParentPair
from synthesis.preprocessing.discretization import GeneralizeCategorical, GeneralizeContinuous, get_high_cardinality_features
from synthesis.preprocessing.date_sequence import GeneralizeDateSequence
from thomas.core.cpt import CPT
from thomas.core.factor import Factor
from thomas.core.bayesiannetwork import BayesianNetwork


class MetaSynthesizer(BaseEstimator, TransformerMixin):

    def __init__(self,  epsilon: float = 1.0,
                 variable_group_order=None, categorical_vars=None, continuous_vars=None,
                 date_sequence_vars=None, n_records_synth=None, **kwargs):
        self.epsilon = epsilon
        # self.synthesis_method = synthesis_method #todo allow list of methods
        self.variable_group_order = variable_group_order
        self.categorical_vars = categorical_vars
        self.continuous_vars = continuous_vars
        self.date_sequence_vars = date_sequence_vars
        self.n_records_synth = n_records_synth

    def fit(self, X, y=None):
        """Process:
        1. determine epsilon allocation
        2. fit_transform of cat, cont, date
        3. fit synthesis algorithm on variable groupings - identify linking variables
        """
        self._check_init(X)
        X = self._check_input_data(X)
        # converts to dataframe in case of numpy input and make all columns categorical.
        X = pd.DataFrame(X).astype(str)

        local_epsilon = self._distribute_epsilon()

        self._fit_discretizers(X, local_epsilon)
        Xt = self._transform_discretizers(X)

        self._fit_synthesis(Xt, local_epsilon)

    def transform(self, X):
        Xt = self._transform_discretizers(X)
        Xs = self._transform_synthesis(Xt)
        Xs_inv = self._inverse_discretizers(Xs)
        return Xs_inv

    def _transform_synthesis(self, Xt):
        Xs_dict = {}

        for i in range(len(self.fitted_synthesizers_)):
            synthesizer = self.fitted_synthesizers_[i]
            feature_group = self.variable_group_order[i]
            linking_variable = self.linking_variables_[i]

            if linking_variable:
                assert synthesizer.__class__.__name__ == "PrivBayesFix", "needs PrivBayesFix in order to fix columns"
                # todo fix need to make dataframe
                fix_column = pd.DataFrame(Xs_dict[linking_variable].values(), columns=[linking_variable])
                synthesizer.set_fixed_columns(fix_column)

            Xs_featuregroup = synthesizer.transform(Xt[feature_group])

            Xs_dict = {**Xs_dict, **Xs_featuregroup.to_dict()}
        Xs = pd.DataFrame(Xs_dict)
        return Xs

    def _inverse_discretizers(self, Xs):
        Xs_inv = Xs.copy()

        for discetizer, features in zip(self.discretizers_, self.discretized_features_):
            Xs_inv[features] = discetizer.inverse_transform(Xs_inv[features])
        return Xs_inv

    def _check_init(self, X):
        self.categorical_vars = list(set(self.categorical_vars)) if self.categorical_vars is not None else None
        self.continuous_vars = list(set(self.continuous_vars)) if self.continuous_vars is not None else None

        self.fitted_columns_ = set()
        self.linking_variables_ = []
        # todo fix the identification of linking variables
        if self.variable_group_order is not None:
            for group in self.variable_group_order:
                linking_variable = list(self.fitted_columns_.intersection(set(group)))
                assert linking_variable is None or len(linking_variable) < 2, "can only have 1 overlapping (linking) column between groups"
                linking_variable = list(linking_variable)[0] if bool(linking_variable) else None

                self.linking_variables_.append(linking_variable)
                self.fitted_columns_.update(set(group))
        else:
            self.variable_group_order = [list(X.columns)]
            self.linking_variables_.append(None)
        return self

    def _check_input_data(self, X):
        # converts to dataframe in case of numpy input and make all columns categorical.
        X = pd.DataFrame(X).astype(str, copy=False)
        assert X.shape[1] > 1, "input needs at least 2 columns"
        # prevent integer column indexing issues
        X.columns = X.columns.astype(str)
        if hasattr(self, '_header'):
            assert set(X.columns) == set(self._header), "input contains different columns than seen in fit"
        else:
            self._header = list(X.columns)

        return X

    def _distribute_epsilon(self):
        total_epsilon = self.epsilon

        n_synthesis_algorithms = len(self.variable_group_order) if self.variable_group_order is not None else 1
        dp_categorical = self.categorical_vars is not None
        n_dp_algorithms = n_synthesis_algorithms + dp_categorical
        local_epsilon = total_epsilon / n_dp_algorithms

        print("Total number of algorithms that access data: {}, "
              "thus total epsilon is evenly split {} / {} = {} per algorithm".format(n_dp_algorithms, total_epsilon,
                                                                               n_dp_algorithms, local_epsilon))
        return local_epsilon

    def _fit_discretizers(self, X, local_epsilon):
        # only generalize categorical according with number of unique values above threshold
        generalize_categorical_columns = get_high_cardinality_features(X[self.categorical_vars], threshold=50)
        # remove linking variables from discretization
        generalize_categorical_columns = [i for i in generalize_categorical_columns if i not in self.linking_variables_]

        # todo exclude linking variable from generalize cat

        self.discretizers_ = []
        self.discretized_features_ = []
        Xt = X.copy()

        if generalize_categorical_columns:
            gen_cat = GeneralizeCategorical(epsilon=local_epsilon).fit(Xt[generalize_categorical_columns])
            self.discretizers_.append(gen_cat)
            self.discretized_features_.append(generalize_categorical_columns)

        if self.continuous_vars:
            gen_cont = GeneralizeContinuous().fit(Xt[self.continuous_vars])
            self.discretizers_.append(gen_cont)
            self.discretized_features_.append(self.continuous_vars)

        if self.date_sequence_vars:
            gen_date = GeneralizeDateSequence(date_sequence=self.date_sequence_vars).fit(Xt[self.date_sequence_vars])
            self.discretizers_.append(gen_date)
            self.discretized_features_.append(self.date_sequence_vars)
        return self

    def _transform_discretizers(self, X):
        Xt = X.copy()

        for discetizer, features in zip(self.discretizers_, self.discretized_features_):
            Xt[features] = discetizer.transform(Xt[features])
        return Xt

    # def _preprocess_features(self, X, local_epsilon):
    #     # only generalize categorical according with number of unique values above threshold
    #     generalize_categorical_columns = get_high_cardinality_features(X[self.categorical_vars], threshold=50)
    #     # discretizer = ColumnTransformer([
    #     #     ('generalize_categorical', GeneralizeCategorical(epsilon=local_epsilon), generalize_categorical_columns),
    #     #     ('generalize_continuous', GeneralizeContinuous(), self.continuous_vars),
    #     #     ('generalize_datesequence', GeneralizeDateSequence(), self.date_sequence_Vars)
    #     # ])
    #     self.discretizers_= []
    #     self.discretized_features_ = []
    #     Xt = X.copy()
    #
    #     if generalize_categorical_columns:
    #         gen_cat = GeneralizeCategorical(epsilon=local_epsilon).fit(Xt[generalize_categorical_columns])
    #         # Xt[generalize_categorical_columns] = gen_cat.transform(Xt[generalize_categorical_columns])
    #         self.discretizers_.append(gen_cat)
    #         self.discretized_features_.append(generalize_categorical_columns)
    #
    #     if self.continuous_vars:
    #         gen_cont = GeneralizeContinuous().fit(Xt[self.continuous_vars])
    #         # Xt[self.continuous_vars] = gen_cont.transform(Xt[self.continuous_vars])
    #         self.discretizers_.append(gen_cont)
    #         self.discretized_features_.append(self.continuous_vars)
    #
    #
    #     if self.date_sequence_vars:
    #         gen_date = GeneralizeDateSequence(date_sequence=self.date_sequence_Vars).fit(Xt[self.date_sequence_Vars])
    #         # Xt[self.date_sequence_Vars] = gen_date.transform(Xt[self.date_sequence_Vars])
    #         self.discretizers_.append(gen_date)
    #         self.discretized_features_.append(self.date_sequence_vars)
    #
    #
    #     return self


    def _fit_synthesis(self, Xt, local_epsilon):
        self.fitted_synthesizers_ = []
        fitted_columns = set()
        for i in range(len(self.variable_group_order)):
            group = self.variable_group_order[i]
            prefit_col = self.linking_variables_[i]

            if not prefit_col:
                synthesizer = PrivBayes(epsilon=local_epsilon, n_records_synth=self.n_records_synth)
                synthesizer.fit(Xt[group])
            else:
                print('Fixing column: {}'.format(prefit_col))
                network_init = [NodeParentPair(node=prefit_col, parents=None)]
                synthesizer = PrivBayesFix(epsilon=local_epsilon, network_init=network_init)
                synthesizer.fit(Xt[group])
            self.fitted_synthesizers_.append(synthesizer)

        return self



