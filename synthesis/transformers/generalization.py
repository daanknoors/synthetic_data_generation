"""A module for pre/postprocessing dates
"""

import numpy as np
import pandas as pd
import random
import pandas_flavor as pf
import warnings
from collections import defaultdict
from scipy.stats import truncnorm

from sklearn.utils.validation import check_array, FLOAT_DTYPES
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from synthesis.transformers._base import BaseReversibleTransformer
from synthesis.synthesizers.utils import dp_marginal_distribution, _normalize_distribution
from synthesis.evaluation import visual

@pf.register_dataframe_method
def bin_numeric_column(df: pd.DataFrame, column_name: str, n_bins: int, col_min: int, col_max: int, strategy: str = 'uniform'):
    """Bin a numeric column according to the number of bins. Can either use staregy 'uniform'
    for obtaining equal-width bins, or strategy 'quanitle' for bins with equal number of
    observations

    Preserves NaN values. Inspired by sklearn KBinsDiscretizer

    Manually add column min and max. As inference from the data can reveal individual values."""

    if strategy not in ['uniform', 'quantile']:
        raise ValueError("Strategy must either be 'uniform' or 'quantile'")

    df = df.copy()

    # exclude nan values
    mask_missing = df[column_name].isna()
    column = df.loc[~mask_missing, column_name][~mask_missing]

    # create equal-width bins
    if strategy == 'uniform':
        bin_edges = np.linspace(col_min, col_max, n_bins + 1)

    # create bins with equal number of observations
    elif strategy == 'quantile':
        quantiles = np.linspace(0, 100, n_bins + 1)
        column_values = np.append(column.values, [col_min, col_max])
        bin_edges = np.asarray(np.percentile(column_values, quantiles))

        # remove bins with too small width
        mask = np.ediff1d(bin_edges, to_begin=np.inf) > 1e-8
        bin_edges = bin_edges[mask]
        if len(bin_edges) - 1 != n_bins:
            warnings.warn('Bins whose width are too small (i.e., <= '
                          '1e-8). Consider decreasing the number of bins.')
            n_bins = len(bin_edges) - 1

    # Values which are close to a bin edge are susceptible to numeric instability
    # Add eps to X so these values are binned correctly with respect to their decimal truncation
    # eps = 1.e-8 + 1.e-5 * np.abs(column)
    # column_name = np.digitize(column_name + eps, bin_edges[1:])
    column = np.digitize(column, bin_edges[1:])

    column = np.clip(column, 0, n_bins - 1)

    # round to 2 decimals to prevent long numbers
    lower_bound = np.round(bin_edges[np.int_(column)], 2)
    upper_bound = np.round(bin_edges[np.int_(column) + 1], 2)

    # transform column to IntervalArray
    df.loc[~mask_missing, column_name] = pd.arrays.IntervalArray.from_arrays(lower_bound, upper_bound, closed='left')
    # force Interval dtype for whole column_name
    df[column_name] = pd.arrays.IntervalArray(df[column_name])
    return df


@pf.register_dataframe_method
def sample_from_binned_column(df: pd.DataFrame, column_name: str, numeric_type='int', mean=None, std=None):
    """Reverse the binning of a numeric column, by sampling from the bin ranges. If mean and standard
    deviation of column are given, then sample from truncated normal, else take the uniform distribution"""
    if numeric_type not in ['int', 'float']:
        raise ValueError("Numeric type must be 'int' or 'float'")
    df = df.copy()
    mask_missing = df[column_name].isna()

    try:
        column = pd.arrays.IntervalArray(df[column_name][~mask_missing])
    except TypeError:
        raise ValueError("Numeric column needs to be converted to bins first. "
                         "First run .bin_numeric_column on a continuous column")

    lower_bound, upper_bound = column.left, column.right

    # sample from truncated normal distribution
    if mean and std:
        low = (lower_bound - mean) / std
        upp = (upper_bound - mean) / std
        X = truncnorm(low, upp, loc=mean, scale=std)

        sampled_values = X.rvs()
        if numeric_type == 'int':
            sampled_values = np.around(sampled_values, decimals=0).astype(int)
        df.loc[~mask_missing, column_name] = sampled_values

    # sample from uniform distribution
    else:
        if numeric_type == 'int':
            df.loc[~mask_missing, column_name] = np.random.randint(lower_bound, upper_bound)
        elif numeric_type == 'float':
            df.loc[~mask_missing, column_name] = np.random.uniform(lower_bound, upper_bound)
    return df

@pf.register_dataframe_method
def sample_from_reversed_dict(df: pd.DataFrame, **generalize_dict):
    """reverse generalization schema and sample unique values.
    dict structure: {column_name: {value: generalized_value}}"""
    for column_name, mapper in generalize_dict.items():
        df = _sample_from_reversed_dict(df, column_name, mapper)
    return df

def _sample_from_reversed_dict(df: pd.DataFrame, column_name, mapper):

    df = df.copy()

    # reverse schema
    reversed_dict = defaultdict(list)
    for key, value in mapper.items():
        reversed_dict[value].append(key)

    # sample from values
    column_generalized = df[column_name].values
    column_sampled = np.empty_like(column_generalized)
    for i in range(len(column_sampled)):
        reverse_candidates = reversed_dict[column_generalized[i]]

        if reverse_candidates:
            column_sampled[i] = random.choice(reverse_candidates)
        else:
            column_sampled[i] = column_generalized[i]
    df[column_name] = column_sampled

    # convert string 'nan' to NaN which automatically converts column to float if all other elements are numeric
    df[column_name] = df[column_name].replace({'nan': np.nan})
    return df

@pf.register_dataframe_method
def replace_rare_values(df, column, min_support=0.01, new_name='other'):
    """Replace values that occur less than minimum support with new name"""
    df = df.copy()

    # normalize if floating number, use counts if int
    if isinstance(min_support, float):
        normalize = True
        n_records_support = np.ceil(min_support * df.shape[0])
    else:
        normalize = False
        n_records_support = min_support

    column_value_counts = df[column].value_counts(normalize=normalize, dropna=False)

    # if no rare categories under min support - no change required
    if not (column_value_counts < min_support).any():
        return df

    rare_categories = list(column_value_counts[column_value_counts < min_support].index)

    # if mode then replace rare categories with the most frequent value in column (including NaN)
    if new_name == 'mode':
        new_name = df[column].mode(dropna=False)[0]

    print(f"Categories in column '{column}' that occur less than {n_records_support}x "
          f"and are replaced by {new_name} - converted categories: {rare_categories}")

    map_value_probs_to_rows = df[column].map(column_value_counts)
    convert_low_freq_rows = df[column].mask(map_value_probs_to_rows < min_support, new_name)
    df[column] = convert_low_freq_rows
    return df

class GeneralizeContinuous(KBinsDiscretizer):

    def __init__(self, n_bins=10, strategy='uniform', labeled_missing=None):
        super().__init__(n_bins=n_bins, strategy=strategy, encode='ordinal')
        # self.n_bins = n_bins
        # self.strategy = strategy

        self.labeled_missing = labeled_missing

    def fit(self, X, y=None):
        """
        Fit the estimator.
        Parameters
        ----------
        X : numeric array-like, shape (n_samples, n_features)
            Data to be discretized.
        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`sklearn.pipeline.Pipeline`.
        Returns
        -------
        self
        """
        self._infer_numerical_type(X)
        self._header = X.columns

        # X = self._validate_data(X, dtype='numeric')
        X = check_array(X, dtype='numeric', force_all_finite='allow-nan')

        valid_strategy = ('uniform', 'quantile')
        if self.strategy not in valid_strategy:
            raise ValueError("Valid options for 'strategy' are {}. "
                             "Got strategy={!r} instead."
                             .format(valid_strategy, self.strategy))

        n_features = X.shape[1]
        n_bins = self._validate_n_bins(n_features)

        bin_edges = np.zeros(n_features, dtype=object)
        for jj in range(n_features):
            column = X[:, jj]
            missing_idx = self._get_missing_idx(column)
            column = column[~missing_idx]
            col_min, col_max = column.min(), column.max()

            if col_min == col_max:
                warnings.warn("Feature %d is constant and will be "
                              "replaced with 0." % jj)
                n_bins[jj] = 1
                bin_edges[jj] = np.array([-np.inf, np.inf])
                continue

            if self.strategy == 'uniform':
                bin_edges[jj] = np.linspace(col_min, col_max, n_bins[jj] + 1)

            elif self.strategy == 'quantile':
                quantiles = np.linspace(0, 100, n_bins[jj] + 1)
                bin_edges[jj] = np.asarray(np.percentile(column, quantiles))


            # Remove bins whose width are too small (i.e., <= 1e-8)
            if self.strategy == 'quantile':
                mask = np.ediff1d(bin_edges[jj], to_begin=np.inf) > 1e-8
                bin_edges[jj] = bin_edges[jj][mask]
                if len(bin_edges[jj]) - 1 != n_bins[jj]:
                    warnings.warn('Bins whose width are too small (i.e., <= '
                                  '1e-8) in feature %d are removed. Consider '
                                  'decreasing the number of bins.' % jj)
                    n_bins[jj] = len(bin_edges[jj]) - 1

        self.bin_edges_ = bin_edges
        self.n_bins_ = n_bins

        return self

    def transform(self, X):
        """
        Discretize the data.
        Parameters
        ----------
        X : numeric array-like, shape (n_samples, n_features)
            Data to be discretized.
        Returns
        -------
        Xt : numeric array-like or sparse matrix
            Data in the binned space.
        """
        # check_is_fitted(self, attributes=None)

        Xt = check_array(X, copy=True, dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
        n_features = self.n_bins_.shape[0]
        if Xt.shape[1] != n_features:
            raise ValueError("Incorrect number of features. Expecting {}, "
                             "received {}.".format(n_features, Xt.shape[1]))

        bin_edges = self.bin_edges_
        for jj in range(Xt.shape[1]):
            missing_idx = self._get_missing_idx(Xt[:, jj])

            # Values which are close to a bin edge are susceptible to numeric
            # instability. Add eps to X so these values are binned correctly
            # with respect to their decimal truncation. See documentation of
            # numpy.isclose for an explanation of ``rtol`` and ``atol``.
            rtol = 1.e-5
            atol = 1.e-8
            eps = atol + rtol * np.abs(Xt[~missing_idx, jj])
            Xt[~missing_idx, jj] = np.digitize(Xt[~missing_idx, jj] + eps, bin_edges[jj][1:])
        np.clip(Xt, 0, self.n_bins_ - 1, out=Xt)
        return pd.DataFrame(Xt, columns=self._header)

    def _infer_numerical_type(self, X):
        """Determine if numerical column is an integer of float for inverse transform"""
        # assert X.select_dtypes(exclude=['int', 'float']).shape[1] == 0, "input X contains non-numeric columns"
        self.integer_columns = []
        self.float_columns = []

        for c in X.columns:
            only_integers = (X[c].dropna().astype(float) % 1 == 0).all()
            if only_integers:
                self.integer_columns.append(c)
            else:
                self.float_columns.append(c)
            # conversion from string to int directly can cause ValueError
            # if np.array_equal(X[c].dropna(), X[c].dropna().astype(float).astype(int)):
            #     self.integer_columns.append(c)
            # else:
            #     self.float_columns.append(c)

    def _get_missing_idx(self, column):
        return np.isnan(column) | np.isin(column, self.labeled_missing)


    def inverse_transform(self, Xt):
        # check_is_fitted(self)
        assert set(Xt.columns) == set(self._header), "input contains different columns than seen in fit"

        Xinv = check_array(Xt, copy=True, dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
        # Xinv = Xt.copy()
        n_features = self.n_bins_.shape[0]
        if Xinv.shape[1] != n_features:
            raise ValueError("Incorrect number of features. Expecting {}, "
                             "received {}.".format(n_features, Xinv.shape[1]))

        for jj in range(n_features):
            bin_edges = self.bin_edges_[jj]
            missing_idx = self._get_missing_idx(Xinv[:, jj])

            lower_bounds = bin_edges[np.int_(Xinv[~missing_idx, jj])]
            upper_bounds = bin_edges[np.int_(Xinv[~missing_idx, jj]) + 1]
            Xinv[~missing_idx, jj] = np.random.uniform(lower_bounds, upper_bounds)

            # todo transfer to numpy
            if self._header[jj] in self.integer_columns:
                Xinv[~missing_idx, jj] = np.round(Xinv[~missing_idx, jj])

        return pd.DataFrame(Xinv, columns=self._header)



class GeneralizeCategorical(GeneralizeContinuous):

    def __init__(self, epsilon=1.0, n_bins=5, strategy='uniform', max_cardinality=10):
        super().__init__(n_bins=n_bins, strategy=strategy)
        self.epsilon = epsilon
        self.max_cardinality = max_cardinality


    def fit(self, X, y=None):
        """ Steps:
        1. Transform categorical to continuous
        2. Store DP marginal counts for optional inverse transform
        3. Run super().fit() to get groups
        """
        X = X.astype(str).fillna('missing')
        self._ordinalencoder = OrdinalEncoder().fit(X)
        #todo: turn into numpy -> df needed for marginal distribution
        X_enc = self._ordinalencoder.transform(X)
        X_enc = pd.DataFrame(X_enc, columns=X.columns)

        # get dp marginal of encoded feature
        # todo turn into list of arrays
        local_epsilon = self.epsilon / X.shape[1]
        self.marginals_ = []
        for jj, c in enumerate(X.columns):
            self.marginals_.append(dp_marginal_distribution(X_enc.loc[:, c], local_epsilon).values)

        return super().fit(X_enc, y)

    def transform(self, X):
        """Equivalent to continuous transform but we still need to encode the data beforehand"""
        X = X.astype(str).fillna('missing')
        X_enc = self._ordinalencoder.transform(X)
        return super().transform(X_enc)

    def inverse_transform(self, Xt):
        assert set(Xt.columns) == set(self._header), "input contains different columns than seen in fit"

        X_enc = check_array(Xt, copy=True, dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
        # Xinv = Xt.copy()
        n_records, n_features = X_enc.shape
        if X_enc.shape[1] != n_features:
            raise ValueError("Incorrect number of features. Expecting {}, "
                             "received {}.".format(n_features, X_enc.shape[1]))


        self._marginal_group_alloc = []

        for jj, c in enumerate(Xt.columns):
            bin_edges = self.bin_edges_[jj]
            marginals = self.marginals_[jj]
            marginals_idx = np.arange(len(marginals))

            rtol = 1.e-5
            atol = 1.e-8
            eps = atol + rtol * np.abs(marginals)
            marginal_group_alloc = np.digitize(marginals_idx + eps, bin_edges[1:])
            np.clip(marginal_group_alloc, 0, self.n_bins_[jj] - 1, out=marginal_group_alloc)

            self._marginal_group_alloc.append(marginal_group_alloc)

            # lower_bounds = np.int_(bin_edges[np.int_(X_enc[:, jj])])
            # upper_bounds = np.int_(bin_edges[np.int_(X_enc[:, jj]) + 1])

            for i in range(n_records):
                # Values which are close to a bin edge are susceptible to numeric
                # instability. Add eps to X so these values are binned correctly
                # with respect to their decimal truncation. See documentation of
                # numpy.isclose for an explanation of ``rtol`` and ``atol``.
                # rtol = 1.e-5
                # atol = 1.e-8
                # eps = atol + rtol * np.abs(upper_bounds[i])
                # marginal_candidates = marginals[
                #     (marginals.keys() >= lower_bounds[i]) &
                #     (marginals.keys() < upper_bounds[i] + eps)]

                #np.where returns 1d tuple, thus index 0
                marginal_candidate_idx = np.where(X_enc[i, jj] == marginal_group_alloc)[0]
                marginal_candidate_probs = marginals[marginal_candidate_idx]
                marginal_candidate_probs_normalized = _normalize_distribution(marginal_candidate_probs)


                # marginal_idx = np.arange(lower_bounds[i], upper_bounds[i])
                # marginal_probs = marginals[marginal_idx]

                # marginal_probs_normalized = marginal_probs / marginal_probs.sum()
                # sample encoded (numerical) value based on marginal probabilities
                # print(jj)
                # print(X_enc.shape)
                # print(X_enc[i, jj])
                # print(marginal_candidate_idx)
                X_enc[i, jj] = np.random.choice(marginal_candidate_idx, p=marginal_candidate_probs_normalized)

                # X_enc[i, jj] = np.random.choice(list(marginal_candidates.keys()), p=marginal_candidates_normalized.values)

        # inverse transform numerical value to original categorical
        X_inv = self._ordinalencoder.inverse_transform(X_enc)
        return pd.DataFrame(X_inv, columns=self._header)



def get_high_cardinality_features(X, threshold=50):
    """Get features with more unique values than the specified threshold."""
    high_cardinality_features = []
    for c in X.columns:
        if X[c].nunique() > threshold:
            high_cardinality_features.append(c)
    return high_cardinality_features


class GeneralizeSchematic(TransformerMixin, BaseEstimator):

    def __init__(self, schema_dict, label_unknown=None):
        self.schema_dict = schema_dict
        self.label_unknown = label_unknown

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Replaces all values in col with its generalized form in schema dict"""
        Xt = X.copy()

        # convert categories not present in schema to value given for label_unknown
        if self.label_unknown:
            Xt[~Xt.isin(self.schema_dict.keys())] = self.label_unknown

        return Xt.replace(self.schema_dict)

    def _reverse_schema(self):
        """reverse schema dict with non-unique values"""
        reversed_dict = defaultdict(list)

        for key, value in self.schema_dict.items():
            reversed_dict[value].append(key)
        return reversed_dict

    def inverse_transform(self, X):
        """reverse the schema by sampling from the available candidates"""
        reversed_schema = self._reverse_schema()
        X_generalized = X.values
        X_sampled = np.empty_like(X)
        for i in range(len(X)):
            reverse_candidates = reversed_schema[X_generalized[i]]

            if reverse_candidates:
                X_sampled[i] = np.random.choice(reverse_candidates)
            else:
                X_sampled[i] = X_generalized[i]
        return X_sampled


class GroupRareCategories(BaseReversibleTransformer):
    """Transformer to group rare categories"""

    def __init__(self, threshold=0.05, name_group='Other'):
        self.threshold = threshold
        self.name_group = name_group

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Group categories that occur less then threshold"""
        return X.mask(X.map(X.value_counts(normalize=True)) < self.threshold, self.name_group)

# class GeneralizeCategorical(GeneralizeContinuous):
#
#     def __init__(self, epsilon=1.0, n_bins=10, strategy='uniform', labeled_missing=None):
#         super().__init__(n_bins=n_bins, strategy=strategy, encode='ordinal')
#         self.epsilon = epsilon
#
#     def fit(self, X, y=None):
#         # if user-specified bins in form of iterable or dict
#         if self.bins:
#             pass
#
#         self.marginals_ = {}
#         self._ordinalencoders = {}
#         X_enc = np.empty_like(X)
#         for jj, c in enumerate(X.columns):
#             uniques = sorted(set(X[c]))
#             uniques, counts = X[c].value_counts(dropna=False)
#
#             local_epsilon = self.epsilon / X.shape[1]
#             self.marginals_[c] = dp_marginal_distribution(X[c], local_epsilon)
#
#             # get numeric values - store encoder for inverse transform
#             self._ordinalencoders[c] = OrdinalEncoder().fit(X[c])
#             X_enc[:, jj] = self._ordinalencoders[c].transform(X[c])
#
#
#         return super().fit(X_enc, y)
#
#     def inverse_transform(self, Xt):
#         X_enc = super().inverse_transform(Xt)
#         for jj in range(X_enc.shape[1]):
#             # todo fix column names indexing
#             # todo ensure inverse gives back X_enc which OrdinalEncoder.inverse_transform(X_le)
#             pass

