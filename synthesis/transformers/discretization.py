"""A module for pre/postprocessing dates
"""

import numpy as np
import pandas as pd
import warnings

from sklearn.utils.validation import check_array, FLOAT_DTYPES
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin


from synthesis.synthesizers.utils import dp_marginal_distribution, _normalize_distribution, _ensure_arg_is_list
from synthesis.evaluation import visual


class HierarchicalEncoder(TransformerMixin, BaseEstimator):
    """Fit multiple encoders on one attribute and transform data with desired encoding level"""

    def __init__(self, encoders):
        self.encoders = encoders # list of encoders for one attribute
        self.encoding_cardinality = [e.n_bins for e in self.encoders]

    def fit(self, data):
        for encoder in self.encoders:
            encoder.fit(data)
        return self

    def transform(self, data, level=0):
        assert level <= len(self.encoders), "Level should not exceed the number of encoders available"
        if level == 0:
            return data
        selected_encoder = self.encoders[level-1]
        return selected_encoder.transform(data)

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        msg = f"HierarchicalEncoder(encoders={self.encoders})"
        return msg

class GeneralizeContinuous(KBinsDiscretizer):

    def __init__(self, n_bins=10, strategy='uniform', labeled_missing=None, enable_inverse=True):
        super().__init__(n_bins=n_bins, strategy=strategy, encode='ordinal')
        # self.n_bins = n_bins
        # self.strategy = strategy

        self.labeled_missing = labeled_missing
        self.enable_inverse = enable_inverse

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
        if isinstance(X, pd.Series):
            X = X.to_frame()
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

        if isinstance(X, pd.Series):
            X = X.to_frame()

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

    def __init__(self, epsilon=1.0, n_bins=5, strategy='uniform', max_cardinality=10, enable_inverse=True):
        super().__init__(n_bins=n_bins, strategy=strategy, enable_inverse=enable_inverse)
        self.epsilon = epsilon
        self.max_cardinality = max_cardinality


    def fit(self, X, y=None):
        """ Steps:
        1. Transform categorical to continuous
        2. Store DP marginal counts for optional inverse transform
        3. Run super().fit() to get groups
        """
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







if __name__ == '__main__':
    # data_path = Path("c:/data/1_iknl/processed/bente/cc_9col.csv")
    # X = pd.read_csv(data_path)
    # columns = ['tum_topo_sublokalisatie_code', 'tum_differentiatiegraad_code', 'tum_lymfklieren_positief_atl']
    # # columns = ['tum_lymfklieren_positief_atl']
    # X = X.loc[:, columns]
    # print(X.head(20))
    #
    # gen_cont = GeneralizeContinuous(n_bins=10, strategy='quantile', labeled_missing=[999])
    # # X = X.dropna()
    #
    # gen_cont.fit(X)
    # X_cat = gen_cont.transform(X)
    # print(X_cat)
    #
    # X_inv = gen_cont.inverse_transform(X_cat)
    # print(X_inv)
    # print(gen_cont.bin_edges_)

    data_path = "../../examples/data/input/adult_9c.csv"
    df = pd.read_csv(data_path, delimiter=', ').astype(str)
    print(df.head())
    df_s = df[['native-country', 'occupation']]

    # epsilon = float(np.inf)
    epsilon = 0.1
    gen_cat = GeneralizeCategorical(epsilon=epsilon, n_bins=5)
    gen_cat.fit(df_s)
    df_sT = gen_cat.transform(df_s)
    df_sT = pd.DataFrame(df_sT, columns=df_s.columns)

    df_sI = pd.DataFrame(gen_cat.inverse_transform(df_sT), columns=df_s.columns)

    visual.compare_value_counts(df_s, df_sI)