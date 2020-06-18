"""A module for pre/postprocessing dates
"""

import numpy as np
import pandas as pd
import warnings

from pyhere import here
from pathlib import Path


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, FLOAT_DTYPES
from sklearn.preprocessing import KBinsDiscretizer

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
        self.feature_names = X.columns

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
        return Xt

    def _infer_numerical_type(self, X):
        """Determine if numerical column is an integer of float for inverse transform"""
        assert X.select_dtypes(exclude=['int', 'float']).shape[1] == 0, "input X contains non-numeric columns"
        self.integer_columns = []
        self.float_columns = []

        for c in X.columns:
            if np.array_equal(X[c].dropna(), X[c].dropna().astype(int)):
                self.integer_columns.append(c)
            else:
                self.float_columns.append(c)

    def _get_missing_idx(self, column):
        return np.isnan(column) | np.isin(column, self.labeled_missing)


    def inverse_transform(self, Xt):
        # check_is_fitted(self)

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
            if self.feature_names[jj] in self.integer_columns:
                Xinv[~missing_idx, jj] = np.round(Xinv[~missing_idx, jj])

        return Xinv


if __name__ == '__main__':
    data_path = Path("c:/data/1_iknl/processed/bente/cc_9col.csv")
    X = pd.read_csv(data_path)
    columns = ['tum_topo_sublokalisatie_code', 'tum_differentiatiegraad_code', 'tum_lymfklieren_positief_atl']
    columns = ['tum_lymfklieren_positief_atl']
    X = X.loc[:, columns]
    print(X.head(20))

    gen_cont = GeneralizeContinuous(n_bins=10, strategy='quantile', labeled_missing=[999])
    # X = X.dropna()

    gen_cont.fit(X)
    X_cat = gen_cont.transform(X)
    print(X_cat)

    X_inv = gen_cont.inverse_transform(X_cat)
    print(X_inv)
    print(gen_cont.bin_edges_)