"""Base classes for all synthesizers"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class BaseReversibleTransformer(TransformerMixin, BaseEstimator):

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        X = self._check_input_data(X)
        return self

    def transform(self, X, y=None):
        raise NotImplementedError("Implement transform method")

    def inverse_transform(self, Xt):
        Xt = self._check_output_data(Xt)
        return Xt

    def _check_input_data(self, data_input):
        self.dtypes_fit_ = data_input.dtypes
        return data_input

    def _check_output_data(self, data_output):
        # convert dtypes back to original - first convert object to bool to prevent all values from becoming True
        bool_cols = self.dtypes_fit_[self.dtypes_fit_ == bool].index
        data_output[bool_cols] = data_output[bool_cols].replace({'False': False, 'True': True})
        data_output = data_output.astype(self.dtypes_fit_)
        return data_output