"""Base classes for all metrics"""
from abc import ABC, abstractmethod

from synthesis.synthesizers.utils import astype_categorical

COLOR_PALETTE = ['#393e46', '#ff5722', '#d72323']

class BaseMetric(ABC):

    def __init__(self, labels=None, exclude_columns=None, astype_cat=True):
        self.labels = labels or ['original', 'synthetic']
        self.exclude_columns = exclude_columns
        self.astype_cat = astype_cat

    def fit(self, data_original, data_synthetic):
        raise NotImplementedError("Implement fit method")

    def score(self, *args, **kwargs):
        raise NotImplementedError("Implement score method")

    def plot(self, *args, **kwargs):
        pass

    def _check_input_data(self, data_original, data_synthetic):
        # prevent integer column_name indexing issues
        data_original.columns = data_original.columns.astype(str)
        data_synthetic.columns = data_synthetic.columns.astype(str)

        # exclude columns
        if self.exclude_columns:
            columns = [c for c in data_original.columns if c not in self.exclude_columns]
            data_original = data_original[columns]
            data_synthetic = data_synthetic[columns]

        # ensure same columns order original data
        data_synthetic = data_synthetic[data_original.columns]

        # convert to categorical and add 'nan' as category
        if self.astype_cat:
            data_original = astype_categorical(data_original, include_nan=True)
            data_synthetic = astype_categorical(data_synthetic, include_nan=True)

        # todo check data alignment, i.e. whether synthetic data has same columns and categories as original
        # todo warn if dataset does not have same dimensions
        # todo convert column_name dtypes synthetic to original dtypes - warn user.
        return data_original, data_synthetic


class BasePredictiveMetric(BaseMetric):

    def __init__(self,  labels=None, exclude_columns=None, astype_cat=True, y_column=None,
                 random_state=None, n_jobs=None):
        super().__init__(labels=labels, exclude_columns=exclude_columns, astype_cat=astype_cat)
        self.y_column = y_column
        self.random_state = random_state
        self.n_jobs = n_jobs

    def score(self, data_original_test):
        raise NotImplementedError("Implement score method")

    def _check_input_args(self, data_original):
        if self.y_column is None:
            self.y_column = data_original.columns[-1]