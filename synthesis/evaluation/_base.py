"""Base classes for all metrics"""
from abc import ABC, abstractmethod

class BaseMetric(ABC):

    def __init__(self, labels=None):
        self.labels = labels or ['original', 'synthetic']

    def fit(self, data_original, data_synthetic):
        raise NotImplementedError("Implement fit method")

    def score(self, *args, **kwargs):
        raise NotImplementedError("Implement score method")

    def plot(self, *args, **kwargs):
        pass

    def _check_input_data(self, data_original, data_synthetic):
        # ensure same columns order original data
        data_synthetic = data_synthetic[data_original.columns]
        # todo check data alignment, i.e. whether synthetic data has same columns and categories as original
        # todo warn if dataset does not have same dimensions
        return data_original, data_synthetic


class BasePredictiveMetric(BaseMetric):

    def __init__(self, y_column=None, random_state=None, n_jobs=None, labels=None):
        super().__init__(labels=labels)
        self.y_column = y_column
        self.random_state = random_state
        self.n_jobs = n_jobs

    def score(self, data_original_test):
        raise NotImplementedError("Implement score method")