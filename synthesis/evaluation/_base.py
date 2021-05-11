"""Base classes for all metrics"""
from abc import ABC, abstractmethod



class BaseMetric(ABC):

    def __init__(self, labels=None):
        self.labels = labels or ['original', 'synthetic']

    def score(self, data_original, data_synthetic):
        raise NotImplementedError("Implement score method")

    def plot(self, data_original, data_synthetic):
        pass

    # def __repr__(self):
    #     msg = f"{self.__class__.__name__}(score='{self.score_}')"
    #     return msg

class BaseStatistic(ABC):

    def __init__(self):
        pass

    def describe(self, data):
        pass


class BaseModel(ABC):

    def __init__(self):
        pass

    def fit(self, data):
        pass