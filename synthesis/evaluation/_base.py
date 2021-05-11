"""Base classes for all metrics"""
from abc import ABC, abstractmethod



class BaseMetric(ABC):

    def __init__(self):
        pass

    def score(self, data_original, data_synth):
        pass

    def plot(self, data_original, data_synth):
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