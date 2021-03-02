"""Base classes for all metrics"""
from abc import ABC, abstractmethod



class BaseMetric(ABC):
    """A metric that stores scores_ each tiem score is run with a different name """

    def __init__(self):


    def score(self, name, data_original, data_synth, data_validation=None):
        self._check_saved_scores(name)
        pass

    def plot(self, data_original, data_synth):
        pass

    def _check_data_alignment(self, data_original, data_synth, data_validation):
        pass

    def _check_saved_scores(self, name):
        if not hasattr(self, 'scores_'):
            self.scores_ = {}
        if not self.scores_[name]:
            self.scores_[name] = {}