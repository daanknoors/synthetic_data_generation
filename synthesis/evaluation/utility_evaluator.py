"""
Utility evaluator. Comparing a reference dataset to 1 or more target datasets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import synthesis.evaluation.metrics as metrics
import synthesis.evaluation.statistics as statistics

STATISTICS = {
    'associations': statistics.Associations()
}

METRICS = {
    'js_distance_columns': metrics.JSDistanceColumns(),
    'js_distance_average': metrics.JSDistanceAverage()
}

MODELS = {
    'random_forest': RandomForestClassifier()
}


class UtilityEvaluator():
    """Compare datasets with similar structure"""

    def __init__(self, statistics=None, metrics=None, models=None):
        """Choose which statistics and metrics to compute"""
        self.statistics = statistics
        self.metrics = metrics
        self.models = models

    def fit(self, data_original, data_synthetic):
        """load dataset original and dictionary of synthetic datasets
        self.descriptives_: summary statistics of original dataset and each synthetic dataset
        self.coefs_: fitted machine learning model on original dataset and each synthetic dataset
        self.scores_: results comparing original dataset to each synthetic dataset

        """
        self._check_input_args()
        data_original, data_synthetic = self._check_input_data(data_original, data_synthetic)

        self.descriptives_ = {'original': self.describe(data_original)}
        # self.coefs_ = {'original': self.model(data_original)}
        self.scores_ = {}

        for name_synth, data_synth in data_synthetic.items():
            self.descriptives_[name_synth] = self.describe(data_synth)
            # self.coefs_[name_synth] = self.model(data_synth)
            self.scores_[name_synth] = self.score(data_original, data_synth)


    def describe(self, data):
        descriptives = {}
        for name, statistic in self.statistics.items():
            descriptives[name] = statistic.describe(data)
        return descriptives

    def score(self, data_original, data_synth):
        scores = {}
        for name, metric in self.metrics.items():
            scores[name] = metric.score(data_original, data_synth)
        return scores

    # def model(self, data):
    #     # for simplicity we assume that y is last column of input data
    #     x = data.iloc[:, :-1]
    #     y = data.iloc[:, -1]
    #     coefs = {}
    #     for name, model in self.models.items():
    #         coefs[name] = model.fit(x, y)
    #     return coefs

    def predict(self, data_test):
        """test fitted models from original and synthetic data on the original test data"""
        pass

    def plot(self):
        """Plot results - run after fit"""
        pass

    def _check_input_args(self):
        if self.statistics is not None and set(self.statistics).issubset(set(STATISTICS.keys())):
            self.statistics = {k: v for k, v in STATISTICS if k in self.statistics}
        elif self.statistics is None:
            self.statistics = STATISTICS
        else:
            raise ValueError("unknown statistics given, can only contain metric names from STATISTICS")

        if self.metrics is not None and set(self.metrics).issubset(set(METRICS.keys())):
            self.metrics = {k: v for k, v in METRICS if k in self.metrics}
        elif self.metrics is None:
            self.metrics = METRICS
        else:
            raise ValueError("unknown metrics given, can only contain metric names from METRICS")

        if self.models is not None and set(self.models).issubset(set(MODELS.keys())):
            self.models = {k: v for k, v in METRICS if k in self.models}
        elif self.models is None:
            self.models = MODELS
        else:
            raise ValueError("unknown models given, can only contain metric names from MODELS")


    def _check_input_data(self, data_original, data_synthetic):
        if not isinstance(data_synthetic, dict):
            raise ValueError("datasets must be dictionary of {'name': pd.DataFrame}")

        for name, data_synth in data_synthetic.items():
            # ensure same columns order original data
            data_synthetic[name] = data_synth[data_original.columns]

        # todo check data alignment, i.e. whether synthetic data has same columns and categories as original
        # todo warn if dataset does not have same dimensions

        return data_original, data_synthetic




if __name__ == '__main__':
    path = Path('C:\\projects\\synthetic_data_generation\\examples\\data')
    columns = ['workclass', 'education', 'relationship', 'occupation', 'age']
    df = pd.read_csv(path / 'original/adult_8c.csv')[columns]
    df_ms = pd.read_csv(path / 'synthetic/adult_8c_MarginalSynthesizer_0.01eps.csv')[columns]
    df_pb = pd.read_csv(path / 'synthetic/adult_8c_PrivBayes_0.01eps.csv')[columns]

    df_synth = {
        'MarginalSynthesizer': df_ms,
        'PrivBayes': df_pb
    }

    ue = UtilityEvaluator()
    ue.fit(df, df_synth)

    # access fitted results
    ue.scores_
    ue.descriptives_
    # ue.coefs_