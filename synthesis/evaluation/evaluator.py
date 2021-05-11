"""
Utility evaluator. Comparing a reference dataset to 1 or more target datasets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import synthesis.evaluation.metrics as metrics
from synthesis.evaluation._base import BaseMetric


METRICS = {
    'js_distance_columns': metrics.JSDistanceColumns(),
    'js_distance_average': metrics.JSDistanceAverage(),
    'associations': metrics.Associations()
}


class SyntheticEvaluator():
    def __init__(self, metrics=None):
        """Choose which metrics to compute"""
        self.metrics = metrics

    def fit(self, data_original, data_synthetic):
        self._check_input_args()
        data_original, data_synthetic = self._check_input_data(data_original, data_synthetic)

        for name, metric in self.metrics.items():
            metric.fit(data_original, data_synthetic)
        return self

    def score(self):
        scores = {}
        for name, metric in self.metrics.items():
            scores[name] = metric.score()
        return scores

    def plot(self):
        for name, metrics in self.metrics.items():
            metrics.plot()

    def _check_input_args(self):
        if self.metrics is not None:
            for m in metrics:
                if not isinstance(m, BaseMetric):
                    raise ValueError("Input metric {} should subclass synthesis.evaluation._base.BaseMetric".format(m))
        else:
            self.metrics = METRICS

    def _check_input_data(self, data_original, data_synthetic):
        # ensure same columns order original data
        data_synthetic = data_synthetic[data_original.columns]
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

    se = SyntheticEvaluator()
    se.fit(df, df_pb)
    se.score()
    se.plot()