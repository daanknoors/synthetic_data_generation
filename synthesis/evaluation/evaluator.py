"""
Utility evaluator. Comparing a reference dataset to 1 or more target datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import synthesis.evaluation.metrics as metrics
from synthesis.evaluation._base import BaseMetric, COLOR_PALETTE



DEFAULT_METRICS = {
    'average_js_distance': metrics.MarginalComparison(),
    'pairwise_correlation_distance': metrics.AssociationsComparison()
}

class SyntheticDataEvaluator(BaseMetric):
    """Class to compare synthetic data to the original"""
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
        for name, metric in self.metrics.items():
            metric.plot()

    def _check_input_args(self):
        if self.metrics is not None:
            for name, metric in self.metrics.items():
                if not isinstance(metric, BaseMetric):
                    raise ValueError("Input metric {} should subclass synthesis.evaluation._base.BaseMetric".format(metric))
        else:
            self.metrics = DEFAULT_METRICS


class OriginalDataEvaluator():
    """Class to evaluate input dataframe"""
    def __init__(self, cardinality_threshold=50, rare_category_threshold=0.05):
        self.cardinality_threshold = cardinality_threshold
        self.rare_category_threshold = rare_category_threshold

    def fit(self, data):
        self.stats_ = {}
        self.stats_['columns_high_cardinality'] = self.get_high_cardinality_columns(data, self.cardinality_threshold)
        self.stats_['rare_column_categories'] = self.get_rare_column_categories(data, self.rare_category_threshold)
        return self

    def plot(self, data, normalize=True):
        column_names = data.columns
        fig, ax = plt.subplots(len(column_names), 1, figsize=(8, len(column_names) * 4))

        for idx, col in enumerate(column_names):
            column_value_counts = data.value_counts(normalize=normalize)

            bar_position = np.arange(len(column_value_counts.values))
            bar_width = 0.5

            ax[idx].bar(x=bar_position, height=column_value_counts.values,
                            color=COLOR_PALETTE[0], label='original', width=bar_width)

            ax[idx].set_xticks(bar_position + bar_width / 2)
            if len(column_value_counts.values) <= 20:
                ax[idx].set_xticklabels(column_value_counts.keys(), rotation=25)
            else:
                ax[idx].set_xticklabels('')

            title = r"$\bf{" + col + "}$"
            ax[idx].set_title(title)
            if normalize:
                ax[idx].set_ylabel('Probability')
            else:
                ax[idx].set_ylabel('Count')

            ax[idx].legend()
        fig.tight_layout()

    @staticmethod
    def get_high_cardinality_columns(data, threshold):
        """Get features with more unique values than the specified threshold."""
        return data.columns[data.nunique() > threshold].tolist()

    @staticmethod
    def get_rare_column_categories(data, threshold):
        """Get rare categories per column"""
        rare_categories = {}
        for c in data.columns:
            rare_categories[c] = [k for k, v in data[c].value_counts(normalize=True).items() if v < threshold]
        return rare_categories

