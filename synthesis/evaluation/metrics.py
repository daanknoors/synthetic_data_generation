"""Module with metrics for comparison of datasets"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import jensenshannon
from dython.nominal import compute_associations

from synthesis.evaluation._base import BaseMetric

class JSDistanceColumns(BaseMetric):

    def __init__(self, labels=None):
        super().__init__(labels=labels)

    def fit(self, data_original, data_synthetic):
        """Calculate jensen_shannon distance between original and synthetic data.
        Look for more elaborate evaluation techniques in synthesis.evaluation.

        Parameters
        ----------
        data_original: pandas.DataFrame
            Original data that was seen in fit
        data_synthetic: pandas.DataFrame
            Synthetic data that was generated based on original data
        Returns
        -------
        per-column jensen_shannon distance (if score_dict): dict
            Per-column jensen_shannon distance
        """
        data_original, data_synthetic = self._check_input_data(data_original, data_synthetic)
        self.stats_original_ = {}
        self.stats_synthetic_ = {}
        for c in data_original.columns:
            # compute value_counts for both original and synthetic - align indexes as certain column
            # values in the original data may not have been sampled in the synthetic data
            self.stats_original_[c], self.stats_synthetic_[c] = \
                data_original[c].astype(str).value_counts(dropna=False).align(
                    data_synthetic[c].astype(str).value_counts(dropna=False), join='outer', axis=0, fill_value=0
                )
        return self

    def score(self):
        column_distances = {}
        for c in self.stats_original_.keys():
            js_distance = jensenshannon(self.stats_original_[c], self.stats_synthetic_[c])
            column_distances[c] = js_distance
        return column_distances

    def plot(self):
        column_distances = self.score()
        columns = list(column_distances.keys())
        y_pos = np.arange(len(columns))

        plt.barh(y_pos, column_distances.values())
        plt.yticks(y_pos, columns)
        plt.xlim(0, 1)
        plt.xlabel('Jensen-Shannon distance')
        plt.title('Jensen-Shannon distance per column')
        plt.tight_layout()
        plt.show()

class JSDistanceAverage(JSDistanceColumns):
    """
    Returns
    -------
    average jensen_shannon distance: float
        Average jensen_shannon distance over all columns
    """
    def __init__(self, labels=None):
        super().__init__(labels=labels)

    def fit(self, data_original, data_synthetic):
        super().fit(data_original, data_synthetic)
        return self

    def score(self):
        column_distances = super().score()
        return sum(column_distances.values()) / len(column_distances.keys())

    def plot(self):
        """Score is only a single value, thus no need to plot"""
        pass

class Associations(BaseMetric):

    def __init__(self, theil_u=True, nominal_columns='auto', labels=None):
        super().__init__(labels=labels)
        self.theil_u = theil_u
        self.nominal_columns = nominal_columns

    def fit(self, data_original, data_synthetic):
        data_original, data_synthetic = self._check_input_data(data_original, data_synthetic)

        # bug nominal.associations: fills nan values inplace
        data_original = data_original.copy()
        data_synthetic = data_synthetic.copy()
        self.stats_original_ = compute_associations(data_original, theil_u=self.theil_u,
                                                     nominal_columns=self.nominal_columns, nan_replace_value='nan')
        self.stats_synthetic_ = compute_associations(data_synthetic, theil_u=self.theil_u,
                                                     nominal_columns=self.nominal_columns, nan_replace_value='nan')
        return self

    def score(self):
        pairwise_correlation_distance = np.linalg.norm(self.stats_original_-self.stats_synthetic_, 'fro')
        return pairwise_correlation_distance

    def plot(self):
        pcd = self.score()

        fig, ax = plt.subplots(1, 2, figsize=(6, 4))
        cbar_ax = fig.add_axes([.91, 0.3, .01, .4])

        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Original
        heatmap_original = sns.heatmap(self.stats_original_, ax=ax[0], square=True, annot=False, center=0, linewidths=0,
                         cmap=cmap, xticklabels=False, yticklabels=False, cbar_kws={'shrink': 0.8},
                         cbar_ax=cbar_ax, fmt='.2f')
        ax[0].set_title(self.labels[0])

        # Synthetic
        heatmap_synthetic = sns.heatmap(self.stats_synthetic_, ax=ax[1], square=True, annot=False, center=0, linewidths=0,
                         cmap=cmap, xticklabels=False, yticklabels=False, cbar=False, cbar_kws={'shrink': 0.8})
        ax[1].set_title(self.labels[1] + '\n' + 'PCD: {}'.format(round(pcd, 4)))

        cbar = heatmap_original.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
        fig.suptitle('Association matrices', fontsize=14)
        plt.show()



