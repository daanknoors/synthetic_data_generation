"""Module with metrics for comparison of datasets"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import jensenshannon
from dython.nominal import compute_associations

from synthesis.evaluation._base import BaseMetric

COLOR_PALETTE = ['#393e46', '#ff5722', '#d72323']

class MarginalComparison(BaseMetric):

    def __init__(self, labels=None, normalize=True):
        super().__init__(labels=labels)
        self.normalize = normalize

    def fit(self, data_original, data_synthetic):
        data_original, data_synthetic = self._check_input_data(data_original, data_synthetic)
        self.stats_original_ = {}
        self.stats_synthetic_ = {}
        self.stats_ = {}
        for c in data_original.columns:
            # compute value_counts for both original and synthetic - align indexes as certain column
            # values in the original data may not have been sampled in the synthetic data
            self.stats_original_[c], self.stats_synthetic_[c] = \
                data_original[c].astype(str).value_counts(dropna=False, normalize=self.normalize).align(
                    data_synthetic[c].astype(str).value_counts(dropna=False, normalize=self.normalize),
                    join='outer', axis=0, fill_value=0
                )
            self.stats_[c] = jensenshannon(self.stats_original_[c], self.stats_synthetic_[c])

        return self

    def score(self):
        average_js_distance = sum(self.stats_.values()) / len(self.stats_.keys())
        return average_js_distance

    def plot(self):
        column_names = self.stats_original_.keys()
        fig, ax = plt.subplots(len(self.stats_original_.keys()), 1, figsize=(8, len(column_names) * 4))

        for idx, col in enumerate(column_names):

            column_value_counts_original = self.stats_original_[col]
            column_value_counts_synthetic = self.stats_synthetic_[col]

            bar_position = np.arange(len(column_value_counts_original.values))
            bar_width = 0.35

            # with small column cardinality plot original distribution as bars, else plot as line
            if len(column_value_counts_original.values) <= 20:
                ax[idx].bar(x=bar_position, height=column_value_counts_original.values,
                            color=COLOR_PALETTE[0], label=self.labels[0], width=bar_width)
            else:
                ax[idx].plot(column_value_counts_original.index, column_value_counts_original.values, marker='o',
                             markersize=4, color=COLOR_PALETTE[0], linewidth=2, label=self.labels[0])

            # synthetic distribution
            ax[idx].bar(x=bar_position + bar_width, height=column_value_counts_synthetic.values,
                        color=COLOR_PALETTE[1], label=self.labels[1], width=bar_width)

            ax[idx].set_xticks(bar_position + bar_width / 2)
            if len(column_value_counts_original.values) <= 20:
                ax[idx].set_xticklabels(column_value_counts_original.keys(), rotation=25)
            else:
                ax[idx].set_xticklabels('')
            # ax[idx].set_title('Column: ' +  r"$\bf{" + col + "}$" +
            #                   ' ~ jensen-shannon distance: ' + '{:.2f}'.format(self.stats_[col]))
            title = r"$\bf{" + col + "}$" + "\n jensen-shannon distance: {:.2f}".format(self.stats_[col])
            ax[idx].set_title(title)
            if self.normalize:
                ax[idx].set_ylabel('Probability')
            else:
                ax[idx].set_ylabel('Count')

            ax[idx].legend()
        fig.tight_layout()
        plt.show()

class AssociationsComparison(BaseMetric):

    def __init__(self, theil_u=True, nominal_columns='auto', labels=None):
        super().__init__(labels=labels)
        self.theil_u = theil_u
        self.nominal_columns = nominal_columns

    def fit(self, data_original, data_synthetic):
        data_original, data_synthetic = self._check_input_data(data_original, data_synthetic)

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

        fig, ax = plt.subplots(1, 2, figsize=(8, 6))
        cbar_ax = fig.add_axes([.91, 0.3, .01, .4])

        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Original
        heatmap_original = sns.heatmap(self.stats_original_, ax=ax[0], square=True, annot=False, center=0, linewidths=0,
                         cmap=cmap, xticklabels=True, yticklabels=True, cbar_kws={'shrink': 0.8},
                         cbar_ax=cbar_ax, fmt='.2f')
        ax[0].set_title(self.labels[0] + '\n')

        # Synthetic
        heatmap_synthetic = sns.heatmap(self.stats_synthetic_, ax=ax[1], square=True, annot=False, center=0, linewidths=0,
                         cmap=cmap, xticklabels=True, yticklabels=False, cbar=False, cbar_kws={'shrink': 0.8})
        ax[1].set_title(self.labels[1] + '\n' + 'pairwise correlation distance: {}'.format(round(pcd, 4)))

        cbar = heatmap_original.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
        # fig.tight_layout()
        plt.show()
