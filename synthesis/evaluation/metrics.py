"""Module with metrics for comparison of datasets"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import jensenshannon
from dython.nominal import associations
from itertools import combinations

import synthesis.synthesizers.utils as utils
from synthesis.evaluation._base import BaseMetric, COLOR_PALETTE


class MarginalComparison(BaseMetric):

    def __init__(self, labels=None, exclude_columns=None, normalize=True):
        super().__init__(labels=labels, exclude_columns=exclude_columns)
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
                data_original[c].value_counts(dropna=False, normalize=self.normalize).align(
                    data_synthetic[c].value_counts(dropna=False, normalize=self.normalize),
                    join='outer', axis=0, fill_value=0
                )
            self.stats_[c] = jensenshannon(self.stats_original_[c], self.stats_synthetic_[c])

        return self

    def score(self):
        average_js_distance = sum(self.stats_.values()) / len(self.stats_.keys())
        return average_js_distance

    def plot(self):
        column_names = self.stats_original_.keys()
        fig, ax = plt.subplots(len(column_names), 1, figsize=(8, len(column_names) * 4))

        for idx, col in enumerate(column_names):
            ax_i = ax[idx] if len(column_names) > 1 else ax

            column_value_counts_original = self.stats_original_[col]
            column_value_counts_synthetic = self.stats_synthetic_[col]

            bar_position = np.arange(len(column_value_counts_original.values))
            bar_width = 0.35

            # with small column cardinality plot original distribution as bars, else plot as line
            if len(column_value_counts_original.values) <= 25:
                ax_i.bar(x=bar_position, height=column_value_counts_original.values,
                            color=COLOR_PALETTE[0], label=self.labels[0], width=bar_width)
            else:
                ax_i.plot(bar_position + bar_width, column_value_counts_original.values, marker='o',
                             markersize=3, color=COLOR_PALETTE[0], linewidth=2, label=self.labels[0])

            # synthetic distribution
            ax_i.bar(x=bar_position + bar_width, height=column_value_counts_synthetic.values,
                        color=COLOR_PALETTE[1], label=self.labels[1], width=bar_width)

            ax_i.set_xticks(bar_position + bar_width / 2)
            if len(column_value_counts_original.values) <= 25:
                ax_i.set_xticklabels(column_value_counts_original.keys(), rotation=25)
            else:
                ax_i.set_xticklabels('')

            title = r"$\bf{" + col.replace('_', '\_') + "}$" + "\n jensen-shannon distance: {:.2f}".format(self.stats_[col])
            ax_i.set_title(title)
            if self.normalize:
                ax_i.set_ylabel('Probability')
            else:
                ax_i.set_ylabel('Count')

            ax_i.legend()
        fig.tight_layout()

class AssociationsComparison(BaseMetric):

    def __init__(self, labels=None, exclude_columns=None, nom_nom_assoc='theil', nominal_columns='auto'):
        super().__init__(labels=labels, exclude_columns=exclude_columns, astype_cat=False)
        self.nom_nom_assoc = nom_nom_assoc
        self.nominal_columns = nominal_columns

    def fit(self, data_original, data_synthetic):
        data_original, data_synthetic = self._check_input_data(data_original, data_synthetic)

        self.stats_original_ = associations(data_original, nom_nom_assoc=self.nom_nom_assoc,
                                            nominal_columns=self.nominal_columns, nan_replace_value='nan',
                                            compute_only=True)['corr']
        self.stats_synthetic_ = associations(data_synthetic, nom_nom_assoc=self.nom_nom_assoc,
                                             nominal_columns=self.nominal_columns, nan_replace_value='nan',
                                             compute_only=True)['corr']
        return self

    def score(self):
        pairwise_correlation_distance = np.linalg.norm(self.stats_original_-self.stats_synthetic_, 'fro')
        return pairwise_correlation_distance

    def plot(self):
        pcd = self.score()

        fig, ax = plt.subplots(1, 2, figsize=(12, 10))
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


class JointDistributionComparison(BaseMetric):

    def __init__(self, labels=None, exclude_columns=None, normalize=True, n_variables=2):
        super().__init__(labels=labels, exclude_columns=exclude_columns)
        self.normalize = normalize
        self.n_variables = n_variables

    def fit(self, data_original, data_synthetic):
        data_original, data_synthetic = self._check_input_data(data_original, data_synthetic)
        self.stats_original_ = {}
        self.stats_synthetic_ = {}
        self.stats_ = {}

        variable_combinations = combinations(data_original.columns, self.n_variables)
        for vars in variable_combinations:
            # compute joint distributions and convert Factors to series
            jt_original = utils.compute_distribution(data_original.loc[:, vars]).as_series()
            jt_synthetic = utils.compute_distribution(data_synthetic.loc[:, vars]).as_series()

            # align both original and synthetic in case values are missing in either dataset
            jt_original, jt_synthetic = jt_original.align(jt_synthetic, join='outer', fill_value=0, axis=0)
            self.stats_original_[vars] = jt_original.sort_index()
            self.stats_synthetic_[vars] = jt_synthetic.sort_index()

            # convert Factor to pd.Series and compute jensen-shannon distance
            self.stats_[vars] = jensenshannon(jt_original, jt_synthetic)

        return self

    def score(self):
        average_js_distance = sum(self.stats_.values()) / len(self.stats_.keys())
        return average_js_distance