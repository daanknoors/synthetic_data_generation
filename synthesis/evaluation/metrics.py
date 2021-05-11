"""Module with utiliy metrics for comparison of datasets"""

import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon

from synthesis.evaluation._base import BaseMetric

class JSDistanceColumns(BaseMetric):

    def score(self, data_original, data_synth):
        """Calculate jensen_shannon distance between original and synthetic data.
        Look for more elaborate evaluation techniques in synthesis.evaluation.

        Parameters
        ----------
        data_original: pandas.DataFrame
            Original data that was seen in fit
        data_synth: pandas.DataFrame
            Synthetic data that was generated based on original data
        Returns
        -------
        per-column jensen_shannon distance (if score_dict): dict
            Per-column jensen_shannon distance
        """
        column_distances = {}
        for i, c in enumerate(data_original.columns):
            # compute value_counts for both original and synthetic - align indexes as certain column
            # values in the original data may not have been sampled in the synthetic data
            counts_original, counts_synthetic = \
                data_original[c].value_counts(dropna=False).align(
                    data_synth[c].value_counts(dropna=False), join='outer', axis=0, fill_value=0
                )

            js_distance = jensenshannon(counts_original, counts_synthetic)
            column_distances[c] = js_distance
        self.score_ = column_distances
        return self.score_

class JSDistanceAverage(JSDistanceColumns):
    """
    Returns
    -------
    average jensen_shannon distance: float
        Average jensen_shannon distance over all columns
    """

    def score(self, data_original, data_synth):
        column_distances = super().score(data_original, data_synth)
        self.score_ = sum(column_distances.values()) / len(data_original.columns)
        return self.score_


def feature_distances(x1, x2, distance_function=None):
    """Distance between each feature. Optional: input own distance function"""
    assert (x1.columns == x2.columns).all(), "input x1 and x2 have different features"
    if distance_function is None:
        distance_function = jensenshannon

    features = list(x1.columns)
    feature_distances = np.empty_like(x1.columns)
    for i, c in enumerate(x1.columns):
        counts_x1, counts_x2 = x1[c].value_counts(dropna=False).align(x2[c].value_counts(dropna=False), join='outer',
                                                                      axis=0, fill_value=0)
        js_distance = distance_function(counts_x1, counts_x2)
        feature_distances[i] = js_distance
    return features, feature_distances


def avg_feature_distance(x1, x2, distance_function=None):
    features, distances = feature_distances(x1, x2, distance_function)
    return np.sum(distances) / len(features)


def jensenshannon_df(x1, x2, j_ways=1):
    """Jensen_shannon distance over all features in dataframe.
    Returns distance per feature and average distance
    """
    assert (x1.columns == x2.columns).all(), "input x1 and x2 have different features"
    feature_distances = {}
    average_feature_distance = np.empty_like(x1.columns)
    for i, c in enumerate(x1.columns):
        counts_x1, counts_x2 = x1[c].value_counts(dropna=False).align(x2[c].value_counts(dropna=False), join='outer',
                                                                      axis=0, fill_value=0)
        js_distance = jensenshannon(counts_x1, counts_x2)
        average_feature_distance[i] = js_distance
        feature_distances[c] = js_distance
    average_feature_distance = np.sum(average_feature_distance) / len(x1.columns)
    return feature_distances, average_feature_distance
