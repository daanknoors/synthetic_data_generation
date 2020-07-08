"""Module with utiliy metrics for comparison of datasets"""

import numpy as np
import pandas as pd
from pyhere import here

from scipy.spatial.distance import jensenshannon


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
