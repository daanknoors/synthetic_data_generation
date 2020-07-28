"""Module with functions for visualizing the difference between datasets"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from synthesis.evaluation import metrics

def plot_feature_distances(x1, x2, labels=None):
    x1 = x1.copy().astype(str)
    x2 = x2.copy().astype(str)

    if labels is None:
        labels = ['x1', 'x2']

    features, feature_distances = metrics.feature_distances(x1, x2)
    y_pos = np.arange(len(features))

    plt.barh(y_pos, feature_distances)
    plt.yticks(y_pos, features)
    plt.xlabel('Feature distance')
    plt.title('Distances per feature')
    plt.tight_layout()
    plt.show()


def compare_value_counts(x1, x2):
    x1 = x1.copy().astype(str)
    x2 = x2.copy().astype(str)
    for c in x1.columns:
        counts_X, counts_y = x1[c].value_counts(dropna=False).align(x2[c].value_counts(dropna=False), join='outer',
                                                                   axis=0, fill_value=0)
        df_compare = pd.concat([counts_X, counts_y], axis=1).astype(int)
        df_compare.columns = ['real', 'synthetic']

        print('='*100)
        print(c)
        print(df_compare)
