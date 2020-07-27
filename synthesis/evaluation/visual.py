"""Module with functions for visualizing the difference between datasets"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from synthesis.evaluation import metrics

def plot_feature_distances(x1, x2, labels=None):
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
    x1 = x1.astype(str)
    x2 = x2.astype(str)
    for c in x1.columns:
        df_compare = pd.concat([x1[c].value_counts(), x2[c].value_counts()], axis=1, sort=True)
        df_compare.columns = ['x1', 'x2']

        print('='*100)
        print(c)
        print(df_compare)

