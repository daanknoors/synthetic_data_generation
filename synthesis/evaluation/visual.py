"""Module with functions for visualizing the difference between datasets"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter

from synthesis.evaluation import metrics


def plot_kmf_comparison(datasets, dataset_names, T_varname, E_varname, G_varname):
    """
    Plot side-bys-side kaplan-meier of input datasets

    Parameters
    ----------
    datasets: list of input data
    dataset_names: names of input data - note: len equal to datasets
    T_varname: time variable name
    E_varname: event variable name
    G_varname: grouping variable name

    Returns Kaplan-Meier plot of input datasets
    -------

    """
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not isinstance(dataset_names, list):
        dataset_names = [dataset_names]
    assert len(datasets) == len(dataset_names), "input datasets and dataset_names are of different lengths"

    figsize = (8 * len(datasets), 7)
    fig, ax = plt.subplots(1, len(datasets), figsize=figsize, sharey=True)

    sns.set(font_scale=1.5)
    #     sns.set_context('paper', rc={"lines.linewidth": 1.2})
    sns.despine()
    palette = ['#0d3d56', '#006887', '#0098b5', '#00cbde', '#00ffff']

    for X, X_name, ax_cur in zip(datasets, dataset_names, ax):
        T = X[T_varname].astype(float)
        E = X[E_varname].astype(float)

        kmf = KaplanMeierFitter()
        unique_values = np.sort(X[G_varname].unique())
        for g, color in zip(unique_values, palette):
            mask = (X[G_varname] == g)
            kmf.fit(T[mask], event_observed=E[mask], label=g)
            ax_cur = kmf.plot(ax=ax_cur, color=color)
            ax_cur.legend(title=G_varname)
            ax_cur.set_title('Survival Analysis C50 - {} Data'.format(X_name))
            ax_cur.set_ylim(0, 1)
    plt.tight_layout()
