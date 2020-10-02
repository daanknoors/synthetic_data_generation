"""
General functions for differentially private computations
"""
import warnings
from sys import maxsize
from numbers import Real
import numpy as np
import pandas as pd
from numpy.core import multiarray as mu
from numpy.core import umath as um
from itertools import product
from thomas.core import CPT

from diffprivlib.tools.histograms import histogram as diffprivlib_hist

from diffprivlib.mechanisms import Laplace, LaplaceBoundedDomain
from diffprivlib.utils import PrivacyLeakWarning


# def contingency_table(X):
#     """Represent data as contingency table of all attributes"""
#     # X is normally categorical, convert everything to string to prevent indexing issues
#     X = X.astype(str)
#     columns = list(X.columns)
#
#     counts = X.fillna('nan').groupby(columns).size().astype(float)
#
#     # get variable combinations that do not occur in the data and set count to 0
#     full_space_index = pd.MultiIndex.from_tuples(tuple(product(*counts.index.levels)),
#                                                  names=counts.index.names)
#     contingency_table_ = pd.Series(data=0, index=full_space_index).combine(counts, max)
#     return contingency_table_


def contingency_table(X):
    """Represent data as contingency table of all attributes"""
    index = [X[c] for c in X.columns[:-1]]
    column = X[X.columns[-1]]
    return pd.crosstab(index, column, dropna=False).stack()


def joint_distribution(X):
    """Get joint distribution by normalizing contingency table"""
    contingency_table_ = contingency_table(X)
    joint_distribution_ = contingency_table_ / contingency_table_.sum()
    return joint_distribution_


def conditional_distribution(X, conditioned_variables):
    """Conditional distribution with custom normalization procedure to fill missing values with uniform distribution"""
    joint_distribution_ = joint_distribution(X)
    cpt = CPT(joint_distribution_, conditioned_variables=conditioned_variables)
    # todo: use custom normalization to fill missing values with uniform
    cpt = normalize_cpt(cpt, dropna=False)
    return cpt


def normalize_cpt(cpt, dropna=False):
    """normalization of cpt with option to fill missing values with uniform distribution"""
    if dropna or not cpt.conditioning:
        return cpt.normalize()

    # fill missing combinations with uniform distribution
    cpt_norm_full = cpt._data / cpt.unstack().sum(axis=1)
    uniform_prob = 1 / len(cpt.variable_states[cpt.conditioned[-1]])
    cpt_norm_full = cpt_norm_full.fillna(uniform_prob)
    return CPT(cpt_norm_full)

def get_size_contingency_table(X):
    """Get size of contingency table prior to calculating it"""
    size = 1
    for jj in X.columns:
        size *= X[jj].nunique()
    return size


# def contingency_table(X):
#     """Represent data as contingency table of all attributes"""
#     #todo can we do this faster using pd_crosstab or the old privbayes method? -> check nan bins
#     for col in X.columns:
#         X[col] = X[col].astype("object")
#
#     columns = list(X.columns)
#
#     contingency_table_ = X.fillna('nan').groupby(columns).size().astype(float)
#     contingency_table_ = _add_zerocount_bins(contingency_table_)
#
#     return contingency_table_
#
#
# def _add_zerocount_bins(counts):
#     """Adds combinations of attributes that do not exist in the input data"""
#     nlevels = counts.index.nlevels - 1
#     stack = counts
#
#     # unstack to get nan's in pd.Dataframe
#     for _ in range(nlevels):
#         stack = stack.unstack()
#     # add count of 0 to non-existing combinations
#     stack = stack.fillna(0)
#     # reverse stack back to a pd.Series
#     for _ in range(nlevels):
#         stack = stack.stack()
#     return stack



def _init_synth_data(columns):
    # initialize synthetic data structure based on original input
    synth_data = {}
    # column_names_ = list(hist.index.names)
    for c in columns:
        synth_data[c] = []
    return synth_data

def sample_record_from_hist(histogram, bins, n_records=1):
    prob = np.array(histogram) / sum(histogram)
    idx = np.arange(len(histogram))
    try:
        sampled_idx = np.random.choice(idx, n_records, p=prob)
    except:
        print('not working')
    sampled_tuple = bins[sampled_idx][0]
    return sampled_tuple


if __name__ == '__main__':
    pass