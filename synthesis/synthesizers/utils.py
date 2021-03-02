"""
General utility functions and building blocks for synthesizers
"""
import numpy as np
import pandas as pd

from diffprivlib.mechanisms import LaplaceTruncated
from thomas.core import CPT
from sys import maxsize



def dp_contingency_table(X, epsilon=1.0):
    """Compute differentially private contingency table of input data"""
    contingency_table_ = contingency_table(X)

    # if we remove one record from X the count in one cell decreases by 1 while the rest stays the same.
    sensitivity = 1
    dp_mech = LaplaceTruncated(epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity)

    dp_contingency_table = np.zeros_like(contingency_table_.values)
    for i in np.arange(dp_contingency_table.shape[0]):
        # round counts upwards to preserve bins with noisy count between [0, 1]
        dp_contingency_table[i] = np.ceil(dp_mech.randomise(contingency_table_.values[i]))

    return pd.Series(dp_contingency_table, index=contingency_table_.index)


def dp_marginal_distribution(X, epsilon=1.0):
    """Compute differentially private marginal distribution of input data"""
    assert len(X.shape) == 1, 'can only do 1-way marginal distribution, check contingency table or ' \
                            'joint distribution for higher dimensions'
    marginal_ = X.value_counts(normalize=True, dropna=False)

    # removing one record from X will decrease probability 1/n in one cell of the
    # marginal distribution and increase the probability 1/n in the remaining cells
    sensitivity = 2/X.shape[0]

    dp_mech = LaplaceTruncated(epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity)
    dp_marginal = np.zeros_like(marginal_.values)

    for i in np.arange(dp_marginal.shape[0]):
        # round counts upwards to preserve bins with noisy count between [0, 1]
        dp_marginal[i] = dp_mech.randomise(marginal_.values[i])

    dp_marginal = _normalize_distribution(dp_marginal)
    return pd.Series(dp_marginal, index=marginal_.index)


def dp_joint_distribution(X, epsilon=1.0, range=None):
    """Compute differentially private joint distribution of input data"""
    joint_distribution_ = joint_distribution(X)

    # removing one record from X will decrease probability 1/n in one cell of the
    # joint distribution and increase the probability 1/n in the remaining cells
    sensitivity = 2/X.shape[0]

    dp_mech = LaplaceTruncated(epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity)
    dp_joint_distribution_ = np.zeros_like(joint_distribution_.values)

    for i in np.arange(dp_joint_distribution_.shape[0]):
        dp_joint_distribution_[i] = dp_mech.randomise(joint_distribution_.values[i])

    dp_joint_distribution_ = _normalize_distribution(dp_joint_distribution_)
    return pd.Series(dp_joint_distribution_, index=joint_distribution_.index)



def dp_conditional_distribution(X, epsilon=1.0, conditioned_variables=None, range=None):
    dp_joint_distribution_ = dp_joint_distribution(X, epsilon=epsilon)
    cpt = CPT(dp_joint_distribution_, conditioned_variables=conditioned_variables)
    # todo: use custom normalization to fill missing values with uniform
    cpt = utils.normalize_cpt(cpt, dropna=False)
    return cpt


"""Non-differentially private functions below"""

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
    cpt = _normalize_cpt(cpt, dropna=False)
    return cpt

"""Check and fix distributions"""
def _normalize_distribution(distribution):
    """Check whether probability distribution sums to 1"""
    distribution = _check_all_zero(distribution)
    distribution = distribution / distribution.sum()
    return distribution

def _check_all_zero(distribution):
    """In case distribution contains only zero values due to DP noise, convert to uniform"""
    if not np.any(distribution):
        distribution = np.repeat(1/len(distribution), repeats=len(distribution))
    return distribution

def _normalize_cpt(cpt, dropna=False):
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

def rank_columns_on_cardinality(X):
    """Rank columns based on number of unique values"""
    column_cardinalities = {}
    for col in X.columns:
        column_cardinalities[col] = X[col].nunique()

    ranked_column_cardinalities = pd.Series(column_cardinalities).sort_values(ascending=False)
    return ranked_column_cardinalities

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
