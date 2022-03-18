"""
General utility functions and building blocks for synthesizers
"""
import numpy as np
import pandas as pd

from diffprivlib.mechanisms import LaplaceTruncated
from thomas.core import Factor, CPT, JPT
from sys import maxsize

def dp_contingency_table(data, epsilon):
    """Compute differentially private contingency table of input data"""
    contingency_table_ = contingency_table(data)

    # if we remove one record from X the count in one cell decreases by 1 while the rest stays the same.
    sensitivity = 1
    dp_mech = LaplaceTruncated(epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity)

    contingency_table_values = contingency_table_.values.flatten()
    dp_contingency_table = np.zeros_like(contingency_table_values)
    for i in np.arange(dp_contingency_table.shape[0]):
        # round counts upwards to preserve bins with noisy count between [0, 1]
        dp_contingency_table[i] = np.ceil(dp_mech.randomise(contingency_table_values[i]))

    return Factor(dp_contingency_table, states=contingency_table_.states)


def dp_marginal_distribution(data, epsilon):
    """Compute differentially private marginal distribution of input data"""
    marginal_ = marginal_distribution(data)

    # removing one record from X will decrease probability 1/n in one cell of the
    # marginal distribution and increase the probability 1/n in the remaining cells
    sensitivity = 2 / data.shape[0]
    dp_mech = LaplaceTruncated(epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity)

    marginal_values = marginal_.values.flatten()
    dp_marginal = np.zeros_like(marginal_.values)

    for i in np.arange(dp_marginal.shape[0]):
        # round counts upwards to preserve bins with noisy count between [0, 1]
        dp_marginal[i] = dp_mech.randomise(marginal_.values[i])

    dp_marginal = _normalize_distribution(dp_marginal)
    return Factor(dp_marginal, states=marginal_.states)


def dp_joint_distribution(data, epsilon):
    """Compute differentially private joint distribution of input data"""
    joint_distribution_ = joint_distribution(data)

    # removing one record from X will decrease probability 1/n in one cell of the
    # joint distribution and increase the probability 1/n in the remaining cells
    sensitivity = 2 / data.shape[0]
    dp_mech = LaplaceTruncated(epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity)

    joint_distribution_values = joint_distribution_.values.flatten()
    dp_joint_distribution_ = np.zeros_like(joint_distribution_values)

    for i in np.arange(dp_joint_distribution_.shape[0]):
        dp_joint_distribution_[i] = dp_mech.randomise(joint_distribution_values[i])

    dp_joint_distribution_ = _normalize_distribution(dp_joint_distribution_)
    return JPT(dp_joint_distribution_, states=joint_distribution_.states)


def dp_conditional_distribution(data, epsilon, conditioned=None):
    """Compute differentially private conditional distribution of input data
    Inferred from marginal or joint distribution"""
    # if only one columns (series or dataframe), i.e. no conditioning columns
    if len(data.squeeze().shape) == 1:
        dp_distribution = dp_marginal_distribution(data, epsilon=epsilon)
    else:
        dp_distribution = dp_joint_distribution(data, epsilon=epsilon)
    cpt = CPT(dp_distribution, conditioned=conditioned)

    # normalize if cpt has conditioning columns
    if cpt.conditioning:
        cpt = _normalize_cpt(cpt)
    return cpt


"""Non-differentially private functions below"""

def contingency_table(data):
    return Factor.from_data(data)

def joint_distribution(data):
    """Get joint distribution by normalizing contingency table"""
    return contingency_table(data).normalize()

def marginal_distribution(data):
    assert len(data.squeeze().shape) == 1, "data can only consist of a single column"
    # converts single column dataframe to series
    data = data.squeeze()

    marginal = data.value_counts(normalize=True, dropna=False)
    states = {data.name: marginal.index.tolist()}
    return Factor(marginal, states=states)

def uniform_distribution(data):
    assert len(data.squeeze().shape) == 1, "data can only consist of a single column"
    # converts single column dataframe to series
    data = data.squeeze()
    n_unique = data.nunique(dropna=False)
    uniform = np.full(n_unique, 1/n_unique)
    states = {data.name: data.unique().tolist()}
    return Factor(uniform, states=states)

def compute_distribution(data):
    """"Draws a marginal or joint distribution depending on the number of input dimensions"""
    if len(data.squeeze().shape) == 1:
        return marginal_distribution(data)
    else:
        return joint_distribution(data)

"""Check and fix distributions"""


def _normalize_distribution(distribution):
    """Check whether probability distribution sums to 1"""
    distribution = _check_all_zero(distribution)
    distribution = distribution / distribution.sum()
    return distribution


def _check_all_zero(distribution):
    """In case distribution contains only zero values due to DP noise, convert to uniform"""
    if not np.any(distribution):
        distribution = np.repeat(1 / len(distribution), repeats=len(distribution))
    return distribution


def _normalize_cpt(cpt):
    """normalization of cpt with option to fill missing values with uniform distribution"""
    # convert to series as normalize does not work with thomas cpts
    series = cpt.as_series()
    series_norm_full = series / series.unstack().sum(axis=1)
    # fill missing combinations with uniform distribution
    uniform_prob = 1 / len(cpt.states[cpt.conditioned[-1]])
    series_norm_full = series_norm_full.fillna(uniform_prob)
    return CPT(series_norm_full, cpt.states)

def _ensure_arg_is_list(arg):
    if not arg:
        raise ValueError('Argument is empty: {}'.format(arg))

    arg = [arg] if isinstance(arg, str) else arg
    arg = list(arg) if isinstance(arg, tuple) else arg
    assert isinstance(arg, list), "input argument should be either string, tuple or list"
    return arg

def cardinality(X):
    """Compute cardinality of input data"""
    return np.prod(X.nunique(dropna=False))

def rank_columns_on_cardinality(X):
    """Rank columns based on number of unique values"""
    return X.nunique().sort_values(ascending=False)

def astype_categorical(data, include_nan=True):
    """Convert data to categorical and optionally adds nan as unique category"""
    # converts to dataframe in case of numpy input and make all columns categorical.
    data = pd.DataFrame(data).astype('category', copy=False)

    # add nan as category
    if include_nan:
        nan_columns = data.columns[data.isna().any()]
        for c in nan_columns:
            data[c] = data[c].cat.add_categories('nan').fillna('nan')
    return data