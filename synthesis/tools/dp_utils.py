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

from diffprivlib.tools.histograms import histogram as diffprivlib_hist

from diffprivlib.mechanisms import Laplace, LaplaceBoundedDomain
from diffprivlib.utils import PrivacyLeakWarning
from thomas.core import CPT

import synthesis.tools.utils as utils


def dp_contingency_table(X, epsilon=1.0, range=None):
    """Represent data as a differentiall private contingency table of all attributes"""
    # if range is None:
    #     warnings.warn("Range parameter has not been specified. Falling back to taking range from the data.\n"
    #                   "To ensure differential privacy, and no additional privacy leakage, the range must be "
    #                   "specified independently of the data (i.e., using domain knowledge).", PrivacyLeakWarning)
    # todo evaluate range privacy leakage

    contingency_table_ = utils.contingency_table(X)

    # sensitivity similar to histogram, if we remove one record from X the count in one
    # cell will decrease by 1.
    sensitivity = 1
    # todo evaluate set_bound and geometric mechanism
    # dp_mech = Laplace().set_epsilon(epsilon).set_sensitivity(1).set_bounds(0, maxsize)
    dp_mech = Laplace().set_epsilon(epsilon).set_sensitivity(sensitivity)

    dp_contingency_table = np.zeros_like(contingency_table_.values)

    for i in np.arange(dp_contingency_table.shape[0]):
        # round counts upwards to preserve bins with noisy count lower than 1
        dp_contingency_table[i] = np.ceil(dp_mech.randomise(contingency_table_[i]))

    # noise can result into negative counts, thus set boundary at 0
    # dp_contingency_table[dp_contingency_table < 0] = 0
    dp_contingency_table = np.clip(dp_contingency_table, a_min=0, a_max=None)
    return pd.Series(dp_contingency_table, index=contingency_table_.index)


def dp_joint_distribution(X, epsilon=1.0, range=None):
    """Represent data as a differentially private joint distribution of all attributes"""
    # if range is None:
    #     warnings.warn("Range parameter has not been specified. Falling back to taking range from the data.\n"
    #                   "To ensure differential privacy, and no additional privacy leakage, the range must be "
    #                   "specified independently of the data (i.e., using domain knowledge).", PrivacyLeakWarning)
    # todo evaluate range privacy leakage

    joint_distribution_ = utils.joint_distribution(X)

    # removing one record from X will decrease probability 1/n in one cell of the
    # joint distribution and increase the probability 1/n in the remaining cells
    sensitivity = 2/X.shape[0]

    dp_mech = Laplace().set_epsilon(epsilon).set_sensitivity(sensitivity)

    dp_joint_distribution_ = np.zeros_like(joint_distribution_.values)

    for i in np.arange(dp_joint_distribution_.shape[0]):
        dp_joint_distribution_[i] = dp_mech.randomise(joint_distribution_[i])

    # laplacian_noise = np.random.laplace(0, scale=sensitivity / epsilon, size=joint_distribution_.shape[0])
    # dp_joint_distribution = joint_distribution_ + laplacian_noise

    # noise can result into negative probabilities, thus set boundary at 0 and re-normalize
    dp_joint_distribution_[dp_joint_distribution_ < 0] = 0
    # dp_joint_distribution_ = np.clip(dp_joint_distribution_, a_min=0, a_max=None)
    dp_joint_distribution_ = dp_joint_distribution_ / dp_joint_distribution_.sum()
    return pd.Series(dp_joint_distribution_, index=joint_distribution_.index)


def dp_conditional_distribution(X, epsilon=1.0, conditioned_variables=None, range=None):
    dp_joint_distribution_ = dp_joint_distribution(X, epsilon=epsilon)
    cpt = CPT(dp_joint_distribution_, conditioned_variables=conditioned_variables)
    # todo: use custom normalization to fill missing values with uniform
    cpt = utils.normalize_cpt(cpt, dropna=False)
    return cpt

