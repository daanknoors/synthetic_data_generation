"""
Synthetic Data Generation via marginal distributions.
"""

import numpy as np
import pandas as pd
from synthesis.synthesizers._base import BaseDPSynthesizer
from synthesis.synthesizers.utils import dp_marginal_distribution, uniform_distribution


class MarginalSynthesizer(BaseDPSynthesizer):
    """Synthetic Data Generation via marginal distributions.

    Aims to only preserve the marginal distributions of each column independently. Does
    not preserve the statistical patterns between columns. Will work with any dataset
    that fits into memory."""

    def __init__(self, epsilon=1.0, verbose=True):
        super().__init__(epsilon=epsilon, verbose=verbose)

    def fit(self, data):
        data = self._check_input_data(data)
        self._check_init_args()
        # divide epsilon budget over each column
        local_epsilon = self.epsilon / data.shape[1]

        # note that in Python 3 dicts remember insertion order - thus no need to use ordered dict
        self.model_ = {}
        for c in data.columns:
            marginal = dp_marginal_distribution(data[c], local_epsilon)
            self.model_[c] = dict(zip(marginal.as_dict()['states'][c], marginal.as_dict()['data']))
            if self.verbose:
                print('Marginal fitted: {}'.format(c))
        return self

    def sample(self, n_records=None):
        self._check_is_fitted()
        n_records = n_records or self.n_records_fit_

        data_synth = {}
        # sample columns independently from marginal distributions
        for c in self.columns_:
            column_values = list(self.model_[c].keys())
            column_value_probabilities = list(self.model_[c].values())
            column_sampled = np.random.choice(column_values, p=column_value_probabilities, size=n_records, replace=True)
            data_synth[c] = column_sampled
            if self.verbose:
                print('Column sampled: {}'.format(c))
        data_synth = pd.DataFrame(data_synth)
        data_synth = self._check_output_data(data_synth)
        return data_synth


class UniformSynthesizer(MarginalSynthesizer):
    """Synthetic Data Generation via uniform distribution.
    """

    def __init__(self, epsilon=1.0, verbose=True):
        super().__init__(epsilon=epsilon, verbose=verbose)

    def fit(self, data):
        data = self._check_input_data(data)
        self._check_init_args()

        self.model_ = {}
        for c in data.columns:
            uniform = uniform_distribution(data[c])
            self.model_[c] = dict(zip(uniform.as_dict()['states'][c], uniform.as_dict()['data']))
            if self.verbose:
                print('Uniform fitted: {}'.format(c))
        return self

