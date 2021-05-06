"""
Synthetic Data Generation via marginal distributions.
"""

import numpy as np
import pandas as pd
from synthesis.synthesizers._base import BaseDPSynthesizer
from synthesis.synthesizers.utils import dp_marginal_distribution


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

        synth_data = {}
        # sample columns independently from marginal distributions
        for c in self.columns_:
            column_values = list(self.model_[c].keys())
            column_value_probabilities = list(self.model_[c].values())
            column_sampled = np.random.choice(column_values, p=column_value_probabilities, size=n_records, replace=True)
            synth_data[c] = column_sampled
            if self.verbose:
                print('Column sampled: {}'.format(c))
        return pd.DataFrame(synth_data)


