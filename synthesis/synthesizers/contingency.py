"""
Synthetic Data Generation via contingency tables.
"""

import numpy as np
import pandas as pd

from synthesis.synthesizers._base import BaseDPSynthesizer, FixedSamplingMixin
from synthesis.synthesizers.utils import dp_contingency_table, cardinality


class ContingencySynthesizer(FixedSamplingMixin, BaseDPSynthesizer):
    """Synthetic Data Generation via contingency table.

    Creates a contingency tables based on the whole input dataset.
    Note: memory-intensive - data with many columns and/or high column cardinality might not fit into memory.
    """

    def __init__(self, epsilon=1.0, verbose=True):
        super().__init__(epsilon=epsilon, verbose=verbose)

    def fit(self, data):
        data = self._check_input_data(data)
        self._check_init_args()

        ct_size = cardinality(data)
        print('Estimated size contingency table: {}'.format(ct_size))

        self.model_ = dp_contingency_table(data, self.epsilon)
        if self.verbose:
            print('Contingency table fitted')
        return self

    def sample(self, n_records=None):
        self._check_is_fitted()
        n_records = n_records or self.n_records_fit_

        prob = np.array(self.model_) / sum(self.model_)
        idx = np.arange(len(self.model_))

        sampled_idx = np.random.choice(idx, size=n_records, p=prob, replace=True)
        sampled_records = self.model_[sampled_idx].index.to_frame(index=False)
        return sampled_records


class ContingencySynthesizerFix(ContingencySynthesizer, FixedSamplingMixin):
    """Synthetic Data Generation via contingency table. Fixes column defined in sample.

    Retains fixed input column and generates remaining columns seen in fit.

    Use cases:
    - to generate only a subset of columns in original data
    - to combine the output of multiple synthesizers

    Creates a contingency tables based on the whole input dataset.
    Note: memory-intensive - data with many columns and/or high column cardinality might not fit into memory."""

    def __init__(self, epsilon=1.0, verbose=True):
        super().__init__(epsilon=epsilon, verbose=verbose)

    def sample_remaining_columns(self, fixed_column):
        self._check_is_fitted()
        self._check_fixed_data(fixed_column)
        n_records = fixed_column.shape[0]

        sampled_records = self._sample_conditioned(fixed_column, n_records)
        return pd.DataFrame(sampled_records, columns=self.columns_)

    def _sample_conditioned(self, fixed_column, n_records):

        # conditioning variable is categorical -> transform NaN to 'nan'
        fixed_column = pd.Series(fixed_column).astype(str)

        # sample records from histogram and add to synthetic data
        sampled_records = np.empty(n_records, dtype=object)
        for idx, condition in enumerate(fixed_column):
            # todo optimize slicing -> no need to find histogram idx everytime
            sliced_hist = self.slice_contingency_table(column=fixed_column.name, value=condition)
            if len(sliced_hist) == 0:
                print('empty')
            sampled_tuple = self._sample(sliced_hist)
            sampled_records[idx] = sampled_tuple

        return np.vstack(sampled_records)

    def _sample(self, contingency_table):
        prob = np.array(contingency_table) / sum(contingency_table)
        idx = np.arange(len(contingency_table))
        try:
            sampled_idx = np.random.choice(idx, 1, p=prob)
        except:
            print('not working')
        # [0] in order to retrieve tuple instead of array per row
        sampled_tuple = contingency_table.index[sampled_idx].values[0]
        return sampled_tuple

    def slice_contingency_table(self, column, value):
        """Return slice of contingency table based on conditioning variable"""
        condition_idx = self.model_.get_pandas_index().get_level_values(column) == value
        return self.model_[condition_idx]


if __name__ == '__main__':
    data_path = 'C:/projects/synthetic_data_generation/examples/data/original/adult_8c.csv'
    df = pd.read_csv(data_path)
    epsilon = float(np.inf)
    df_sub = df[['education', 'occupation', 'relationship']]

    cs = ContingencySynthesizer(epsilon=epsilon)
    cs.fit(df_sub)
    df_cs = cs.sample()
    df_cs.head()

    cs.score(df_sub, df_cs, score_dict=True)

    cs_fix = ContingencySynthesizerFix(epsilon=epsilon)
    cs_fix.fit(df_sub)
    cs_fix.sample_remaining_columns(df_cs['education'])
