"""
Synthetic Data Generation via contingency tables.
"""

import numpy as np
import pandas as pd

from synthesis.synthesizers._base import BaseDPSynthesizer
from synthesis.synthesizers.utils import dp_contingency_table, cardinality


class ContingencySynthesizer(BaseDPSynthesizer):
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

