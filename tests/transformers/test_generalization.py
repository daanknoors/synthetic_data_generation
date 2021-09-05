import numpy as np
import pandas as pd

import synthesis.transformers.generalization as generalization

def test_sample_from_binned_column():
    test_df = pd.DataFrame({'integer_column': [0, 15, 33, np.nan, 44, 14, 50]})
    test_df_gen = generalization.bin_numeric_column(test_df, 'integer_column', n_bins=3, col_min=0, col_max=50, strategy='quantile')
    test_df_inverse = generalization.sample_from_binned_column(test_df_gen, 'integer_column', numeric_type='int')
