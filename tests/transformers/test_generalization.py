import numpy as np
import pandas as pd
import pytest

import synthesis.transformers.generalization as generalization

def test_sample_from_binned_column():
    test_df = pd.DataFrame({'integer_column': [0, 15, 33, np.nan, 44, 14, 50]})
    test_df_gen = generalization.bin_numeric_column(test_df, 'integer_column', n_bins=3, col_min=0, col_max=50, strategy='quantile')
    test_df_inverse = generalization.sample_from_binned_column(test_df_gen, 'integer_column', numeric_type='int')

    assert isinstance(test_df_inverse.integer_column[0], int)
    assert test_df_inverse['integer_column'].min() >= test_df['integer_column'].min(), "Min value should be greater than or equal to original min value"
    assert test_df_inverse['integer_column'].max() <= test_df['integer_column'].max(), "Max value should be less than or equal to original max value"
    assert test_df_inverse['integer_column'].isnull().sum() == 1, "Only one np.nan in original column, so only one np.nan should be in inverse column"

    test_df_inverse_seed1 = generalization.sample_from_binned_column(test_df_gen, 'integer_column', numeric_type='int', mean=26, std=20, random_state=42)
    test_df_inverse_seed2 = generalization.sample_from_binned_column(test_df_gen, 'integer_column', numeric_type='int', mean=26, std=20, random_state=42)
    assert test_df_inverse_seed1.equals(test_df_inverse_seed2), "Same seed should produce same results"
