"""Methods for de-identifying original data"""

import numpy as np
import pandas as pd
import pandas_flavor as pf

@pf.register_dataframe_method
def replace_unique_values(df: pd.DataFrame, column_name: str, replace_string: str ='entity_') -> pd.DataFrame:
    """Replaces unique values with numbered string. Retains nans."""
    df[column_name] = df[column_name].astype(str)
    replace_dict = {v: (replace_string + str(i) if v != 'nan' else np.nan) for i, v  in
                    enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].replace(replace_dict)
    return df

@pf.register_dataframe_method
def group_rare_values(df: pd.DataFrame, column_names=None, threshold=0.05, rare_group_label='other'):
    """For each column in dataframe replaces values that occur less than threshold"""
    column_names = column_names or df.columns
    if isinstance(column_names, str):
        column_names = [column_names]

    df = df.copy()
    for c in column_names:
        df.loc[:, c] = df[c].mask(df[c].map(df[c].value_counts(normalize=True)) < threshold, rare_group_label)
    return df

@pf.register_dataframe_method
def add_fake_identifiers(df, colummn_name='id',  id_string='FAKE_PERSON_'):
    """Add column with fake identifiers"""
    df = df.copy()
    df[colummn_name] = [id_string + str(i) for i in np.arange(1, df.shape[0]+1)]
    return df


