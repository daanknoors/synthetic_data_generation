"""Classes for de-identifying original data"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RemoveColumns(TransformerMixin, BaseEstimator):
    """Transformer to remove columns"""

    def __init__(self, remove_columns=None):
        self.remove_columns = remove_columns

    def fit(self, X, y=None):
        if self.remove_columns is None:
            raise ValueError('Specify columns to be removed when instantiating class.')
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.remove_columns)


class RemoveUniqueRows:
    """Transformer to remove unique rows that occur less than threshold"""

    def __init__(self, min_n_duplicates):
        self.min_n_duplicates = min_n_duplicates

    def fit(self, data_original):
        pass
    def transform(self, data_original):
        pass


class KeepFirstIdentifier(TransformerMixin, BaseEstimator):
    """Ensure that each person has only one record in the input data.
    Sort by 1 or more id_columns and only retain the first occurrence of the first id_column."""
    def __init__(self, id_columns):
        self.id_columns = id_columns

    def fit(self, X, y=None):
        self.id_columns = self.id_columns if isinstance(self.id_columns, list) else [self.id_columns]
        return self

    def transform(self, X, y=None):
        return X.sort_values(by=self.id_columns, ascending=True).drop_duplicates(subset=self.id_columns[0], keep='first')

    def inverse_transform(self, Xt):
        pass


class ReplaceUniqueValues(TransformerMixin, BaseEstimator):
    """Replace sensitive values with numbered string (e.g. organisation names)"""

    def __init__(self, column, replace_string=None):
        self.column = column
        self.replace_string = replace_string

    def fit(self, X, y=None):
        self.replace_string = self.replace_string or 'entity_'
        return self

    def transform(self, X, y=None):
        X[self.column] = X[self.column].astype(str)
        replace_dict = {v: self.replace_string + str(i) for i, v in
                        enumerate(X[self.column].unique()) if v != 'nan'}
        X[self.column] = X[self.column].replace(replace_dict)
        return X

    def inverse_transform(self, Xt):
        pass



def identify_unique_records(threshold):
    """identify records that occur less than threshold"""
    pass



def fake_identifiers(data, id_string='FAKE_PERSON_'):
    """generate fake identifiers"""
    return [id_string + str(i) for i in np.arange(0, data.shape[0])]


