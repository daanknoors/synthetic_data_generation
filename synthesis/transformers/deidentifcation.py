"""Classes for de-identifying original data"""


class RemoveIdentifyingColumns:
    """Transformer to remove identifying columns"""

    def __init__(self, remove_columns=None):
        self.remove_columns = remove_columns

    def fit(self, data_original):
        if self.remove_columns is None:
            raise ValueError('Specify columns to be removed when instantiating class.')
        return self

    def transform(self, data_original):
        return data_original.drop(columns=self.remove_columns)


class RemoveUniqueRows:
    """Transformer to remove unique rows that occur less than threshold"""

    def __init__(self, min_n_duplicates):
        self.min_n_duplicates = min_n_duplicates

    def fit(self, data_original):
        pass
    def transform(self, data_original):
        pass


def identify_unique_records(threshold):
    """identify records that occur less than threshold"""
    pass