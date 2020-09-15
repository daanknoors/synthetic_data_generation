import numpy as np
import pandas as pd
import os

from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path



class _BaseSynthesizer(BaseEstimator, TransformerMixin):
    """
    Base class for synthesizers that includes code to check input and save models and output datasets
    """
    epsilon: float

    def _check_input_args(self):
        pass

    def _check_input_data(self, X):
        """Ensures that input data in the correct format"""

        # converts to dataframe in case of numpy input and make all columns categorical.
        X = pd.DataFrame(X).astype(str, copy=False)
        assert X.shape[1] > 1, "input needs at least 2 columns"
        # prevent integer column indexing issues
        X.columns = X.columns.astype(str)
        if hasattr(self, '_header'):
            assert set(X.columns) == set(self._header), "input contains different columns than seen in fit"
        else:
            self._header = list(X.columns)
        return X

    def write_csv(self, X, X_name, path=None):
        """"Write csv with a descriptive name of the algorithm and used parameter settings"""
        if path is None:
            path = os.getcwd()
            print('Path not specified - will save data in '
                  'current working directory: {}'.format(path))
        path = Path(path)
        filename = X_name + '_' + self.__class__.__name__ + '_' \
                   + str(self.epsilon) + 'eps.csv'
        full_path = path / filename
        X.to_csv(full_path, index=False)
        print("Data written to csv with path: {}".format(full_path))
        return self



