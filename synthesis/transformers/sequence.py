"""Classes for transforming sequential data
"""

import numpy as np
import pandas as pd
import datetime
from pathlib import Path


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state


class FlattenSequentialRecords(TransformerMixin, BaseEstimator):
    pass


class GeneralizeDateSequence(TransformerMixin, BaseEstimator):

    def __init__(self, date_sequence=None):
        self.date_sequence = date_sequence

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples. Contain only date columns
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        if self.date_sequence is None:
            self.date_sequence = X.columns
            print("no data_sequence specified in init - assume columns in X "
                  "represent the ordered data sequence")
        return self

    def transform(self, X, y=None):
        X = X.copy()
        self.reference_date_col = self.date_sequence[0]

        # for col in self.date_sequence:
        #     try:
        #         X[col] = pd.to_datetime(X[col], infer_datetime_format=True)
        #     except:
        #         X[col] = pd.to_datetime(X[col], infer_datetime_format=True, errors='coerce')
        #
        X = pd.to_datetime(X.stack()).unstack()
        reverse_date_sequence = self.date_sequence[::-1]

        for idx, col in enumerate(reverse_date_sequence):
            if col is not self.reference_date_col:
                prior_event_date = X[reverse_date_sequence[idx+1]]
                days_after_prior_event = (X[col] - prior_event_date).dt.days
                X[col] = days_after_prior_event

        # cat_year_month = X[reference_date_col].astype(str).str.rsplit('-', 1).str[0]
        cat_year_month = X[self.reference_date_col].dt.to_period('m')
        X[self.reference_date_col] = cat_year_month
            # X[c + '_year'] = X[c].dt.year
            # X[c + '_month'] = X[c].dt.month
            # del X[c]
        return X

    def inverse_transform(self, Xt):
        """Transform categorical dates back to datetime format"""

        assert Xt.columns.isin(self.date_sequence).all(), 'input dataframe contains columns not seen in fit'

        if not pd.core.dtypes.common.is_period_dtype(Xt[self.reference_date_col]):
            Xt[self.reference_date_col] = pd.to_datetime(Xt[self.reference_date_col].astype(str)).dt.to_period('m')

        reference_period_ym = Xt[self.reference_date_col]
        days_in_month = reference_period_ym.dt.days_in_month

        sample_days = np.random.uniform(0, days_in_month)
        # missing dates have days in month of -1
        sample_days[sample_days < 0] = np.nan
        sample_days = np.ceil(sample_days)

        reference_date_ymd = pd.to_datetime({'year': reference_period_ym.dt.year,
                                             'month': reference_period_ym.dt.month,
                                             'day': sample_days})

        # Xinv = check_array(Xt, copy=True, dtype=FLOAT_DTYPES)
        Xinv = Xt.copy()
        Xinv[self.reference_date_col] = reference_date_ymd

        for c in self.date_sequence[1:]:
            # add days to reference date
            Xinv[c] = reference_date_ymd + pd.to_timedelta(Xinv[c].astype(float), unit='d')
            reference_date_ymd = Xinv[c]

        return Xinv


if __name__ == '__main__':
    data_path = Path("c:/data/1_iknl/processed/crc_behandeling.csv")
    X = pd.read_csv(data_path)
    columns = ['gbs_begin_dtm', 'gbs_eind_dtm']
    X = X.loc[:, columns]
    print(X.head(20))

    date_sequence = columns
    gen_dates = GeneralizeDateSequence(date_sequence)
    X_cat = gen_dates.fit_transform(X)
    print(X_cat.head(20))

    X_inv = gen_dates.inverse_transform(X_cat)
    print(X_inv.head(20))