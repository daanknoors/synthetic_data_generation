"""
Synthetic Data Generation using a histogram representation
"""

import numpy as np
import pandas as pd
from pyhere import here
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from scipy.spatial.distance import jensenshannon

import synthesis.tools.dp_utils as sts


class HistSynthesizer(BaseEstimator, TransformerMixin):
    """Represent data as histogram of all attributes and synthesize based on counts

    """

    def __init__(self, epsilon: float = 1.0, random_state=None):
        # super().__init__()
        self.epsilon = epsilon
        self.random_state = random_state


    def fit(self, X, y=None):
        # X = X.loc[columns] if columns is not None else X

        self.random_state_ = check_random_state(self.random_state)
        #todo evaluate need for sklearn checks, e.g. check_x_y()
        self.column_names_ = X.columns
        self.n_records_ = X.shape[0]
        self.dp_contingency_table_, self.X_bins_ = sts.dp_contingency_table(X, self.epsilon)
        print('Obtained noisy hist')
        return self

    def transform(self, X, n_records=None):
        # Check that the input is of the same shape as the one passed
        # during fit.
        if not np.all([c in self.column_names_ for c in X.columns]):
            raise ValueError('Input has different columns from what was seen in `fit`')

        n_records = self.n_records_ if n_records is None else n_records
        sampled_records = self._sample(n_records)

        return sampled_records.to_frame(index=False)
        # return pd.DataFrame(sampled_records, columns=self.column_names_)

    def score(self, X, y=None, j_ways=1):
        """Return average variation distance

           Parameters
           ----------
            X : synth array, shape(n_samples, n_features)
                The data.
            y : None
                Ignored variable.

           Returns
           -------
        """
        # marginal_domain = get_marginal_domain(y, j_ways)
        # marginal_counts_X = count_marginals(X, marginal_domain)
        # marginal_counts_y = count_marginals(y, marginal_domain)
        #
        # synth_hist, synth_bins = sts.contingency_table(X, epsilon=np.inf(float))
        self.feature_distances = {}
        average_feature_distance = np.empty_like(X.columns)
        for i, c in enumerate(X.columns):
            counts_X, counts_y = X[c].value_counts(dropna=False).align(y[c].value_counts(dropna=False), join='outer',
                                                                       axis=0, fill_value=0)
            js_distance = jensenshannon(counts_X, counts_y)
            average_feature_distance[i] = js_distance
            self.feature_distances[c] = js_distance
        average_feature_distance = np.sum(average_feature_distance) / len(X.columns)
        return average_feature_distance


    def _sample(self, n_records):
        prob = np.array(self.dp_contingency_table_) / sum(self.dp_contingency_table_)
        idx = np.arange(len(self.dp_contingency_table_))

        sampled_idx = np.random.choice(idx, size=n_records, p=prob, replace=True)
        sampled_records = self.X_bins_[sampled_idx]

        return sampled_records



class ConditionalHistSynthesizer(HistSynthesizer):

    def __init__(self, epsilon: float = 1.0, random_state=None):
        super().__init__(epsilon=epsilon, random_state=random_state)

    def transform(self, X):
        self._check_condition(X)

        n_records = X.shape[0]
        sampled_records = self._sample_conditioned(X, n_records)
        return pd.DataFrame(sampled_records, columns=self.column_names_)

    def _check_condition(self, X):
        if isinstance(X, pd.Series):
            assert X.name in self.column_names_, "Conditioning variable X not seen in fit"
        if isinstance(X, pd.DataFrame):
            if np.all([c in self.column_names_ for c in X.columns]):
                raise ValueError('Input has different columns from what was seen in `fit`')

    def _sample_conditioned(self, X, n_records):

        # conditioning variable is categorical -> transform NaN to 'nan'
        X = X.astype(str)

        assert set(np.unique(self.X_bins_.get_level_values(X.name))) == \
               set(np.unique(X)), 'conditioning variable has different categories'

        # sample records from histogram and add to synthetic data
        sampled_records = np.empty(n_records, dtype=object)
        for idx, condition in enumerate(X):
            # todo optimize slicing -> no need to find histogram idx everytime
            sliced_hist, sliced_bins = self.slice_hist(column=X.name, value=condition)
            if len(sliced_hist) == 0:
                print('empty')
            sampled_tuple = self._sample(sliced_hist, sliced_bins, n_records=1)
            sampled_records[idx] = sampled_tuple

        return np.vstack(sampled_records)

    def _sample(self, histogram, bins, n_records):
        prob = np.array(histogram) / sum(histogram)
        idx = np.arange(len(histogram))
        try:
            sampled_idx = np.random.choice(idx, n_records, p=prob)
        except:
            print('not working')
        # [0] in order to retrieve tuple instead of array per row
        sampled_tuple = bins[sampled_idx].values[0]
        return sampled_tuple

    def slice_hist(self, column, value):
        """Return slice of histogram based on conditioning variable"""
        condition_idx = self.X_bins_.get_level_values(column) == value
        return self.dp_contingency_table_[condition_idx], self.X_bins_[condition_idx]





def compare_synthetic_data(X, y):
    for c in X.columns:
        # counts_real = df_real[c].value_counts().sort_index()
        # # in case the synth dataset did not sample certain attribute values from the real data we assign a count of 0
        # _, counts_synth = counts_real.align(df_synth[c].value_counts(), join='left', axis=0, fill_value=0)
        counts_X, counts_y = X[c].value_counts(dropna=False).align(y[c].value_counts(dropna=False), join='outer',
                                                                   axis=0, fill_value=0)
        df_compare = pd.concat([counts_X, counts_y], axis=1)
        df_compare.columns = ['real', 'synthetic']

        print('='*100)
        print(c)
        print(df_compare)



if __name__ == '__main__':
    # data_path = r"c:/data/1_iknl/processed/crc_stage_subsettumor.csv"
    # data_path = r"c:/data/1_iknl/raw/jrc/CancerCases_NL_nov2016.csv"
    data_path = here("examples/data/input/adult.csv")
    df = pd.read_csv(data_path, delimiter=', ').astype(str)
    columns = ['age', 'sex', 'education', 'workclass', 'income']
    df = df.loc[:, columns]
    print(df.head())

    # epsilon = float(np.inf)
    epsilon = 0.01
    hist_synthesizer = HistSynthesizer(epsilon=epsilon)
    hist_synthesizer.fit(df)
    df_synth = hist_synthesizer.transform(df)
    print(df_synth.head())

    compare_synthetic_data(df, df_synth)


    con_hist = ConditionalHistSynthesizer(epsilon=float(np.inf))
    con_hist.fit(df)
    df_synth_tuned = con_hist.transform(df_synth.loc[:, 'age'])
    compare_synthetic_data(df, df_synth_tuned)


