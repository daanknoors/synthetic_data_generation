"""
Synthetic Data Generation using a histogram representation
"""

import numpy as np
import pandas as pd
from pyhere import here
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from scipy.spatial.distance import jensenshannon
from synthesis._base_synthesis import _BaseSynthesizer
from synthesis.tools import dp_utils
from synthesis.evaluation import visual


class HistSynthesizer(_BaseSynthesizer):
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
        self.dp_contingency_table_, self.X_bins_ = dp_utils.dp_contingency_table(X, self.epsilon)
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
        """Return jensen-shannon distance

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
    """Represent data as histogram. When generating can fix condition for certain values in histogram"""

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


class MarginalSynthesizer(_BaseSynthesizer):
    """Per-column histogram synthesis - sample from DP marginal distributions.
    Will work with any dataset that fits into memory. Does not aim to preserve
    patterns between columns."""

    def __init__(self, epsilon: float = 1.0, random_state=None, n_records_synth=None, verbose=2):
        self.epsilon = epsilon
        self.random_state = random_state
        self.n_records_synth = n_records_synth
        self.verbose = verbose

    def fit(self, X, y=None):
        if hasattr(self, 'schema_'):
            print('Schema is already fitted')
            return self
        X = X.copy().astype(str)
        self._n_records_fit, self._n_columns_fit = X.shape

        self.get_schema(X)
        return self

    def transform(self, X):
        n_records = self.n_records_synth if self.n_records_synth is not None else self._n_records_fit


        Xt = {}
        for c in X.columns:
            column_values = list(self.schema_[c].keys())
            column_value_probabilities = list(self.schema_[c].values())
            column_sampled = np.random.choice(column_values, p=column_value_probabilities, size=n_records, replace=True)
            Xt[c] = column_sampled
            if self.verbose >= 1:
                print('Column sampled: {}'.format(c))
        return pd.DataFrame(Xt)

    def get_schema(self, X):
        local_epsilon = self.epsilon / X.shape[1]
        self.schema_ = {}

        for c in X.columns:
            # note that in Python 3 dicts remember insertion order - thus no need to use ordered dict
            marginal = dp_utils.dp_marginal_distribution(X[c], local_epsilon)
            self.schema_[c] = marginal.to_dict()
            if self.verbose >= 1:
                print('Column fitted: {}'.format(c))
        return self

    def set_schema(self, schema, n_records):
        """Give option to user to define schema, else infer from data"""
        self._n_records_fit = n_records
        self._n_columns_fit = len(schema)
        self.schema_ = schema
        return self


if __name__ == '__main__':
    # data_path = r"c:/data/1_iknl/processed/crc_stage_subsettumor.csv"
    # data_path = r"c:/data/1_iknl/raw/jrc/CancerCases_NL_nov2016.csv"
    data_path = here("examples/data/input/adult_9c.csv")
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

    visual.compare_value_counts(df, df_synth)


    con_hist = ConditionalHistSynthesizer(epsilon=float(np.inf))
    con_hist.fit(df)
    df_synth_tuned = con_hist.transform(df_synth.loc[:, 'age'])
    visual.compare_value_counts(df, df_synth_tuned)


