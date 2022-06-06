"""Methods to determine whether the synthetic data performs similar to the original data on specific tasks"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import KaplanMeierFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.base import BaseEstimator, TransformerMixin, clone

from synthesis.evaluation._base import BaseMetric, BasePredictiveMetric, COLOR_PALETTE


class KaplanMeierComparison(BaseMetric):

    def __init__(self, time_column, event_column, group_column, labels=None):
        super().__init__(labels=labels)
        self.time_column = time_column
        self.event_column = event_column
        self.group_column = group_column

    def _fit(self, data):
        return {
            'time': data[self.time_column].astype(float),
            'event': data[self.event_column].astype(float),
            'group': data[self.group_column]
        }

    def fit(self, data_original, data_synthetic):
        self.stats_original_ = self._fit(data_original)
        self.stats_synthetic_ = self._fit(data_synthetic)
        return self

    def score(self):
        """for now only support plotting -
        could check whether logranktest on original and synthetic for each group agree on significance"""
        return None

    def plot(self):
        """
        Plot side-by-side kaplan-meier of input datasets
        """

        figsize = (10, 5)
        fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)

        # sns.set(font_scale=1.5)
        # sns.despine()
        palette = ['#0d3d56', '#006887', '#0098b5', '#00cbde', '#00ffff']

        datasets = [self.stats_original_, self.stats_synthetic_]
        for data, label, ax_cur in zip(datasets, self.labels, ax):
            t = data['time']
            e = data['event']

            kmf = KaplanMeierFitter()
            groups = np.sort(data['group'].unique())
            for g, color in zip(groups, palette):
                mask = (data['group'] == g)
                kmf.fit(t[mask], event_observed=e[mask], label=g)
                ax_cur = kmf.plot_survival_function(ax=ax_cur, color=color)
                ax_cur.legend(title=self.group_column)
                ax_cur.set_title('Kaplan-Meier - {} data'.format(label))
                ax_cur.set_ylim(0, 1)
        plt.tight_layout()


class EnsureConsistentType(BaseEstimator, TransformerMixin):
    """Ensure consistent type in dataset - used to avoid issues with mixed types in columns"""
    def __init__(self, dtype=str):
        self.dtype=dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.astype(self.dtype)


class GenericColumnTransformer(BaseEstimator, TransformerMixin):
    """Performs standard processing operations on numeric and categorical columns"""
    def __init__(self, categorical_columns=None, numeric_columns=None):
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns

    def fit(self, X, y=None):
        cat_cols = self.categorical_columns if self.categorical_columns else X.select_dtypes(include=['object']).columns
        num_cols = self.numeric_columns if self.numeric_columns else [c for c in X.columns if c not in cat_cols]

        preprocessor_cat = Pipeline(steps=[
            ('simple_imputer_cat', SimpleImputer(strategy='most_frequent')),
            ('ensure_consistent_dtype', EnsureConsistentType(dtype=str)),  # avoid mixed type columns
            ('categorical_encoding', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor_num = Pipeline(steps=[
            ('simple_imputer_num', SimpleImputer(strategy='median')),
            ('num_scaling', MinMaxScaler()),
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ('cat', preprocessor_cat, cat_cols),
            ('num', preprocessor_num, num_cols)
        ])
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.preprocessor.transform(X)


class ClassifierComparison(BasePredictiveMetric):

    def __init__(self, clf=None, labels=None, exclude_columns=None, y_column=None, random_state=None, n_jobs=None):
        super().__init__(labels=labels, exclude_columns=exclude_columns, astype_cat=False, y_column=y_column,
                         random_state=random_state, n_jobs=n_jobs)
        self.clf = clf

    def fit(self, data_original, data_synthetic):
        self._check_input_args(data_original)
        data_original, data_synthetic = self._check_input_data(data_original, data_synthetic)

        X_original, y_original = self._split_xy(data_original)
        X_synthetic, y_synthetic = self._split_xy(data_synthetic)

        self.stats_original_ = self._fit(X_original, y_original)
        self.stats_synthetic_ = self._fit(X_synthetic, y_synthetic)
        return self

    def _fit(self, X, y):
        # clone clf to construct a new unfitted estimator with same parameters
        clf = clone(self.clf) if self.clf else self._get_default_classifier()
        return clf.fit(X, y)

    def score(self, data_original_test):
        X_test, y_test = self._split_xy(data_original_test)

        scores = {
            "score_original": self.stats_original_.score(X_test, y_test),
            "score_synthetic": self.stats_synthetic_.score(X_test, y_test)
        }
        return scores

    def plot(self, data_original_test):
        """"Plot ROC-AOC Curves of both original and synthetic in single figure"""
        X_test, y_test = self._split_xy(data_original_test)

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        sns.despine()
        # roc curve
        RocCurveDisplay.from_estimator(self.stats_original_, X_test, y_test, name=self.labels[0],
                                       color=COLOR_PALETTE[0], ax=ax[0])
        RocCurveDisplay.from_estimator(self.stats_synthetic_, X_test, y_test, name=self.labels[1],
                                       color=COLOR_PALETTE[1], ax=ax[0])

        ax[0].plot([0, 1], [0, 1], linestyle="--", lw=1, color="black", alpha=0.7)
        ax[0].set_title('ROC Curve')

        # pr curve
        PrecisionRecallDisplay.from_estimator(self.stats_original_, X_test, y_test, name=self.labels[0],
                                       color=COLOR_PALETTE[0], ax=ax[1])
        PrecisionRecallDisplay.from_estimator(self.stats_synthetic_, X_test, y_test, name=self.labels[1],
                                              color=COLOR_PALETTE[1], ax=ax[1])
        no_skill = len(y_test[y_test == 1]) / len(y_test)
        ax[1].plot([0, 1], [no_skill, no_skill], lw=1, linestyle='--', color='black', alpha=0.7)
        ax[1].set_title('Precision-Recall Curve')

    def _split_xy(self, data):
        y = data[self.y_column]
        X = data.drop(self.y_column, axis=1)
        return X, y

    def _get_default_classifier(self):
        # load generic preprocessor with distinct transformers for numeric and categorical features
        preprocessor = GenericColumnTransformer()

        # Classifier
        rf = RandomForestClassifier(class_weight='balanced', min_samples_leaf=0.05,
                                        random_state=self.random_state)

        # Pipeline
        pipe = Pipeline([('preprocessor', preprocessor),
                         ('classifier', rf)])

        # Grid search
        params = {'classifier__n_estimators': [100, 200, 500],
                  'classifier__criterion': ['entropy', 'gini'],
                  'classifier__max_depth': [3, 5, 10]}

        # use average precision as default scoring as it also works well on imbalanced data
        clf = GridSearchCV(pipe, param_grid=params, scoring='average_precision', refit=True, cv=3, verbose=1,
                               n_jobs=self.n_jobs)
        return clf
