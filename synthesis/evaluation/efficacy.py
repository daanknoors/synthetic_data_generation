"""Methods to determine whether the synthetic data performs similar to the original data on specific tasks"""

import numpy as np
import pandas as pd
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
from sklearn.linear_model import LogisticRegression


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
        if self.categorical_columns:
            self.categorical_columns_ = self.categorical_columns
        else:
            self.categorical_columns_ = list(X.select_dtypes(include=['object']).columns)

        if self.numeric_columns:
            self.numeric_columns_ = self.numeric_columns
        else:
            self.numeric_columns_ = [c for c in X.columns if c not in self.categorical_columns_]

        preprocessor_num = Pipeline(steps=[
            ('simple_imputer_num', SimpleImputer(strategy='median')),
            ('num_scaling', MinMaxScaler()),
        ])

        preprocessor_cat = Pipeline(steps=[
            ('simple_imputer_cat', SimpleImputer(strategy='most_frequent')),
            ('ensure_consistent_dtype', EnsureConsistentType(dtype=str)),  # avoid mixed type columns
            ('categorical_encoding', OneHotEncoder(handle_unknown='ignore'))
        ])

        # first do numeric columns as categorical will create new columns
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', preprocessor_num, self.numeric_columns_),
            ('cat', preprocessor_cat, self.categorical_columns_)
        ])
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.preprocessor.transform(X)


class ClassifierComparison(BasePredictiveMetric):
    """Class to compare performance of a classifier on both original and synthetic data"""

    def __init__(self, clf=None, default_clf=None, labels=None, exclude_columns=None, y_column=None, random_state=None, n_jobs=None):
        super().__init__(labels=labels, exclude_columns=exclude_columns, astype_cat=False, y_column=y_column,
                         random_state=random_state, n_jobs=n_jobs)
        """Define clf or use a a default classifier 'lr' (logistic_regression) or 'rf' (random_forest)"""
        self.clf = clf
        self.default_clf = default_clf

    def fit(self, data_original, data_synthetic):
        """Fit classifier on both datasets"""
        self._check_input_args(data_original)
        data_original, data_synthetic = self._check_input_data(data_original, data_synthetic)

        X_original, y_original = self._split_xy(data_original)
        X_synthetic, y_synthetic = self._split_xy(data_synthetic)

        self.stats_original_ = self._fit(X_original, y_original)
        self.stats_synthetic_ = self._fit(X_synthetic, y_synthetic)
        return self

    def _fit(self, X, y):
        """Fit classifier on data"""
        # clone clf to construct a new unfitted estimator with same parameters
        clf = clone(self.clf) if self.clf is not None else self._select_default_clf()
        return clf.fit(X, y)

    def score(self, data_original_test):
        """Score both original and synthetic data on test data"""
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
        """Split data into X and y"""
        y = data[self.y_column]
        X = data.drop(self.y_column, axis=1)
        return X, y
    
    def _select_default_clf(self):
        """Select default classifier"""
        if self.default_clf == 'lr':
            return default_logreg_pipeline(random_state=self.random_state)
        elif self.default_clf == 'rf':
            return default_randomforest_pipeline(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown default classifier {self.default_clf}, select 'lr' or 'rf'")

    def _check_input_args(self, data_original):
        """Check whether input arguments are valid"""
        if self.clf is None and self.default_clf is None:
            raise ValueError('Either clf or default_clf must be set')
        if self.clf is not None and self.default_clf is not None:
            raise ValueError('Only one of clf or default_clf can be set')            
        return super()._check_input_args(data_original)


class FeatureImportanceComparison(ClassifierComparison):
    """Fit a Random Forest or Logistic Regression model on both datasets and compare the feature importances
    or the coefficients respectively.
    Note: only works with default models in ClassifierComparison"""

    def __init__(self, default_clf, y_column=None, exclude_columns=None, random_state=None, n_jobs=None):
        """Define clf or use a a default classifier (logistic_regression or random_forest)"""
        super().__init__(default_clf=default_clf, y_column=y_column, exclude_columns=exclude_columns, random_state=random_state, n_jobs=n_jobs)

    def fit(self, data_original, data_synthetic):
        # run default classifier comparison with RandomForest GridSearch
        super().fit(data_original, data_synthetic)

        # get feature importances
        self.fi_original_ = self._get_feature_importance(self.stats_original_.best_estimator_, name=self.labels[0])
        self.fi_synthetic_ = self._get_feature_importance(self.stats_synthetic_.best_estimator_, name=self.labels[1])

        # align feature importances, as some values in columns might not be sampled in synthetic data
        self.fi_original_, self.stats_synthetic_ = \
            self.fi_original_.align(self.fi_synthetic_,join='outer', axis=0, fill_value=0)
        return self

    def plot(self, data_original_test, top=30):
        # merge feature importances
        df_fi = pd.merge(self.fi_original_, self.fi_synthetic_, left_index=True, right_index=True, how='outer').fillna(0)

        # transform index column with feature name to column
        df_fi = df_fi.rename_axis('feature').reset_index()

        # take most importance features from the original model
        df_fi = df_fi.sort_values(by=df_fi.columns[1], ascending=False)[:top]

        # melt to tidy format
        df_fi = df_fi.melt(id_vars='feature').rename(columns={'variable': 'data', 'value': 'importance'})

        # plot feature importance comparison
        fig, ax = plt.subplots(figsize=(6, 10))
        sns.despine()
        sns.barplot(data=df_fi, y='feature', x='importance', hue='data', palette=COLOR_PALETTE, ax=ax)
        ax.legend(loc='lower right')
        if self.default_clf == 'lr':
            ax.set_title('Logistic Regression coefficient comparison')
            ax.set_xlabel('coefficient')
        else:
            ax.set_title('Random Forest feature importance comparison')


    def _get_feature_importance(self, clf, name):
        # get feature names after one hot encoding the categorical columns
        onehot_features = (clf.named_steps['preprocessor'].preprocessor.named_transformers_['cat']
                           .named_steps['categorical_encoding']
                           .get_feature_names_out(clf.named_steps['preprocessor'].categorical_columns_))

        # get numeric columns
        numeric_columns = clf.named_steps['preprocessor'].numeric_columns_
        feature_names_original = numeric_columns + list(onehot_features)

        # get feature importances
        if self.default_clf == 'lr':
            fi = clf.named_steps['classifier'].coef_[0]
        else:
            fi = clf.named_steps['classifier'].feature_importances_

        df_feature_importances = pd.Series(fi, index=feature_names_original, name=f'{name}')
        return df_feature_importances


def default_logreg_pipeline(random_state=None):
    """Default logistic regression pipeline with grid search"""
    # load generic preprocessor with distinct transformers for numeric and categorical features
    preprocessor = GenericColumnTransformer()

    # Classifier
    rf = LogisticRegression(penalty='l2', max_iter=1000, random_state=random_state)

    # Pipeline
    pipe = Pipeline([('preprocessor', preprocessor),
                    ('classifier', rf)])

    # Grid search
    params = {
    'classifier__C': np.logspace(-4, 4, 10),
    'classifier__class_weight': ['balanced', None]}

    # use average precision as default scoring as it also works well on imbalanced data
    clf_logreg = GridSearchCV(pipe, param_grid=params, scoring='average_precision', refit=True, cv=3, verbose=0, n_jobs=-1)
    return clf_logreg


def default_randomforest_pipeline(random_state=None):
    """Default random forest pipeline with grid search"""
    # load generic preprocessor with distinct transformers for numeric and categorical features
    preprocessor = GenericColumnTransformer()

    # Classifier
    rf = RandomForestClassifier(class_weight='balanced', min_samples_leaf=0.05,
                                    random_state=random_state)

    # Pipeline
    pipe = Pipeline([('preprocessor', preprocessor),
                        ('classifier', rf)])

    # Grid search
    params = {'classifier__n_estimators': [100, 200, 500],
                'classifier__criterion': ['entropy', 'gini'],
                'classifier__max_depth': [3, 5, 10]}

    # use average precision as default scoring as it also works well on imbalanced data
    clf = GridSearchCV(pipe, param_grid=params, scoring='average_precision', refit=True, cv=3, verbose=1, n_jobs=-1)
    return clf