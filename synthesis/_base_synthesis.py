"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class _BaseSynthesizer(BaseEstimator):
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    Examples
    --------
    >>> from skltemplate import TemplateEstimator
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TemplateEstimator()
    >>> estimator.fit(X, y)
    _BaseSynthesizer(demo_param='demo_param')
    """
    def __init__(self, demo_param='demo_param'):
        self.demo_param = demo_param

    def fit(self, X, y):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """

        return self






class _BaseHistSynthesizer(TransformerMixin, BaseEstimator):
    """ An example transformer that returns the element-wise square root.
    For more information regarding how to build your own transformer, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """
    def __init__(self, epsilon: float = None, random_state=None):
        # super().__init__()
        self.epsilon = epsilon
        self.random_state = random_state

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """

        self.random_state_ = check_random_state(self.random_state)

        X = check_array(X, accept_sparse=True)

        self.n_features_ = X.shape[1]
        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        # Input validation
        X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        return np.sqrt(X)

    def _get_hist(self, data):
        """Represent data as hist of all attributes"""
        for col in self._data.columns:
            data[col] = data[col].astype("object")

        columns_to_group = list(data.columns)
        hist = data.fillna('nan').groupby(columns_to_group).size().astype(float)
        hist = self.add_zerocount_bins(hist)
        return hist

    def _add_zerocount_bins(self, counts):
        """Adds combinations of attributes that do not exist in the input data"""
        nlevels = counts.index.nlevels - 1
        stack = counts

        # unstack to get nan's in pd.Dataframe
        for _ in range(nlevels):
            stack = stack.unstack()
        # add count of 0 to non-existing combinations
        stack = stack.fillna(0)
        # reverse stack back to a pd.Series
        for _ in range(nlevels):
            stack = stack.stack()
        return stack