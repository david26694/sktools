__author__ = ["david26694", "cmougan"]

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import scipy


class CyclicFeaturizer(BaseEstimator, TransformerMixin):
    """Cyclic featurizer

    Given some numeric columns, applies sine and cosine transformations to
    obtain cyclic features. This is specially suited to month of the year,
    day of the week, day of the month, hour of the day, etc, where the plain
    numeric representation doesn't work very well.

    Parameters
    ----------
    cols : list
        columns to be encoded using sine and cosine transformations. Should be numeric columns
    period_mapping : dict
        keys should be names of cols and values should be tuples indicating minimum and maximum values

    Example
    -------
    >>> from sktools import CyclicFeaturizer
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    >>>     {
    >>>         "posted_at": pd.date_range(
    >>>             start="1/1/2018", periods=365 * 3, freq="d"
    >>>         ),
    >>>         "created_at": pd.date_range(
    >>>             start="1/1/2018", periods=365 * 3, freq="h"
    >>>         )
    >>>     }
    >>> )
    >>> df["month_posted"] = df.posted_at.dt.month
    >>> df["hour_created"] = df.created_at.dt.hour
    >>> transformed_df = CyclicFeaturizer(
    >>>     cols=["month_posted", "hour_created"]
    >>> ).fit_transform(df)

    """

    def __init__(self, cols, period_mapping=None):
        self.cols = cols
        self.period_mapping = period_mapping

    def fit(self, X):

        # If the mapping is given, no need to run it
        if self.period_mapping is not None:
            if set(self.cols) != set(self.period_mapping.keys()):
                raise ValueError("Keys of period_mapping are not the same as cols")
            return self
        else:
            # Learn values to determine periods
            self.period_mapping = {}
            for col in self.cols:
                min_col = X[col].min()
                max_col = X[col].max()
                self.period_mapping[col] = (min_col, max_col)

        return self

    def transform(self, X):

        X = X.copy()

        for col in self.cols:
            min_col, max_col = self.period_mapping[col]
            # 24 hours -> 23 - 0 + 1
            # 365 days -> 365 - 1 + 1
            period = max_col - min_col + 1
            X[f"{col}_sin"] = np.sin(2 * (X[col] - min_col) * np.pi / period)
            X[f"{col}_cos"] = np.cos(2 * (X[col] - min_col) * np.pi / period)

        return X


class GradientBoostingFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Feature generator from a gradient boosting.

    Gradient boosting decision trees are a powerful and very convenient way to implement non-linear and tuple transformations.
    We treat each individual tree as a categorical feature that takes as value the index of the leaf an instance ends up falling in
    and then perform one hot encoding for these features.

    Example
    -------
    >>> from sktools.preprocessing import GradientBoostingFeatureGenerator
    >>> from sklearn.datasets import load_boston
    >>> import numpy as np
    >>> boston = load_boston()['data']
    >>> y = load_boston()['target']
    >>> y = np.where(y>y.mean(),1,0)
    >>> mf = GradientBoostingFeatureGenerator(sparse_feat=False)
    >>> mf.fit(boston, y)
    >>> mf.transform(boston)

    References
    ----------

    .. [1] Practical Lessons from Predicting Clicks on Ads at Facebook, from
    https://research.fb.com/wp-content/uploads/2016/11/practical-lessons-from-predicting-clicks-on-ads-at-facebook.pdf
    """

    def __init__(
        self,
        stack_to_X=True,
        sparse_feat=False,
        add_probs=True,
        criterion="friedman_mse",
        init=None,
        learning_rate=0.1,
        loss="deviance",
        max_depth=3,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_estimators=50,
        n_iter_no_change=None,
        random_state=None,
        subsample=1.0,
        tol=0.0001,
        validation_fraction=0.1,
        verbose=0,
        warm_start=False,
    ):

        # Deciding whether to append features or simply return generated features
        self.stack_to_X = stack_to_X
        self.sparse_feat = sparse_feat
        self.add_probs = add_probs

        # GBM hyperparameters
        self.criterion = criterion
        self.init = init
        self.learning_rate = learning_rate
        self.loss = loss
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.n_estimators = n_estimators
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.subsample = subsample
        self.tol = tol
        self.validation_fraction = validation_fraction
        self.verbose = verbose
        self.warm_start = warm_start

    def _get_leaves(self, X):
        X_leaves = self.gbm.apply(X)
        n_rows, n_cols, _ = X_leaves.shape
        X_leaves = X_leaves.reshape(n_rows, n_cols)

        return X_leaves

    def _decode_leaves(self, X):
        if self.sparse_feat:
            return scipy.sparse.csr.csr_matrix(self.encoder.transform(X))
        else:
            return self.encoder.transform(X).todense()

    def fit(self, X, y):

        self.gbm = GradientBoostingClassifier(
            criterion=self.criterion,
            init=self.init,
            learning_rate=self.learning_rate,
            loss=self.loss,
            max_depth=self.max_depth,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            min_impurity_split=self.min_impurity_split,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            n_estimators=self.n_estimators,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
            subsample=self.subsample,
            tol=self.tol,
            validation_fraction=self.validation_fraction,
            verbose=self.verbose,
            warm_start=self.warm_start,
        )

        self.gbm.fit(X, y)
        self.encoder = OneHotEncoder(categories="auto")
        X_leaves = self._get_leaves(X)
        self.encoder.fit(X_leaves)
        return self

    def transform(self, X):
        """
        Generates leaves features using the fitted self.gbm and saves them in R.
        If 'self.stack_to_X==True' then '.transform' returns the original features with 'R' appended as columns.
        If 'self.stack_to_X==False' then  '.transform' returns only the leaves features from 'R'
        Ìf 'self.sparse_feat==True' then the input matrix from 'X' is cast as a sparse matrix as well as the 'R' matrix.
        """
        R = self._decode_leaves(self._get_leaves(X))

        if self.sparse_feat:
            if self.add_probs:
                P = self.gbm.predict_proba(X)
                X_new = (
                    scipy.sparse.hstack(
                        (
                            scipy.sparse.csr.csr_matrix(X),
                            R,
                            scipy.sparse.csr.csr_matrix(P),
                        )
                    )
                    if self.stack_to_X == True
                    else R
                )
            else:
                X_new = (
                    scipy.sparse.hstack((scipy.sparse.csr.csr_matrix(X), R))
                    if self.stack_to_X == True
                    else R
                )

        else:

            if self.add_probs:
                P = self.gbm.predict_proba(X)
                X_new = (
                    scipy.sparse.hstack(
                        (
                            scipy.sparse.csr.csr_matrix(X),
                            R,
                            scipy.sparse.csr.csr_matrix(P),
                        )
                    )
                    if self.stack_to_X == True
                    else R
                )
            else:
                X_new = np.hstack((X, R)) if self.stack_to_X == True else R

        return X_new
