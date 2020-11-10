import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import scipy

class MedianForestRegressor:
    """Random forest with median aggregation

    Very similar to random forest regressor, but aggregating using the median
    instead of the mean. Can improve the mean absolute error a little.

    Example
    -------
    >>> from sktools import MedianForestRegressor
    >>> from sklearn.datasets import load_boston
    >>> boston = load_boston()['data']
    >>> y = load_boston()['target']
    >>> mf = MedianForestRegressor()
    >>> mf.fit(boston, y)
    >>> mf.predict(boston)[0:10]
    array([24. , 21.6, 34.7, 33.4, 36.2, 28.7, 22.9, 27.1, 16.5, 18.9])

    """

    def __init__(self, *args, **kwargs):

        self.rf = RandomForestRegressor(*args, **kwargs)

    def fit(self, X, y):

        self.rf.fit(X, y)

        return self

    def predict(self, X):

        tree_predictions = [tree.predict(X) for tree in self.rf.estimators_]
        median_tree_pred = np.median(np.array(tree_predictions), axis=0)

        return median_tree_pred


class GradientBoostingFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Feature generator from a gradient boosting

    References:
        - Practical Lessons from Predicting Clicks on Ads at Facebook

    Example
    -------
    from sktools import GradientBoostingFeatureGenerator
    from sklearn.datasets import load_boston
    boston = load_boston()['data']
    y = load_boston()['target']
    y = np.where(y>y.mean(),1,0)
    mf = GradientBoostingFeatureGenerator(sparse_feat=False)
    mf.fit(boston, y)
    mf.transform(boston)

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
        presort="auto",
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
        self.presort = presort
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
            presort=self.presort,
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
