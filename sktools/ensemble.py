import numpy as np
from sklearn.ensemble import RandomForestRegressor


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
