import scipy
import random
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import MinMaxScaler

random.seed(0)


class NewtonRegressor(BaseEstimator, ClassifierMixin):
    """
    from make_data import *

    G = 100
    data = make_newton(samples=1_000, G=G)

    X = data.drop(columns="f")
    y = data.f

    nr = NewtonRegressor(G=G)

    nr.fit(X, y)
    nr.predict(X)

    """

    def __init__(self, G=scipy.constants.G):
        """

        :param G: Gravitational Constant
        """
        self.G = G

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """

        :param X: dataset with columns: m1,m2, and r
        :return: Newtons gravitational force
        """

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Dataframe and numpy data handling
        if isinstance(X, pd.DataFrame):
            return (
                self.G
                * X.values[:, 0]
                * X.values[:, 1]
                / (X.values[:, 2] * X.values[:, 2])
            )
        if isinstance(X, np.ndarray):
            return self.G * X[:, 0] * X[:, 1] / (X[:, 2] * X[:, 2])


class NewtonClassifier(BaseEstimator, ClassifierMixin):
    """
    from make_data import *

    G = 100
    data = make_newton(samples=1_000, G=G)

    X = data.drop(columns="f")
    y = data.f
    threshold = np.mean(y)

    nc = NewtonClassifier(threshold=threshold, G=G)

    nc.fit(X, y)
    nc.predic(X)
    nc.predict_proba(X)
    """

    def __init__(self, threshold=0.5, G=scipy.constants.G):
        self.threshold = threshold
        self.G = G

    def convert_target(self, y):
        """
        Convert a continous target into binary classification based on threshold

        :param y: continous target
        :return: binarized target
        """
        return [1 if a_ > self.threshold else 0 for a_ in y]

    def fit(self, X, y):
        """
        Checks sizes of X and y and handles Nan and Inf
        :param X:
        :param y:
        :return:
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Check if there are nans
        if isinstance(X, pd.DataFrame):
            X = X.fillna(0)
        if isinstance(X, np.ndarray):
            X = np.nan_to_num(X)
        if isinstance(y, pd.DataFrame):
            y = y.fillna(0)
        if isinstance(y, np.ndarray):
            y = np.nan_to_num(y)

        self.X_ = X
        self.y = y

        # If the target has more than 3 uniqeu values we binarize it.
        if len(np.unique(y)) < 3:
            self.y_c_ = self.convert_target(y)
        else:
            self.y_c = self.y

        # Fitting scaler on the target
        self.scaler = MinMaxScaler().fit(y.reshape(-1, 1))
        return self

    def predict(self, X):
        """

        :param X: Data
        :return: Binary predictions
        """

        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        threshold = self.threshold

        # Calculate Newton's Eq and classify by threshold
        if isinstance(X, pd.DataFrame):
            eq = (
                self.G
                * X.values[:, 0]
                * X.values[:, 1]
                / (X.values[:, 2] * X.values[:, 2])
            )
            return [1 if a_ > threshold else 0 for a_ in eq]
        if isinstance(X, np.ndarray):
            eq = self.G * X[:, 0] * X[:, 1] / (X[:, 2] * X[:, 2])
            return [1 if a_ > threshold else 0 for a_ in eq]

    def predict_proba(self, X):
        """
        Predicts probabilities based on sigmoid function

        :param X: Newton data
        :return: array with probabilites
        """

        # Make predictions
        y_pred = self.predict_regression(X)

        # Normalize predictions to make it probabilities

        # In case prediction is a only element
        if len(y_pred) > 1:
            return np.array(
                np.exp(y_pred - self.threshold) / (1 + np.exp(y_pred - self.threshold))
            ).reshape(-1, 1)
        else:
            return np.array(
                np.exp(y_pred - self.threshold) / (1 + np.exp(y_pred - self.threshold))
            ).reshape(1, -1)

    def predict_regression(self, X):

        """

        :param X: dataset with columns: m1,m2, and r
        :return: Newtons gravitational force
        """

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Dataframe and numpy data handling
        if isinstance(X, pd.DataFrame):
            return (
                self.G
                * X.values[:, 0]
                * X.values[:, 1]
                / (X.values[:, 2] * X.values[:, 2])
            )
        if isinstance(X, np.ndarray):
            return self.G * X[:, 0] * X[:, 1] / (X[:, 2] * X[:, 2])

