"""Main module."""

from sklearn.base import BaseEstimator, TransformerMixin


class MatrixDenser(BaseEstimator, TransformerMixin):
    """
    Transformer converts matrix to dense
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()
