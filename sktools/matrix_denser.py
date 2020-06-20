"""Main module."""

from sklearn.base import BaseEstimator, TransformerMixin


class MatrixDenser(BaseEstimator, TransformerMixin):
    """
    Converts matrix to dense.

    Useful when doing an union between dense and sparse matrices.

    Example
    -------
    >>> from sktools import MatrixDenser
    >>> from scipy.sparse import csr_matrix
    >>> import numpy as np
    >>> sparse_matrix = csr_matrix((3, 4), dtype=np.int8)
    >>> dense_matrix = MatrixDenser().fit_transform(sparse_matrix)
    >>> print(dense_matrix)
    [[0 0 0 0]
     [0 0 0 0]
     [0 0 0 0]]
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()
