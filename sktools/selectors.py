import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TypeSelector(BaseEstimator, TransformerMixin):
    """
    Transformer that filters a type of columns of a given data frame. This can
    be useful if we want to treat numeric and object columns differently


    Parameters
    ----------
    dtype : required
        The type we want to filter

    Example
    -------
    >>> from sktools import TypeSelector
    >>> import pandas as pd
    >>> X = pd.DataFrame(
    >>>         {
    >>>             "price": [1., 2., 3.],
    >>>             "city": ["a", "a", "b"]
    >>>         }
    >>>     )
    >>> selector = TypeSelector(
    >>>     dtype='float'
    >>> )
    >>> print(selector.fit_transform(X))
        price
    0    1.0
    1    2.0
    2    3.0
    """

    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >>> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrices (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >>> data = {'a': [1, 5, 2, 5, 2, 8],
                'b': [9, 4, 1, 4, 1, 3]}
    >>> ds = ItemSelector(key='a')
    >>> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]
