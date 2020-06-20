import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class IsEmptyExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that adds columns indicating wether columns have NaN values
    in a row

    Parameters
    ----------
    keep_trivial:
        If a column doesn't have NaN, don't add the column
    cols:
        List of columns to transform. If None, all columns are transformed.
        It only works if input is a DataFrame
    Example
    -------
    >>> from sktools import IsEmptyExtractor
    >>> import pandas as pd
    >>> import numpy as np
    >>> X = pd.DataFrame(
    >>>     {
    >>>         "x": ["a", "b", np.nan],
    >>>         "y": ["c", np.nan, "d"]
    >>>     }
    >>> )
    >>> IsEmptyExtractor().fit_transform(X)
         x    y   x_na   y_na
    0    a    c  False  False
    1    b  NaN  False   True
    2  NaN    d   True  False


    """

    def __init__(self, keep_trivial=False, cols=None):

        self.keep_trivial = keep_trivial
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform_data_frame(self, X):
        """
        Transform method in case of receiving a pandas data frame
        """

        new_x = X.copy()

        if self.cols is None:
            selected_columns = X.columns
        else:
            selected_columns = self.cols

        for column in selected_columns:

            new_column_name = f"{column}_na"

            is_na = X[column].isna()

            if any(is_na) or self.keep_trivial:
                new_x[new_column_name] = is_na

        return new_x

    def transform(self, X):
        """
        For each column, it creates a new one indicating if that column is na
        """

        assert isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray)

        if isinstance(X, pd.DataFrame):
            return self.transform_data_frame(X)

        if isinstance(X, np.ndarray):
            new_x = pd.DataFrame(X.copy())
            transformed_x = self.transform_data_frame(new_x)
            return transformed_x.values
