"""Cyclic transformer"""

__author__ = "david26694"

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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
