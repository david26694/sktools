import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer


"""Grouped Quantile Featurizer"""

__author__ = "david26694"


class GroupedQuantileTransformer(BaseEstimator, TransformerMixin):
    """
    Computes the group quantile of a numeric feature with respect to a categorical feature.

    For instance, if each datum is an apartment, and we have both the price and the city,
    this feature tries to model how expensive is an apartment in its city. The most
    expensive apartment in the city will score 1, and the cheapest will score 0.

    It is equivalent at what it is done in:
    https://stackoverflow.com/questions/33899369/ranking-order-per-group-in-pandas

    Parameters
    ----------

    feature_mapping: dict
        mapping from numeric variables to categories that want to be used as groups.
    n_quantiles: int
        number of quantiles per category.
    subsample: int
        Maximum number of samples used to estimate the quantiles for computational efficiency.
    random_state: Any
        Determines random number generation for subsampling and smoothing noise. Please see ``subsample`` for more details.
        Pass an int for reproducible results across multiple function calls. See :term:`Glossary  `
    copy: bool
        Set to False to perform inplace transformation and avoid a copy (if the input is already a numpy array).

    Example
    -------
    >>> from sktools import GroupedQuantileTransformer
    >>> import pandas as pd
    >>> X = pd.DataFrame(
    >>>         {
    >>>             "price": [1, 2, 3, 3, 2, 10, 0],
    >>>             "city": ["a", "a", "a", "b", "b", None, None],
    >>>         }
    >>>     )
    >>> featurizer = GroupedQuantileTransformer(feature_mapping={"price": "city"})
    >>> print(featurizer.fit_transform(X).columns)
    Index(['price', 'city', 'price_quantile_city'], dtype='object')

    """

    def __init__(
        self,
        feature_mapping,
        handle_missing="value",
        n_quantiles=1000,
        subsample=int(1e5),
        random_state=None,
        copy=True,
    ):
        self.transformer_dict = {}
        self.feature_mapping = feature_mapping
        self.transformers = None
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy
        self.handle_missing = handle_missing

    def fit(self, X, y=None):

        for col, group in self.feature_mapping.items():

            self.transformer_dict[group] = {}
            categories = X[group].unique()

            for category in categories:
                # Regular case - non-nulls -> create a quantile transformer
                # and fit it with data in that category
                if category is not None:
                    self.transformer_dict[group][
                        category
                    ] = QuantileTransformer(
                        n_quantiles=self.n_quantiles,
                        subsample=self.subsample,
                        random_state=self.random_state,
                        copy=self.copy,
                    )

                    x_category = X[X[group] == category]
                    self.transformer_dict[group][category].fit(
                        x_category.loc[:, [col]]
                    )

            # Non-regular case -> impute missings by taking the whole distribution
            if self.handle_missing == "value":
                self.transformer_dict[group][np.nan] = QuantileTransformer(
                    n_quantiles=self.n_quantiles,
                    subsample=self.subsample,
                    random_state=self.random_state,
                    copy=self.copy,
                )

                x_category = X
                self.transformer_dict[group][np.nan].fit(
                    x_category.loc[:, [col]]
                )

        return self

    def transform(self, X):

        if self.copy:
            X = X.copy()

        for col, group in self.feature_mapping.items():

            transform_feature_name = f"{col}_quantile_{group}"
            X[transform_feature_name] = np.zeros(X.shape[0])

            categories = X[group].unique()
            fit_categories = self.transformer_dict[group].keys()

            for category in categories:

                # Easy case - regular category
                # Just use transformer_dict to estimate to establish quantile
                # for the features
                if category in fit_categories and category is not np.nan:
                    x_category = X[X[group] == category]
                    x_col = x_category.loc[:, [col]]
                    transformer = self.transformer_dict[group][category]
                    x_transform = transformer.transform(x_col)

                    X.loc[
                        X[group] == category, transform_feature_name
                    ] = x_transform

                # New categories or nulls -> use default transformer
                else:
                    # Keep new and null
                    nonnull_fit_cats = set(fit_categories).difference([np.nan])
                    other_cats_condition = ~X[group].isin(nonnull_fit_cats)
                    x_category = X[other_cats_condition]

                    # Use default transformer
                    transformer = self.transformer_dict[group][np.nan]
                    x_transform = transformer.transform(
                        x_category.loc[:, [col]]
                    )

                    # Assign to X
                    X.loc[
                        other_cats_condition, transform_feature_name
                    ] = x_transform

        return X


class PercentileGroupFeaturizer(BaseEstimator, TransformerMixin):
    """
        Creates features establishing a relationship between a numeric and a categorical feature,
        by using a given percentile of the numeric feature in each cateogry.

        For instance, if each datum is an apartment, and we have both the price and the city,
        if we use the percentile 50 the features model how expensive is an apartment
        with respect to the median in the city.

    Parameters
    ----------

    feature_mapping: dict
        mapping from numeric variables to categories that want to be used as groups.
    percentile: int
        percentile used to compute features
    create_features: bool
        If false, it just computes percentiles by category
    handle_missing: str
        options are 'return_nan' and 'value', defaults to 'value', which uses the global quantile.
    handle_unknown: str
        options are 'return_nan' and 'value', defaults to 'value', which uses the global quantile.

    Example
    -------
    >>> from sktools import PercentileGroupFeaturizer
    >>> import pandas as pd
    >>> X = pd.DataFrame(
    >>>         {
    >>>             "price": [1, 2, 3, 3, 2, 10, 0],
    >>>             "city": ["a", "a", "a", "b", "b", None, None],
    >>>         }
    >>>     )
    >>> featurizer = PercentileGroupFeaturizer(
    >>>     feature_mapping={"price": "city"}
    >>> )
    >>> print(featurizer.fit_transform(X).columns)
    Index(['price', 'city', 'p50_price_city', 'diff_p50_price_city',
           'relu_diff_p50_price_city', 'ratio_p50_price_city'],
          dtype='object')


    """

    def __init__(
        self,
        feature_mapping,
        percentile=50,
        create_features=True,
        handle_missing="value",
        handle_unknown="value",
    ):
        self.feature_mapping = feature_mapping
        self.percentile = percentile
        self.create_features = create_features
        self.handle_missing = handle_missing
        self.handle_unknown = handle_unknown
        self.saved_percentiles = {}

    def fit(self, X, y=None):

        for col, group in self.feature_mapping.items():

            pctl_col_name = f"p{self.percentile}_{col}_{group}"

            # Create percentile by group
            pctl_df = (
                X.groupby(group, as_index=False)
                .agg({col: lambda x: x.quantile(self.percentile / 100)})
                .rename(columns={col: pctl_col_name})
            )

            # Regular handle missing -> add global percentile to missing
            if self.handle_missing == "value":
                global_pctl = X[col].agg(
                    lambda x: x.quantile(self.percentile / 100)
                )

                global_pctl_df = pd.DataFrame(
                    {group: np.nan, pctl_col_name: [global_pctl]}
                )

                pctl_df = pd.concat([pctl_df, global_pctl_df])

            self.saved_percentiles[col] = pctl_df

        return self

    def transform(self, X):

        X = X.copy()

        for col, group in self.feature_mapping.items():

            X = X.merge(self.saved_percentiles[col], on=group, how="left")
            pctl_col_name = f"p{self.percentile}_{col}_{group}"

            # First assign percentiles to non-trivial cases, which are
            # new categories
            if (
                self.handle_unknown == "value"
                and self.handle_missing == "value"
            ):
                groups_fit = self.saved_percentiles[col][group]
                new_condition = (~X[group].isin(groups_fit)) & X[
                    pctl_col_name
                ].isnull()

                x_fit = self.saved_percentiles[col]
                imputation = float(
                    x_fit.loc[x_fit[group].isnull()][pctl_col_name]
                )
                X.loc[new_condition, pctl_col_name] = imputation

            # Then trivially create features
            if self.create_features:
                diff_name = f"diff_p{self.percentile}_{col}_{group}"
                relu_diff_name = f"relu_diff_p{self.percentile}_{col}_{group}"
                ratio_name = f"ratio_p{self.percentile}_{col}_{group}"

                X[diff_name] = X[col] - X[pctl_col_name]
                X[relu_diff_name] = X[diff_name].clip(0, np.inf)
                X[ratio_name] = X[col] / X[pctl_col_name]

        return X


class MeanGroupFeaturizer(BaseEstimator, TransformerMixin):
    """
        Creates features establishing a relationship between a numeric and a categorical feature,
        by using the mean of the numeric feature in each cateogry.

        For instance, if each datum is an apartment, and we have both the price and the city,
        the features model how expensive is an apartment with respect to the mean in the city.

    Parameters
    ----------

    feature_mapping: dict
        mapping from numeric variables to categories that want to be used as groups.
    percentile: int
        percentile used to compute features
    create_features: bool
        If false, it just computes percentiles by category
    handle_missing: str
        options are 'return_nan' and 'value', defaults to 'value', which uses the global quantile.
    handle_unknown: str
        options are 'return_nan' and 'value', defaults to 'value', which uses the global quantile.

    Example
    -------
    >>> from sktools import MeanGroupFeaturizer
    >>> import pandas as pd
    >>> X = pd.DataFrame(
    >>>         {
    >>>             "price": [1, 2, 3, 3, 2, 10, 0],
    >>>             "city": ["a", "a", "a", "b", "b", None, None],
    >>>         }
    >>>     )
    >>> featurizer = MeanGroupFeaturizer(
    >>>     feature_mapping={"price": "city"}
    >>> )
    >>> print(featurizer.fit_transform(X).columns)
    Index(['price', 'city', 'mean_price_city', 'diff_mean_price_city',
           'relu_diff_mean_price_city', 'ratio_mean_price_city'],
          dtype='object')


    """

    def __init__(
        self,
        feature_mapping,
        create_features=True,
        handle_missing="value",
        handle_unknown="value",
    ):
        self.feature_mapping = feature_mapping
        self.create_features = create_features
        self.handle_missing = handle_missing
        self.handle_unknown = handle_unknown
        self.saved_mean = {}

    def fit(self, X, y=None):

        for col, group in self.feature_mapping.items():

            mean_col_name = f"mean_{col}_{group}"

            # Create percentile by group
            mean_df = (
                X.groupby(group, as_index=False)
                .agg({col: lambda x: x.mean()})
                .rename(columns={col: mean_col_name})
            )

            # Regular handle missing -> add global percentile to missing
            if self.handle_missing == "value":
                global_mean = X[col].agg(lambda x: x.mean())

                global_mean_df = pd.DataFrame(
                    {group: np.nan, mean_col_name: [global_mean]}
                )

                mean_df = pd.concat([mean_df, global_mean_df])

            self.saved_mean[col] = mean_df

        return self

    def transform(self, X):

        X = X.copy()

        for col, group in self.feature_mapping.items():

            X = X.merge(self.saved_mean[col], on=group, how="left")
            mean_col_name = f"mean_{col}_{group}"

            # First assign percentiles to non-trivial cases, which are
            # new categories
            if (
                self.handle_unknown == "value"
                and self.handle_missing == "value"
            ):
                groups_fit = self.saved_mean[col][group]
                new_condition = (~X[group].isin(groups_fit)) & X[
                    mean_col_name
                ].isnull()

                x_fit = self.saved_mean[col]
                imputation = float(
                    x_fit.loc[x_fit[group].isnull()][mean_col_name]
                )
                X.loc[new_condition, mean_col_name] = imputation

            # Then trivially create features
            if self.create_features:
                diff_name = f"diff_mean_{col}_{group}"
                relu_diff_name = f"relu_diff_mean_{col}_{group}"
                ratio_name = f"ratio_mean_{col}_{group}"

                X[diff_name] = X[col] - X[mean_col_name]
                X[relu_diff_name] = X[diff_name].clip(0, np.inf)
                X[ratio_name] = X[col] / X[mean_col_name]

        return X
