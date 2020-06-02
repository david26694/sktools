import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer

from rich.console import Console
from rich.traceback import install

console = Console()
install()


class GroupedQuantileTransformer(BaseEstimator, TransformerMixin):
    '''
    Transformer that computes the group quantile of a feature
    '''

    def __init__(self, feature_mapping, n_quantiles=1000,
                 output_distribution='uniform',
                 ignore_implicit_zeros=False, subsample=int(1e5),
                 random_state=None, copy=True):
        self.transformer_dict = {}
        self.feature_mapping = feature_mapping
        self.transformers = None
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy

    def fit(self, X, y=None):

        for col, group in self.feature_mapping.items():

            self.transformer_dict[group] = {}
            categories = X[group].unique()

            for category in categories:
                if category is not None:
                    self.transformer_dict[group][category] = \
                        QuantileTransformer(
                            n_quantiles=self.n_quantiles,
                            output_distribution=self.output_distribution,
                            ignore_implicit_zeros=self.ignore_implicit_zeros,
                            subsample=self.subsample,
                            random_state=self.random_state,
                            copy=self.copy
                        )

                    x_category = X[X[group] == category]
                    self.transformer_dict[group][category].fit(
                        x_category.loc[:, [col]]
                    )
                else:
                    self.transformer_dict[group][np.nan] = \
                        QuantileTransformer(
                            n_quantiles=self.n_quantiles,
                            output_distribution=self.output_distribution,
                            ignore_implicit_zeros=self.ignore_implicit_zeros,
                            subsample=self.subsample,
                            random_state=self.random_state,
                            copy=self.copy
                        )

                    x_category = X
                    self.transformer_dict[group][np.nan].fit(
                        x_category.loc[:, [col]]
                    )

        return self

    def transform(self, X):

        X = X.copy()

        for col, group in self.feature_mapping.items():

            transform_feature_name = f'{col}_quantile_{group}'
            X[transform_feature_name] = np.zeros(X.shape[0])

            categories = X[group].unique()
            fit_categories = self.transformer_dict[group].keys()

            for category in categories:
                if category is not np.nan and category in fit_categories:
                    x_category = X[X[group] == category]
                    x_col = x_category.loc[:, [col]]
                    transformer = self.transformer_dict[group][category]
                    x_transform = transformer.transform(x_col)

                    X.loc[
                        X[group] == category, transform_feature_name
                    ] = x_transform

                else:
                    nonnull_fit_cats = set(fit_categories).difference([np.nan])
                    other_cats_condition = ~X[group].isin(nonnull_fit_cats)

                    x_category = X[other_cats_condition]
                    transformer = self.transformer_dict[group][np.nan]

                    x_transform = transformer.transform(
                        x_category.loc[:, [col]]
                    )

                    X.loc[other_cats_condition, transform_feature_name] = \
                        x_transform

        return X


class PercentileGroupFeaturizer(BaseEstimator, TransformerMixin):

    def __init__(self, feature_mapping, percentile=50, create_features=True,
                 handle_missing='value', handle_unknown='value'):
        self.feature_mapping = feature_mapping
        self.percentile = percentile
        self.create_features = create_features
        self.handle_missing = handle_missing
        self.handle_unknown = handle_unknown
        self.saved_percentiles = {}

    def fit(self, X, y=None):

        for col, group in self.feature_mapping.items():
            pctl_col_name = f'p{self.percentile}_{col}_{group}'

            pctl_df = X.groupby(group, as_index=False).agg(
                {col: lambda x: x.quantile(self.percentile / 100)}
            ).rename(columns={col: pctl_col_name})

            # If handle missing is value -> add global percentile to missing
            if self.handle_missing == 'value':
                global_pctl = X[col].agg(
                    lambda x: x.quantile(self.percentile / 100)
                )

                global_pctl_df = pd.DataFrame(
                    {
                        group: np.nan,
                        pctl_col_name: [global_pctl]
                    }
                )

                pctl_df = pd.concat([pctl_df, global_pctl_df])

            self.saved_percentiles[col] = pctl_df

        return self

    def transform(self, X):

        for col, group in self.feature_mapping.items():

            X = X.merge(self.saved_percentiles[col], on=group, how='left')
            pctl_col_name = f'p{self.percentile}_{col}_{group}'

            if self.handle_unknown == 'value' and self.handle_missing == 'value':
                groups_fit = self.saved_percentiles[col][group]
                new_condition = (~X[group].isin(groups_fit)) & \
                                X[pctl_col_name].isnull()

                x_fit = self.saved_percentiles[col]
                imputation = float(x_fit.loc[x_fit[group].isnull()][pctl_col_name])
                X.loc[new_condition, pctl_col_name] = imputation

            if self.create_features:
                diff_name = f'diff_p{self.percentile}_{col}_{group}'
                relu_diff_name = f'relu_diff_p{self.percentile}_{col}_{group}'
                ratio_name = f'ratio_p{self.percentile}_{col}_{group}'

                X[diff_name] = X[col] - X[pctl_col_name]
                X[relu_diff_name] = X[diff_name].clip(0, np.inf)
                X[ratio_name] = X[col] / X[pctl_col_name]

        return X
