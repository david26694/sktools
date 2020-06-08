# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.m_estimate import MEstimateEncoder
import category_encoders.utils as util
from sklearn.utils.random import check_random_state

"""Nested target encoder"""

__author__ = "david26694","cmougan"


class NestedTargetEncoder(BaseEstimator, util.TransformerWithTargetMixin):
    """Estimate of likelihood for nested data.

    This is a generalization of the m-probability estimate. The main difference
    is that instead of using a global prior, it can use a more fine-tuned prior.
    This only works for nested data. For instance, I have individuals who live
    in counties, that are inside states. If I want to estimate the likelihood
    encoding for a county, it is better to use as prior the estimate for the
    state instead of the global estimate.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop encoded columns with 0 variance.
    feature_mapping: dict
        dictionary representing the child - parent relationship. keys are children.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    handle_missing: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which returns the prior probability.
    handle_unknown: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which returns the prior probability.
    randomized: bool,
        adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
    sigma: float
        standard deviation (spread or "width") of the normal distribution.
    m_prior: float
        this is the "m" in the m-probability estimate for the global mean. Higher value of m results into stronger shrinking.
        It is used whenever we estimate a likelihood using the global mean as a prior.
        M is non-negative.
    m_parent: float
        this is the "m" in the m-probability estimate. Higher value of m results into stronger shrinking.
        It is used whenever we estimate a likelihood using the parent mean as a prior.
        M is non-negative.

    Example
    -------
    >>> from sktools import NestedTargetEncoder
    >>> import pandas as pd
    >>> X = pd.DataFrame(
    >>>     {
    >>>         "child": ["a", "a", "b", "b", "b", "c", "c", "d", "d", "d"],
    >>>         "parent": ["e", "e", "e", "e", "e", "f", "f", "f", "f", "f",]
    >>>     }
    >>> )
    >>> y = pd.Series([1, 2, 3, 1, 2, 4, 4, 5, 4, 4.5])
    >>> ne = NestedTargetEncoder(feature_mapping={"child": "parent"}, m_prior=0)
    >>> ne.fit_transform(X, y)
          child  parent
    0  2.016667     1.8
    1  2.016667     1.8
    2  2.262500     1.8
    3  2.262500     1.8
    4  2.262500     1.8
    5  3.683333     4.3
    6  3.683333     4.3
    7  4.137500     4.3
    8  4.137500     4.3
    9  4.137500     4.3

    References
    ----------

    .. [1] Additive smoothing, from https://en.wikipedia.org/wiki/Additive_smoothing#Generalized_to_the_case_of_known_incidence_rates

    """

    def __init__(
        self,
        verbose=0,
        cols=None,
        drop_invariant=False,
        feature_mapping={},
        return_df=True,
        handle_unknown="value",
        handle_missing="value",
        random_state=None,
        randomized=False,
        sigma=0.05,
        m_prior=1.0,
        m_parent=1.0,
    ):
        self.verbose = verbose
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.cols = cols
        self.feature_mapping = feature_mapping
        self.ordinal_encoder = None
        self._dim = None
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self._sum = None
        self._count = None
        self.random_state = random_state
        self.randomized = randomized
        self.sigma = sigma
        self.m_prior = m_prior
        self.m_parent = m_parent
        self.feature_names = None
        self.parent_cols = None
        self.parent_encoder = None

    # noinspection PyUnusedLocal
    def fit(self, X, y, **kwargs):
        """Fit encoder according to X and binary y.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Binary target values.

        Returns
        -------

        self : encoder
            Returns self.

        """

        # Create parent encoder and fit it
        self.parent_cols = list(self.feature_mapping.values())
        self.parent_encoder = MEstimateEncoder(
            verbose=self.verbose,
            cols=self.parent_cols,
            drop_invariant=self.drop_invariant,
            return_df=self.return_df,
            handle_unknown=self.handle_unknown,
            handle_missing=self.handle_missing,
            random_state=self.random_state,
            randomized=self.randomized,
            sigma=self.sigma,
            m=self.m_prior,
        )
        self.parent_encoder.fit(X, y)

        # Unite parameters into pandas types
        X = util.convert_input(X)
        y = util.convert_input_vector(y, X.index).astype(float)

        # The lengths must be equal
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "The length of X is "
                + str(X.shape[0])
                + " but length of y is "
                + str(y.shape[0])
                + "."
            )

        self._dim = X.shape[1]

        # If columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        if self.handle_missing == "error":
            if X[self.cols].isnull().any().any():
                raise ValueError("Columns to be encoded can not contain null")

        # Check that children and parents are disjoint
        children = set(self.feature_mapping.keys())
        parents = set(self.feature_mapping.values())
        if len(children.intersection(parents)) > 0:
            raise ValueError("No column should be a child and a parent")

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown="value",
            handle_missing="value",
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)

        # Training
        self.mapping = self._train(X_ordinal, y)

        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = X_temp.columns.tolist()

        # Store column names with approximately constant variance on the training data
        if self.drop_invariant:
            self.drop_cols = []
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [
                x for x in generated_cols if X_temp[x].var() <= 10e-5
            ]
            try:
                [self.feature_names.remove(x) for x in self.drop_cols]
            except KeyError as e:
                if self.verbose > 0:
                    print(
                        "Could not remove column from feature names."
                        "Not found in generated cols.\n{}".format(e)
                    )
        return self

    def transform(self, X, y=None, override_return_df=False):
        """Perform the transformation to new categorical data.

        When the data are used for model training, it is important to also pass the target in order to apply leave one out.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples] when transform by leave one out
            None, when transform without target information (such as transform test set)


        Returns
        -------

        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """

        if self.handle_missing == "error":
            if X[self.cols].isnull().any().any():
                raise ValueError("Columns to be encoded can not contain null")

        if self._dim is None:
            raise ValueError(
                "Must train encoder before it can be used to transform data."
            )

        # Unite the input into pandas types
        X = util.convert_input(X)

        # Then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError(
                "Unexpected input dimension %d, expected %d"
                % (X.shape[1], self._dim,)
            )

        # If we are encoding the training data, we have to check the target
        if y is not None:
            y = util.convert_input_vector(y, X.index).astype(float)
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    "The length of X is "
                    + str(X.shape[0])
                    + " but length of y is "
                    + str(y.shape[0])
                    + "."
                )

        if not list(self.cols):
            return X

        # Do not modify the input argument
        X = X.copy(deep=True)

        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == "error":
            if X[self.cols].isin([-1]).any().any():
                raise ValueError("Unexpected categories found in dataframe")

        # Loop over the columns and replace the nominal values with the numbers
        X = self._score(X, y)

        # Postprocessing
        # Note: We should not even convert these columns.
        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    def _train(self, X, y):
        # Initialize the output
        mapping = {}

        # Calculate global statistics
        self._sum = y.sum()
        self._count = y.count()
        prior = self._sum / self._count

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get("col")
            values = switch.get("mapping")

            # Easy case, the child is not in the child - parent dictionary.
            # We just use the plain m-estimator with the global prior
            if col not in self.feature_mapping:
                stats = y.groupby(X[col]).agg(["sum", "count", "mean"])

                estimate = (stats["sum"] + prior * self.m_prior) / (
                    stats["count"] + self.m_prior
                )

            # Not so easy case, we have to deal with the parent
            else:
                parent_col = self.feature_mapping[col]

                # Check son-parent unique relation
                unique_parents = X.groupby([col]).agg({parent_col: "nunique"})[
                    parent_col
                ]

                more_1_parent = unique_parents[unique_parents > 1]

                if any(unique_parents > 1) and more_1_parent.index >= 0:
                    raise ValueError(
                        f"There are children with more than one parent, {more_1_parent}"
                    )

                # Get parent stats
                te_parent = self.parent_encoder.transform(X)[parent_col]
                parent_mapping = pd.DataFrame(
                    {"te_parent": te_parent, parent_col: X[parent_col]}
                ).drop_duplicates()

                # Compute child statistics
                stats = y.groupby(X[col]).agg(["sum", "count", "mean"])

                # Relate parent and child stats
                groups = X.loc[:, [parent_col, col]].drop_duplicates()

                stats = stats.merge(groups, how="left", on=col).merge(
                    parent_mapping, how="left", on=parent_col
                )

                # In case of numpy array
                stats = stats.rename(columns={"key_0": col})
                stats = stats.set_index(col)

                # Calculate the m-probability estimate using the parent prior
                estimate = (
                    stats["sum"] + stats["te_parent"] * self.m_parent
                ) / (stats["count"] + self.m_parent)

            # Ignore unique columns. This helps to prevent overfitting on id-like columns
            if len(stats["count"]) == self._count:
                estimate[:] = prior

            # Column doesn't have parent - handle imputation as always
            if col not in self.feature_mapping:
                if self.handle_unknown == "return_nan":
                    estimate.loc[-1] = np.nan
                elif self.handle_unknown == "value":
                    estimate.loc[-1] = prior

                if self.handle_missing == "return_nan":
                    estimate.loc[values.loc[np.nan]] = np.nan
                elif self.handle_missing == "value":
                    estimate.loc[-2] = prior
            # With parents - we leave the imputation for afterwards
            else:
                # Unknown
                estimate.loc[-1] = np.nan
                # Missing
                estimate.loc[values.loc[np.nan]] = np.nan

            # Store the m-probability estimate for transform() function
            mapping[col] = estimate

        return mapping

    def _score(self, X, y):

        X_parents = self.parent_encoder.transform(X)

        for col in self.cols:

            # Easy case - not having parents (as m estimator)
            if col not in self.feature_mapping:
                # Score the column
                X[col] = X[col].map(self.mapping[col])

            # Harder case - having parents
            else:

                # Split missing and unknown values
                unknown = X[col] == -1
                missing = X[col] == -2

                # Apply regular transformation
                X[col] = X[col].map(self.mapping[col].drop_duplicates())

                # Impute unknown with parent
                parent_col = self.feature_mapping[col]
                if self.handle_unknown == "value":
                    X[col] = X[col].mask(unknown, X_parents[parent_col])

                # Impute missing with parent
                if self.handle_missing == "value":
                    X[col] = X[col].mask(missing, X_parents[parent_col])

            # Randomization is meaningful only for training data -> we do it only if y is present
            if self.randomized and y is not None:
                random_state_generator = check_random_state(self.random_state)
                X[col] = X[col] * random_state_generator.normal(
                    1.0, self.sigma, X[col].shape[0]
                )

        return X

    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.

        Returns
        -------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!

        """
        if not isinstance(self.feature_names, list):
            raise ValueError(
                "Estimator has to be fitted to return feature names."
            )
        else:
            return self.feature_names


"""Percentile Encoder"""

__author__ = "cmougan & david26694"


class PercentileEncoder(BaseEstimator, util.TransformerWithTargetMixin):
    """Percentile Encoding for categorical features.

    For the case of categorical target: features are replaced with a blend of
    posterior percentile of the target given particular categorical value and
    the prior probability of the target over all the training data.

    For the case of continuous target: features are replaced with a blend of the
    expected value of the target given particular categorical value and the
    expected value of the target over all the training data.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    handle_missing: str
        options are 'error', 'return_nan'  and 'value', defaults to 'value', which returns the target percentile.
    handle_unknown: str
        options are 'error', 'return_nan' and 'value', defaults to 'value', which returns the target percentile.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = PercentileEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 13 columns):
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    CHAS       506 non-null float64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    RAD        506 non-null float64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(13)
    memory usage: 51.5 KB
    None

    References
    ----------

    """

    def __init__(
        self,
        verbose=0,
        cols=None,
        drop_invariant=False,
        return_df=True,
        handle_missing="value",
        handle_unknown="value",
        percentile=50,
    ):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self.ordinal_encoder = None
        self._dim = None
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self._percentile = None
        self.feature_names = None
        self.percentile = percentile

    def fit(self, X, y, **kwargs):
        """Fit encoder according to X and y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : encoder
            Returns self.

        """

        # unite the input into pandas types
        X = util.convert_input(X)
        y = util.convert_input_vector(y, X.index)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "The length of X is "
                + str(X.shape[0])
                + " but length of y is "
                + str(y.shape[0])
                + "."
            )

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        if self.handle_missing == "error":
            if X[self.cols].isnull().any().any():
                raise ValueError("Columns to be encoded can not contain null")

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown="value",
            handle_missing="value",
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)
        self.mapping = self.fit_percentile_encoding(X_ordinal, y)

        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = list(X_temp.columns)

        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [
                x for x in generated_cols if X_temp[x].var() <= 10e-5
            ]
            try:
                [self.feature_names.remove(x) for x in self.drop_cols]
            except KeyError as e:
                if self.verbose > 0:
                    print(
                        "Could not remove column from feature names."
                        "Not found in generated cols.\n{}".format(e)
                    )

        return self

    def fit_percentile_encoding(self, X, y):
        mapping = {}

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get("col")
            values = switch.get("mapping")

            prior = self._percentile = np.percentile(y, self.percentile)

            stats = y.groupby(X[col]).apply(
                lambda x: np.percentile(x, self.percentile)
            )

            smoothing = stats

            if self.handle_unknown == "return_nan":
                smoothing.loc[-1] = np.nan
            elif self.handle_unknown == "value":
                smoothing.loc[-1] = prior

            if self.handle_missing == "return_nan":
                smoothing.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == "value":
                smoothing.loc[-2] = prior

            mapping[col] = smoothing

        return mapping

    def transform(self, X, y=None, override_return_df=False):
        """Perform the transformation to new categorical data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples] when transform by leave one out
            None, when transform without target info (such as transform test set)

        Returns
        -------
        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """

        if self.handle_missing == "error":
            if X[self.cols].isnull().any().any():
                raise ValueError("Columns to be encoded can not contain null")

        if self._dim is None:
            raise ValueError(
                "Must train encoder before it can be used to transform data."
            )

        # unite the input into pandas types
        X = util.convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError(
                "Unexpected input dimension %d, expected %d"
                % (X.shape[1], self._dim,)
            )

        # if we are encoding the training data, we have to check the target
        if y is not None:
            y = util.convert_input_vector(y, X.index)
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    "The length of X is "
                    + str(X.shape[0])
                    + " but length of y is "
                    + str(y.shape[0])
                    + "."
                )

        if not list(self.cols):
            return X

        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == "error":
            if X[self.cols].isin([-1]).any().any():
                raise ValueError("Unexpected categories found in dataframe")

        X = self.percentile_encode(X)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    def percentile_encode(self, X_in):
        X = X_in.copy(deep=True)

        for col in self.cols:
            X[col] = X[col].map(self.mapping[col])

        return X

    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.

        Returns
        -------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!

        """

        if not isinstance(self.feature_names, list):
            raise ValueError(
                "Must fit data first. Affected feature names are not known "
                "before."
            )
        else:
            return self.feature_names
