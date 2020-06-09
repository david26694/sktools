#!/usr/bin/env python

"""Tests for `sktools` package."""

import unittest

import sktools
import pandas as pd
from scipy.sparse import csr_matrix
from category_encoders import MEstimateEncoder
import numpy as np


class TestTypeSelector(unittest.TestCase):
    """Tests for type selector."""

    def setUp(self):
        """Create dataframe with different column types"""
        self.df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "char_col": ["a", "b", "c"],
                "other_char_col": ["d", "e", "f"],
            }
        )

    def run_test_type(self, dtype):
        """
        This test applies the fit and transform methods with a given type.
        It then asserts that the type of each column is the same as the input


        @param dtype: Transformer is built using this type
        """

        type_transformer = sktools.TypeSelector(dtype)
        type_cols = type_transformer.fit_transform(self.df)
        output_types = type_cols.dtypes

        for type_col in output_types:
            self.assertEqual(type_col, dtype)

    def test_integer_works(self):
        self.run_test_type("int64")

    def test_object_works(self):
        self.run_test_type("object")

    def test_float_works(self):
        self.run_test_type("float64")


class TestItemSelector(unittest.TestCase):
    """Tests for item selector."""

    def setUp(self):
        """Create dataframe with different column types"""
        self.df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "char_col": ["a", "b", "c"],
                "other_char_col": ["d", "e", "f"],
            }
        )

    def test_select_items(self):
        """
        Check that item selector works for each column, that is:
        * Name is the same as the column of the dataframe
        * Values are the same as the values in the dataframe
        """

        for col in self.df.columns:
            col_transformer = sktools.ItemSelector(col)
            output_transformer = col_transformer.fit_transform(self.df)
            output_column = output_transformer.name

            self.assertEqual(output_column, col)

            self.assertTrue(
                (output_transformer == self.df[col]).all(),
                "Not all values of the series are equal",
            )


class TestMatrixDenser(unittest.TestCase):
    """Tests for item selector."""

    def setUp(self):
        """Create dataframe with different column types"""
        self.sparse_matrix = csr_matrix((3, 4), dtype=np.int8)

    def test_zero_matrix(self):
        dense_matrix = sktools.MatrixDenser().fit_transform(self.sparse_matrix)

        expected_dense = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int8
        )

        self.assertTrue(
            (dense_matrix == expected_dense).all(), "Not all values are 0"
        )


class TestEmptyExtractor(unittest.TestCase):
    """Tests for item selector."""

    def setUp(self):
        """Create dataframe with different column types"""
        self.df = pd.DataFrame(
            {
                "int_col": [1, 2, np.NaN],
                "float_col": [1.1, np.NaN, 3.3],
                "char_col": [np.NaN, "b", "c"],
                "other_char_col": ["d", "e", "f"],
            }
        )
        self.expected_output = pd.DataFrame(
            {
                "int_col": [1, 2, np.NaN],
                "float_col": [1.1, np.NaN, 3.3],
                "char_col": [np.NaN, "b", "c"],
                "other_char_col": ["d", "e", "f"],
                "int_col_na": [False, False, True],
                "float_col_na": [False, True, False],
                "char_col_na": [True, False, False],
                "other_char_col_na": [False, False, False],
            }
        )

    def test_defaults(self):
        pd.testing.assert_frame_equal(
            sktools.IsEmptyExtractor().fit_transform(self.df),
            self.expected_output.drop("other_char_col_na", axis=1),
        )

    def test_non_delete(self):
        pd.testing.assert_frame_equal(
            sktools.IsEmptyExtractor(keep_trivial=True).fit_transform(self.df),
            self.expected_output,
        )


class TestPercentileEncoder(unittest.TestCase):
    """Tests for percentile encoder."""

    def setUp(self):
        """Create dataframe with categories and a target variable"""

        self.df = pd.DataFrame(
            {"categories": ["a", "b", "c", "a", "b", "c", "a", "b"]}
        )
        self.target = np.array([1, 2, 0, 4, 5, 0, 6, 7])

    def test_median_works(self):
        """
        Expected output of percentile 50 in df:
            - a median is 4 (a values are 1, 4, 6)
            - b median is 5 (b values are 2, 5, 7)
            - c median is 0 (c values are 0)
        """

        expected_output_median = pd.DataFrame(
            {"categories": [4.0, 5, 0, 4, 5, 0, 4, 5]}
        )

        pd.testing.assert_frame_equal(
            sktools.PercentileEncoder(percentile=50).fit_transform(
                self.df, self.target
            ),
            expected_output_median,
        )

    def test_max_works(self):
        """
        Expected output of percentile 100 in df:
            - a max is 6
            - b max is 7
            - c max is 0
        """
        expected_output_max = pd.DataFrame(
            {"categories": [6.0, 7, 0, 6, 7, 0, 6, 7]}
        )

        pd.testing.assert_frame_equal(
            sktools.PercentileEncoder(percentile=100).fit_transform(
                self.df, self.target
            ),
            expected_output_max,
        )

    def test_new_category(self):
        """
        The global median of the target is 3. If new categories are passed to
        the transformer, then the output should be 3
        """
        transformer_median = sktools.PercentileEncoder(percentile=50)
        transformer_median.fit(self.df, self.target)

        new_df = pd.DataFrame({"categories": ["d", "e"]})

        new_medians = pd.DataFrame({"categories": [3.0, 3.0]})

        pd.testing.assert_frame_equal(
            transformer_median.transform(new_df), new_medians
        )


class TestNestedTargetEncoder(unittest.TestCase):
    """Tests for nested target encoder."""

    def setUp(self):
        """Create dataframe with categories and a target variable"""

        self.col = "col_1"
        self.parent_col = "parent_col_1"
        self.X = pd.DataFrame(
            {
                self.col: ["a", "a", "b", "b", "b", "c", "c", "d", "d", "d"],
                self.parent_col: [
                    "e",
                    "e",
                    "e",
                    "e",
                    "e",
                    "f",
                    "f",
                    "f",
                    "f",
                    "f",
                ],
            }
        )

        self.X_array = pd.DataFrame(
            {
                self.col: ["a", "a", "b", "b", "b", "c", "c", "d", "d", "d"],
                self.parent_col: [
                    "e",
                    "e",
                    "e",
                    "e",
                    "e",
                    "f",
                    "f",
                    "f",
                    "f",
                    "f",
                ],
            }
        ).values

        self.y = pd.Series([1, 2, 3, 1, 2, 4, 4, 5, 4, 4.5])

        self.parent_means = list(self.y.groupby(self.X[self.parent_col]).mean())
        self.parents = ["e", "f"]

    def test_parent_prior(self):
        """
        Simple case:
        There is no prior from the global to the group mean (m_prior = 0).
        As the m_parent is 1, the mean for group a is (as mean_group_e = 1.8):
        (1 + 2 + mean_group_e ) / 3 = (1 + 2 + 1.8) / 3 = 1.6
        The same works for b, c and d
        """
        expected_output = pd.DataFrame(
            dict(
                col_1=[1.6, 1.6, 1.95, 1.95, 1.95, 4.1, 4.1, 4.45, 4.45, 4.45],
                parent_col_1=self.X[self.parent_col],
            )
        )

        te = sktools.NestedTargetEncoder(
            cols=self.col,
            feature_mapping=dict(col_1=self.parent_col),
            m_prior=0,
        )
        pd.testing.assert_frame_equal(
            te.fit_transform(self.X, self.y), expected_output
        )

    def test_numpy_array(self):
        """
        Check that nested target encoder also works for numpy arrays
        """
        expected_output = pd.DataFrame(
            dict(
                col_1=[1.6, 1.6, 1.95, 1.95, 1.95, 4.1, 4.1, 4.45, 4.45, 4.45],
                parent_col_1=self.X[self.parent_col],
            )
        ).values

        te = sktools.NestedTargetEncoder(
            cols=0, feature_mapping={0: 1}, m_prior=0
        )

        te.fit(self.X_array, self.y)
        output = te.transform(self.X_array).values

        np.testing.assert_almost_equal(output[:, 0], expected_output[:, 0])

        np.testing.assert_equal(output[:, 1], expected_output[:, 1])

    def test_no_parent(self):
        """
        When using no priors, the functionalities should be the same as for
        m estimator.
        """

        te = sktools.NestedTargetEncoder(
            cols=self.col,
            feature_mapping=dict(col_1=self.parent_col),
            m_prior=0,
            m_parent=0,
        )

        m_te = MEstimateEncoder(cols=self.col, m=0)
        pd.testing.assert_frame_equal(
            te.fit_transform(self.X, self.y), m_te.fit_transform(self.X, self.y)
        )

    def test_unknown_missing_imputation(self):
        """
        When new categories and unknown values are given, we expect the encoder
        to give the parent means (at least with default configuration).
        """

        # First two rows are new categories
        # Last two rows are missing values
        # Parents are e, f, e, f
        new_x = pd.DataFrame(
            {
                self.col: ["x", "y", np.NaN, np.NaN],
                self.parent_col: self.parents + self.parents,
            }
        )

        # We expect to get parent means
        expected_output_df = pd.DataFrame(
            {
                self.col: self.parent_means + self.parent_means,
                self.parent_col: self.parents + self.parents,
            }
        )

        te = sktools.NestedTargetEncoder(
            cols=self.col,
            feature_mapping=dict(col_1=self.parent_col),
            m_prior=0,
        )

        te.fit(self.X, self.y)

        pd.testing.assert_frame_equal(te.transform(new_x), expected_output_df)

    def test_missing_na(self):
        """
        When new categories and unknown values are given, we expect the encoder
        to give the parent means. If we specify return_nan, we want it to
        return nan
        """

        # First two rows are new categories
        # Last two rows are missing values
        # Parents are e, f, e, f
        new_x = pd.DataFrame(
            {
                self.col: ["x", "y", np.nan, np.nan],
                self.parent_col: self.parents + self.parents,
            }
        )

        # In the transformer we specify unknown -> return nan
        # We expect to get:
        # - nan for the unknown
        # - parent means for the missing
        expected_output_df = pd.DataFrame(
            {
                self.col: [np.nan, np.nan] + self.parent_means,
                self.parent_col: self.parents + self.parents,
            }
        )

        te = sktools.NestedTargetEncoder(
            cols=self.col,
            feature_mapping=dict(col_1=self.parent_col),
            m_prior=0,
            handle_missing="value",
            handle_unknown="return_nan",
        )

        te.fit(self.X, self.y)

        pd.testing.assert_frame_equal(te.transform(new_x), expected_output_df)

    def test_all_missing(self):
        """
        If everything's missing or unknow , we expect by default to return
        global mean
        """
        new_x = pd.DataFrame(
            {
                self.col: ["x", np.nan, "x", np.nan],
                self.parent_col: ["z", "z", np.nan, np.nan],
            }
        )

        te = sktools.NestedTargetEncoder(
            cols=self.col,
            feature_mapping=dict(col_1=self.parent_col),
            m_prior=0,
        )

        te.fit(self.X, self.y)

        expected_output_df = pd.DataFrame(
            {
                self.col: self.y.mean(),
                self.parent_col: ["z", "z", np.nan, np.nan],
            }
        )

        pd.testing.assert_frame_equal(te.transform(new_x), expected_output_df)


class TestGroupQuantile(unittest.TestCase):
    """Tests for group quantile."""

    def setUp(self):
        """Create dataframe with different column types"""
        self.X = pd.DataFrame(
            {
                "x": [1, 2, 3, 2, 20, 0, 10],
                "group": ["a", "a", "b", "b", None, None, "c"],
            }
        )

        self.new_X = pd.DataFrame({"x": [100.0], "group": ["d"]})

        # TODO: only 1 class should give .5 -> smoothing?
        self.output = self.X.copy().assign(
            x_quantile_group=[0.0, 1, 1, 0, 1, 0, 0.]
        )
        self.new_output = self.new_X.copy().assign(x_quantile_group=[1.0])

    def test_basic_example(self):
        groupedquantile = sktools.GroupedQuantileTransformer(
            feature_mapping={"x": "group"}
        )
        groupedquantile.fit(self.X)

        transformation = groupedquantile.transform(self.X)
        pd.testing.assert_frame_equal(transformation, self.output)

    def test_unknown(self):
        groupedquantile = sktools.GroupedQuantileTransformer(
            feature_mapping={"x": "group"}
        )
        groupedquantile.fit(self.X)

        transformation = groupedquantile.transform(self.new_X)
        pd.testing.assert_frame_equal(transformation, self.new_output)


class TestGroupQuantileFeaturizer(unittest.TestCase):
    """Tests for group quantile."""

    def setUp(self):
        """Create dataframe with different column types"""
        self.X = pd.DataFrame(
            {
                "x": [1, 2, 3, 2, 20, 0, 10],
                "group": ["a", "a", "b", "b", None, None, "c"],
            }
        )

        self.new_X = pd.DataFrame(
            {"x": [1, 2, 3, 4, 5], "group": ["a", "b", "c", "d", None]}
        )

        self.output = self.X.copy().assign(
            p50_x_group=[1.5, 1.5, 2.5, 2.5, 2.0, 2.0, 10],
            diff_p50_x_group=[-0.5, 0.5, 0.5, -0.5, 18.0, -2, 0],
            relu_diff_p50_x_group=[0, 0.5, 0.5, 0, 18.0, 0.0, 0],
            ratio_p50_x_group=[2.0 / 3, 4.0 / 3, 1.2, 0.8, 10, 0.0, 1.0],
        )

        self.missing_output = self.X.copy().assign(
            p50_x_group=[1.5, 1.5, 2.5, 2.5, None, None, 10],
            diff_p50_x_group=[-0.5, 0.5, 0.5, -0.5, None, None, 0],
        )

        self.new_output = self.new_X.copy().assign(
            p50_x_group=[1.5, 2.5, 10, 2.0, 2.0]
        )

    def test_basic_featurizer(self):
        featurizer = sktools.PercentileGroupFeaturizer(
            feature_mapping={"x": "group"}
        )

        featurizer.fit(self.X)

        pd.testing.assert_frame_equal(featurizer.transform(self.X), self.output)

    def test_missing(self):
        featurizer = sktools.PercentileGroupFeaturizer(
            feature_mapping={"x": "group"}, handle_missing="return_nan"
        )
        featurizer.fit(self.X)

        pd.testing.assert_frame_equal(
            featurizer.transform(self.X).iloc[:, :4], self.missing_output
        )

    def test_new_input(self):
        featurizer = sktools.PercentileGroupFeaturizer(
            feature_mapping={"x": "group"}
        )
        featurizer.fit(self.X)

        pd.testing.assert_frame_equal(
            featurizer.transform(self.new_X).iloc[:, 0:3], self.new_output
        )


class TestMeanFeaturizer(unittest.TestCase):
    """Tests for mean quantile."""

    def setUp(self):
        """Create dataframe with different column types"""
        self.X = pd.DataFrame(
            {
                "x": [1, 2, 3, 2, 10, 0, 10],
                "group": ["a", "a", "b", "b", None, None, "c"],
            }
        )

        self.new_X = pd.DataFrame(
            {"x": [1, 2, 3, 4, 5], "group": ["a", "b", "c", "d", None]}
        )

        self.output = self.X.copy().assign(
            mean_x_group=[1.5, 1.5, 2.5, 2.5, 4.0, 4.0, 10],
            diff_mean_x_group=[-0.5, 0.5, 0.5, -0.5, 6, -4, 0],
            relu_diff_mean_x_group=[0, 0.5, 0.5, 0, 6, 0.0, 0],
            ratio_mean_x_group=[2.0 / 3, 4.0 / 3, 1.2, 0.8, 10.0 / 4, 0.0, 1.0],
        )

        self.new_output = self.new_X.copy().assign(
            mean_x_group=[1.5, 2.5, 10, 4.0, 4.0]
        )

        self.missing_output = self.X.copy().assign(
            mean_x_group=[1.5, 1.5, 2.5, 2.5, None, None, 10],
            diff_mean_x_group=[-0.5, 0.5, 0.5, -0.5, None, None, 0],
        )

    def test_basic_featurizer(self):
        featurizer = sktools.MeanGroupFeaturizer(feature_mapping={"x": "group"})

        featurizer.fit(self.X)

        pd.testing.assert_frame_equal(featurizer.transform(self.X), self.output)

    def test_missing(self):
        featurizer = sktools.MeanGroupFeaturizer(
            feature_mapping={"x": "group"}, handle_missing="return_nan"
        )
        featurizer.fit(self.X)

        pd.testing.assert_frame_equal(
            featurizer.transform(self.X).iloc[:, :4], self.missing_output
        )

    def test_new_input(self):
        featurizer = sktools.MeanGroupFeaturizer(feature_mapping={"x": "group"})
        featurizer.fit(self.X)

        pd.testing.assert_frame_equal(
            featurizer.transform(self.new_X).iloc[:, 0:3], self.new_output
        )
