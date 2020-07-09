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

        self.assertTrue((dense_matrix == expected_dense).all(), "Not all values are 0")


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
        self.output = self.X.copy().assign(x_quantile_group=[0.0, 1, 1, 0, 1, 0, 0.0])
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

        self.new_output = self.new_X.copy().assign(p50_x_group=[1.5, 2.5, 10, 2.0, 2.0])

    def test_basic_featurizer(self):
        featurizer = sktools.PercentileGroupFeaturizer(feature_mapping={"x": "group"})

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
        featurizer = sktools.PercentileGroupFeaturizer(feature_mapping={"x": "group"})
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
