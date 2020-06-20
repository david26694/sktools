#!/usr/bin/env python

"""Tests for `sktools` package."""

import unittest

import sktools
import pandas as pd
from scipy.sparse import csr_matrix
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
                "other_char_col": ["d", "e", "f"]
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
            self.assertEqual(
                type_col,
                dtype
            )

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
                "other_char_col": ["d", "e", "f"]
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

            self.assertEqual(
                output_column,
                col
            )

            self.assertTrue(
                (output_transformer == self.df[col]).all(),
                "Not all values of the series are equal"
            )


class TestMatrixDenser(unittest.TestCase):
    """Tests for item selector."""

    def setUp(self):
        """Create dataframe with different column types"""
        self.sparse_matrix = csr_matrix((3, 4), dtype=np.int8)

    def test_zero_matrix(self):
        dense_matrix = sktools.MatrixDenser().fit_transform(self.sparse_matrix)

        expected_dense = np.array(
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]], dtype=np.int8
        )

        self.assertTrue(
            (dense_matrix == expected_dense).all(),
            "Not all values are 0"
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
                "other_char_col": ["d", "e", "f"]
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
                "other_char_col_na": [False, False, False]
            }
        )

    def test_defaults(self):
        pd.testing.assert_frame_equal(
            sktools.IsEmptyExtractor().fit_transform(self.df),
            self.expected_output.drop("other_char_col_na", axis=1)
        )

    def test_non_delete(self):
        pd.testing.assert_frame_equal(
            sktools.IsEmptyExtractor(keep_trivial=True).fit_transform(self.df),
            self.expected_output
        )

class TestQuantileEncoder(unittest.TestCase):
    """Tests for quantile encoder."""

    def setUp(self):
        """Create dataframe with categories and a target variable"""

        self.df = pd.DataFrame(
            {
                "categories": ["a", "b", "c", "a", "b", "c", "a", "b"]
            }
        )
        self.target = np.array([1, 2, 0, 4, 5, 0, 6, 7])

    def test_median_works(self):
        """
        Expected output of quantile 0.5 in df:
            - a median is 4 (a values are 1, 4, 6)
            - b median is 5 (b values are 2, 5, 7)
            - c median is 0 (c values are 0)
        """

        expected_output_median = pd.DataFrame(
            {
                "categories": [4., 5, 0, 4, 5, 0, 4, 5]
            }
        )

        pd.testing.assert_frame_equal(
            sktools.QuantileEncoder(quantile=0.5).fit_transform(
                self.df, self.target
            ),
            expected_output_median
        )

    def test_max_works(self):
        """
        Expected output of quantile 1 in df:
            - a max is 6
            - b max is 7
            - c max is 0
        """
        expected_output_max = pd.DataFrame(
            {
                "categories": [6., 7, 0, 6, 7, 0, 6, 7]
            }
        )

        pd.testing.assert_frame_equal(
            sktools.QuantileEncoder(quantile=1).fit_transform(
                self.df, self.target
            ),
            expected_output_max
        )

    def test_new_category(self):
        """The global median of the target is 3. If new categories are passed to
        the transformer, then the output should be 3
        """
        transformer_median = sktools.QuantileEncoder(quantile=0.5)
        transformer_median.fit(
            self.df, self.target
        )

        new_df = pd.DataFrame(
            {
                "categories": ["d", "e"]
            }
        )

        new_medians = pd.DataFrame(
            {
                "categories": [3., 3.]
            }
        )

        pd.testing.assert_frame_equal(
            transformer_median.transform(new_df),
            new_medians
        )
