#!/usr/bin/env python

"""Tests for linear model module."""

import unittest

import sktools
from sklearn.datasets import load_boston
import numpy as np


class TestQuantileRegression(unittest.TestCase):
    """Tests for quantile rgresson."""

    def setUp(self):
        """Load boston data"""

        self.boston = load_boston()["data"]
        self.y = load_boston()["target"]

    def test_without_intercept(self):
        """
        Check f(0) = 0 without intercept
        """

        qr = sktools.QuantileRegression(add_intercept=False)

        qr.fit(self.boston, self.y)

        matrix_0 = 0 * self.boston
        predictions = qr.predict(matrix_0)
        vector_0 = predictions * 0

        np.testing.assert_allclose(predictions, vector_0)

    def test_with_intercept(self):
        """
        Just check that it runs
        """
        qr = sktools.QuantileRegression(quantile=0.1)
        qr.fit(self.boston, self.y)
        predictions = qr.predict(self.boston)
