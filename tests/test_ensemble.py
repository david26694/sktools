#!/usr/bin/env python

"""Tests for ensemble module."""

import unittest

import sktools
from sklearn.datasets import load_boston
import numpy as np


class TestMedianForest(unittest.TestCase):
    """Tests for median forest."""

    def setUp(self):
        """Load boston data"""

        self.boston = load_boston()["data"]
        self.y = load_boston()["target"]

    def rf_mf_predictions(self, n_estimators):

        mf = sktools.MedianForestRegressor(n_estimators=n_estimators)

        mf.fit(self.boston, self.y)

        median_predictions = mf.predict(self.boston)
        # These are plain random forest predictions
        mean_predictions = mf.rf.predict(self.boston)

        return mean_predictions, median_predictions

    def test_1_tree(self):
        """
        For 1 tree, mean and median should be the same
        """

        median_predictions, mean_predictions = self.rf_mf_predictions(1)

        np.testing.assert_allclose(median_predictions, mean_predictions)

    def test_2_trees(self):
        """
        For 2 trees, mean and median should be the same
        """

        median_predictions, mean_predictions = self.rf_mf_predictions(2)

        np.testing.assert_allclose(median_predictions, mean_predictions)

    def test_many_trees(self):
        """
        For many trees, mean and median should not be the same
        """

        median_predictions, mean_predictions = self.rf_mf_predictions(100)

        self.assertNotEqual(median_predictions[0], mean_predictions[0])
