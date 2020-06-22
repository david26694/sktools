#!/usr/bin/env python

"""Tests for model selection module."""

import unittest

import sktools
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression

class TestBootstrapFold(unittest.TestCase):
    """Tests for quantile rgresson."""

    def setUp(self):
        """Load boston data and create cv"""

        self.X = load_boston()["data"]
        self.y = load_boston()["target"]
        self.model = LinearRegression()
        self.n_bootstraps = 10
        self.size_fraction = 0.7
        self.cv = sktools.BootstrapFold(
            n_bootstraps=self.n_bootstraps,
            size_fraction=self.size_fraction
        )

    def test_cross_val_integration(self):
        """Check cv is compatible with cross_val_score"""

        scores = cross_val_score(self.model, self.X, self.y, cv=self.cv)

        self.assertEqual(len(scores), 10)

    def test_grid_integration(self):
        """Check that cv is compatible with GridSearchCV"""

        param_grid = {'fit_intercept': [True, False]}
        grid = GridSearchCV(self.model, param_grid=param_grid, cv=self.cv)

        grid.fit(self.X, self.y)

        self.assertTrue('split9_test_score' in grid.cv_results_.keys())

    def test_n_splits(self):
        """Check that get_n_splits returns the number of bootstraps"""

        self.assertEqual(self.n_bootstraps, self.cv.get_n_splits())

    def test_size_fraction_works(self):
        """Check that size_fraction works as expected"""

        X = pd.DataFrame(dict(x=range(10000)))

        for train_idx, test_idx in self.cv.split(X):
            self.assertEqual(len(train_idx), self.size_fraction * X.shape[0])
