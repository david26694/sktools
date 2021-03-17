#!/usr/bin/env python

"""Tests for linear model module."""

import unittest

import sktools
import numpy as np
import pandas as pd


class TestCyclicFeaturizer(unittest.TestCase):
    """Tests for cyclic featurizer."""

    def setUp(self):
        self.df = pd.DataFrame(
            dict(
                posted_at=pd.date_range(start="1/1/2018", periods=365 * 3, freq="d"),
                created_at=pd.date_range(start="1/1/2018", periods=365 * 3, freq="h"),
            )
        )

        self.df["month_posted"] = self.df.posted_at.dt.month
        self.df["hour_created"] = self.df.created_at.dt.hour

    def test_period_mapping(self):
        """Expect same output by specifying period mapping"""

        automatic_df = sktools.CyclicFeaturizer(
            cols=["month_posted", "hour_created"]
        ).fit_transform(self.df)

        mapped_df = sktools.CyclicFeaturizer(
            cols=["month_posted", "hour_created"],
            period_mapping=dict(month_posted=(1, 12), hour_created=(0, 23)),
        ).fit_transform(self.df)

        np.testing.assert_allclose(
            automatic_df.iloc[:, 2:].values, mapped_df.iloc[:, 2:].values
        )

    def test_trigonometry(self):
        """Expect cosines and sines to work"""

        # Apply transformation
        transformed_df = sktools.CyclicFeaturizer(
            cols=["month_posted", "hour_created"]
        ).fit_transform(self.df)

        # Cosine formula for hour - no shift
        hour_cos = transformed_df["hour_created_cos"]
        period_factor = 2 * np.pi / 24
        hour_cos_expected = np.cos(period_factor * transformed_df["hour_created"])

        np.testing.assert_allclose(hour_cos, hour_cos_expected)

        # Sine formula for month - some shift
        month_sin = transformed_df["month_posted_sin"]
        period_factor = 2 * np.pi / 12
        month_sin_expected = np.sin(
            period_factor * (transformed_df["month_posted"] - 1)
        )

        np.testing.assert_allclose(month_sin, month_sin_expected)


class TestGBFeatures(unittest.TestCase):
    """Tests for Gradient Boosting Feature Generator."""

    def setUp(self):
        """Load boston data"""

        self.boston = load_boston()["data"]
        self.y = load_boston()["target"]
        self.y = np.where(self.y > self.y.mean(), 1, 0)

    def test_1_tree(self):
        """
        For 1 tree, with max_dept = 1, only 4 features should be created
        [0001],[0010],[0100],[1000]
        """

        mf = sktools.GradientBoostingFeatureGenerator(
            sparse_feat=False, max_depth=1, n_estimators=1
        )

        mf.fit(self.boston, self.y)

        original_shape = self.boston.shape[1]
        transformed_shape = mf.transform(self.boston).shape[1]
        dif = transformed_shape - original_shape

        np.testing.assert_equal(dif, 4)

    def test_2_tree(self):
        """
        For 2 tree, with max_dept = 1, only 6 features should be created
        [000001],[000010],[000100],[001000],[010000],[100000]
        """

        mf = sktools.GradientBoostingFeatureGenerator(
            sparse_feat=False, max_depth=1, n_estimators=1
        )

        mf.fit(self.boston, self.y)

        original_shape = self.boston.shape[1]
        transformed_shape = mf.transform(self.boston).shape[1]
        dif = transformed_shape - original_shape

        np.testing.assert_equal(dif, 6)

    def test_leaves_shape(self):
        """
        For different estimators size check if the leaves scale together
        """
        for n in [1, 3, 4]:
            mf = sktools.GradientBoostingFeatureGenerator(
                sparse_feat=False, max_depth=1, n_estimators=n
            )
            mf.fit(self.boston, self.y)
            np.testing(mf._get_leaves(self.boston), n)
