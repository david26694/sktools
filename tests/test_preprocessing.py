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
                posted_at=pd.date_range(start="1/1/2018", periods=365 * 3,
                                        freq="d"),
                created_at=pd.date_range(start="1/1/2018", periods=365 * 3,
                                         freq="h"),
            )
        )

        self.df['month_posted'] = self.df.posted_at.dt.month
        self.df['hour_created'] = self.df.created_at.dt.hour

    def test_period_mapping(self):
        """Expect same output by specifying period mapping"""

        automatic_df = sktools.CyclicFeaturizer(
            cols=['month_posted', 'hour_created']
        ).fit_transform(self.df)

        mapped_df = sktools.CyclicFeaturizer(
            cols=['month_posted', 'hour_created'],
            period_mapping=dict(
                month_posted=(1, 12),
                hour_created=(0, 23)
            )
        ).fit_transform(self.df)

        np.testing.assert_allclose(
            automatic_df.iloc[:, 2:].values,
            mapped_df.iloc[:, 2:].values
        )

    def test_trigonometry(self):
        """Expect cosines and sines to work"""

        # Apply transformation
        transformed_df = sktools.CyclicFeaturizer(
            cols=['month_posted', 'hour_created']
        ).fit_transform(self.df)

        # Cosine formula for hour - no shift
        hour_cos = transformed_df['hour_created_cos']
        period_factor = 2 * np.pi / 24
        hour_cos_expected = np.cos(
            period_factor * transformed_df['hour_created']
        )

        np.testing.assert_allclose(
            hour_cos,
            hour_cos_expected
        )

        # Sine formula for month - some shift
        month_sin = transformed_df['month_posted_sin']
        period_factor = 2 * np.pi / 12
        month_sin_expected = np.sin(
            period_factor * (transformed_df['month_posted'] - 1)
        )

        np.testing.assert_allclose(
            month_sin,
            month_sin_expected
        )
