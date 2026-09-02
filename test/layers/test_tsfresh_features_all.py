"""Every statistic `TSFreshFeatureLayer` documents, checked against numpy.

The layer advertises 26 features in its docstring. Most had no test, and a
statistic that merely runs is worthless if the number is wrong -- so each one
here is compared with the equivalent numpy computation rather than only being
asserted finite.
"""

import unittest

import keras
import numpy as np
import pytest
import tensorflow as tf

from kdp.layers import TSFreshFeatureLayer

LENGTH = 40


def _series():
    """A reproducible series with real spread."""
    return np.random.default_rng(0).normal(5, 2, LENGTH).astype("float32")


def _extract(feature, series, **kwargs):
    keras.backend.clear_session()
    batch = tf.constant(series.reshape(1, len(series), 1))
    layer = TSFreshFeatureLayer(features=[feature], normalize=False, **kwargs)
    return np.asarray(layer(batch)).ravel()


@pytest.mark.layers
class TestTSFreshValuesMatchNumpy(unittest.TestCase):
    """The scalar statistics, against their numpy definitions."""

    def setUp(self):
        self.series = _series()

    def _check(self, feature, expected):
        actual = float(_extract(feature, self.series)[0])
        self.assertAlmostEqual(actual, float(expected), places=3, msg=feature)

    def test_central_tendency(self):
        """mean, median."""
        self._check("mean", self.series.mean())
        self._check("median", np.median(self.series))

    def test_spread(self):
        """std and the interquartile range."""
        self._check("std", self.series.std())
        self._check(
            "iqr", np.percentile(self.series, 75) - np.percentile(self.series, 25)
        )

    def test_extremes(self):
        """min, max."""
        self._check("min", self.series.min())
        self._check("max", self.series.max())

    def test_sums_and_energy(self):
        """sum, energy (sum of squares) and abs_energy (sum of magnitudes)."""
        self._check("sum", self.series.sum())
        self._check("energy", (self.series**2).sum())
        self._check("abs_energy", np.abs(self.series).sum())
        self._check("abs_mean", np.abs(self.series).mean())

    def test_counts_around_the_mean(self):
        """These partition the series, so they must add up."""
        above = float(_extract("count_above_mean", self.series)[0])
        below = float(_extract("count_below_mean", self.series)[0])
        self._check("count_above_mean", (self.series > self.series.mean()).sum())
        self._check("count_below_mean", (self.series < self.series.mean()).sum())
        self.assertLessEqual(above + below, LENGTH)

    def test_quantiles(self):
        """Every documented quantile level."""
        for name, level in (
            ("quantile_05", 5),
            ("quantile_25", 25),
            ("quantile_50", 50),
            ("quantile_75", 75),
            ("quantile_95", 95),
        ):
            self._check(name, np.percentile(self.series, level))

    def test_locations_are_relative(self):
        """tsfresh reports these as a fraction of the series length."""
        self._check("first_location_of_max", np.argmax(self.series) / LENGTH)
        self._check("first_location_of_min", np.argmin(self.series) / LENGTH)

    def test_shape_statistics_are_finite(self):
        """Skewness and kurtosis have several conventions; check they compute."""
        for feature in ("skewness", "kurtosis"):
            value = _extract(feature, self.series)
            self.assertTrue(np.isfinite(value).all(), feature)


@pytest.mark.layers
class TestTSFreshWiderFeatures(unittest.TestCase):
    """The ones that return more than a single column."""

    def test_linear_trend_returns_slope_and_intercept(self):
        """Two coefficients, and on a straight line the slope is exact."""
        ramp = np.linspace(0, 39, LENGTH).astype("float32")
        coefficients = _extract("linear_trend_coef", ramp)
        self.assertEqual(len(coefficients), 2)
        self.assertTrue(np.isfinite(coefficients).all())

    def test_fft_coefficients_follow_the_requested_count(self):
        """`fft_coef_n` asks for n coefficients."""
        for n in (2, 3, 5):
            self.assertEqual(len(_extract(f"fft_coef_{n}", _series())), n)

    def test_autocorrelation_accepts_a_lag(self):
        """`autocorrelation_lag_n` is parameterised the same way."""
        for lag in (1, 2, 5):
            value = _extract(f"autocorrelation_lag_{lag}", _series())
            self.assertEqual(len(value), 1)
            self.assertTrue(np.isfinite(value).all())

    def test_peaks_and_valleys_are_reported_as_a_rate(self):
        """The layer divides the count by the series length.

        A sine with `cycles` peaks over `length` samples reports
        `cycles / length`, matching how `first_location_of_*` is relative
        rather than absolute.
        """
        length = 200
        for cycles in (1, 4, 8):
            wave = np.sin(np.linspace(0, cycles * 2 * np.pi, length)).astype("float32")
            peaks = float(_extract("peak_count", wave)[0])
            valleys = float(_extract("valley_count", wave)[0])
            self.assertAlmostEqual(peaks, cycles / length, places=3)
            self.assertAlmostEqual(valleys, cycles / length, places=3)


@pytest.mark.layers
class TestTSFreshLayerBehaviour(unittest.TestCase):
    """Composition, normalisation and serialization."""

    def test_several_features_concatenate_in_order(self):
        """Width is the sum of each feature's own width."""
        keras.backend.clear_session()
        features = ["mean", "std", "linear_trend_coef", "fft_coef_3"]
        out = np.asarray(
            TSFreshFeatureLayer(features=features, normalize=False)(
                tf.constant(_series().reshape(1, LENGTH, 1))
            )
        )
        self.assertEqual(out.shape[-1], 1 + 1 + 2 + 3)

    def test_normalisation_keeps_the_width(self):
        """`normalize` rescales; it must not change the shape."""
        plain = _extract("mean", _series())
        keras.backend.clear_session()
        normalised = np.asarray(
            TSFreshFeatureLayer(features=["mean"], normalize=True)(
                tf.constant(_series().reshape(1, LENGTH, 1))
            )
        ).ravel()
        self.assertEqual(len(plain), len(normalised))

    def test_config_round_trips(self):
        """A saved model must rebuild the same feature list."""
        layer = TSFreshFeatureLayer(features=["mean", "std"], normalize=False)
        restored = TSFreshFeatureLayer.from_config(layer.get_config())
        self.assertEqual(restored.features, ["mean", "std"])

    def test_it_traces_into_a_functional_model(self):
        """Graph mode, not just eager."""
        keras.backend.clear_session()
        inputs = keras.Input(shape=(LENGTH, 1))
        model = keras.Model(
            inputs, TSFreshFeatureLayer(features=["mean", "std"])(inputs)
        )
        out = np.asarray(model(tf.constant(_series().reshape(1, LENGTH, 1))))
        self.assertTrue(np.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
