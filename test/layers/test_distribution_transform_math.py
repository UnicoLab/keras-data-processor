"""Numerical correctness tests for DistributionTransformLayer.

The scale-oriented transforms (robust-scale, min-max, quantile) must agree with
their textbook definitions and must operate per feature, not across the whole
tensor.
"""

import unittest

import numpy as np
import tensorflow as tf

from kdp.layers.distribution_transform_layer import DistributionTransformLayer

ALL_TRANSFORMS = [
    "none",
    "log",
    "sqrt",
    "box-cox",
    "yeo-johnson",
    "arcsinh",
    "cube-root",
    "logit",
    "min-max",
    "robust-scale",
    "quantile",
]


def _multi_scale_data(seed: int = 0) -> np.ndarray:
    """Three columns with deliberately different centres and spreads."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.normal(loc=50, scale=10, size=200),
            rng.normal(loc=-3, scale=0.5, size=200),
            rng.normal(loc=1000, scale=250, size=200),
        ],
        axis=1,
    ).astype("float32")


class TestRobustScale(unittest.TestCase):
    """robust-scale centres on the median and divides by the IQR."""

    def test_matches_median_and_iqr_definition(self):
        """Output equals (x - median) / IQR computed per column."""
        data = _multi_scale_data()
        actual = np.asarray(
            DistributionTransformLayer(transform_type="robust-scale")(tf.constant(data))
        )

        median = np.median(data, axis=0)
        iqr = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)
        np.testing.assert_allclose(actual, (data - median) / iqr, rtol=1e-4, atol=1e-4)

    def test_output_is_centred_and_unit_iqr_per_feature(self):
        """Each transformed column has median ~0 and IQR ~1."""
        data = _multi_scale_data(seed=7)
        actual = np.asarray(
            DistributionTransformLayer(transform_type="robust-scale")(tf.constant(data))
        )

        np.testing.assert_allclose(np.median(actual, axis=0), 0.0, atol=1e-4)
        observed_iqr = np.percentile(actual, 75, axis=0) - np.percentile(
            actual, 25, axis=0
        )
        np.testing.assert_allclose(observed_iqr, 1.0, rtol=1e-4)

    def test_constant_feature_does_not_divide_by_zero(self):
        """A column with zero IQR yields finite output."""
        data = np.stack([np.full(50, 3.0), np.linspace(0, 1, 50)], axis=1).astype(
            "float32"
        )
        actual = np.asarray(
            DistributionTransformLayer(transform_type="robust-scale")(tf.constant(data))
        )
        self.assertTrue(np.all(np.isfinite(actual)))
        np.testing.assert_allclose(actual[:, 0], 0.0, atol=1e-6)


class TestQuantileTransform(unittest.TestCase):
    """quantile ranks values within each feature."""

    def test_each_feature_is_ranked_independently(self):
        """Columns on very different scales both span the full output range."""
        data = np.stack(
            [np.linspace(0, 1, 40), np.linspace(1000, 2000, 40)], axis=1
        ).astype("float32")
        np.random.default_rng(0).shuffle(data)

        actual = np.asarray(
            DistributionTransformLayer(transform_type="quantile")(tf.constant(data))
        )

        for column in range(data.shape[1]):
            self.assertLess(actual[:, column].min(), -0.9)
            self.assertGreater(actual[:, column].max(), 0.9)

    def test_preserves_rank_order_within_each_feature(self):
        """The transform is monotonic per column."""
        data = _multi_scale_data(seed=3)
        actual = np.asarray(
            DistributionTransformLayer(transform_type="quantile")(tf.constant(data))
        )
        for column in range(data.shape[1]):
            np.testing.assert_array_equal(
                np.argsort(data[:, column]), np.argsort(actual[:, column])
            )

    def test_output_is_bounded(self):
        """Values stay inside [-1, 1]."""
        actual = np.asarray(
            DistributionTransformLayer(transform_type="quantile")(
                tf.constant(_multi_scale_data(seed=11))
            )
        )
        self.assertGreaterEqual(actual.min(), -1.0)
        self.assertLessEqual(actual.max(), 1.0)


class TestMinMax(unittest.TestCase):
    """min-max maps data into [0, 1]."""

    def test_scales_into_unit_interval(self):
        """With clipping disabled, the observed range maps onto [0, 1]."""
        data = _multi_scale_data(seed=5)
        actual = np.asarray(
            DistributionTransformLayer(transform_type="min-max", clip_values=False)(
                tf.constant(data)
            )
        )
        self.assertAlmostEqual(float(actual.min()), 0.0, places=5)
        self.assertAlmostEqual(float(actual.max()), 1.0, places=5)


class TestGraphModeCompatibility(unittest.TestCase):
    """Every transform must build inside a functional model."""

    def test_all_transforms_work_with_a_dynamic_batch_dimension(self):
        """Transforms run when the batch size is only known at call time."""
        data = np.abs(_multi_scale_data(seed=2)[:16]) / 5000.0
        for transform_type in ALL_TRANSFORMS:
            with self.subTest(transform_type=transform_type):
                tf.keras.backend.clear_session()
                inputs = tf.keras.Input(shape=(3,))
                model = tf.keras.Model(
                    inputs,
                    DistributionTransformLayer(transform_type=transform_type)(inputs),
                )
                output = model(tf.constant(data, dtype=tf.float32))
                self.assertEqual(tuple(output.shape), (16, 3))


class TestStatisticsHelpers(unittest.TestCase):
    """The order-statistics helper backing robust-scale."""

    def test_percentile_matches_numpy(self):
        """_percentile reproduces numpy's linear interpolation."""
        layer = DistributionTransformLayer(transform_type="robust-scale")
        data = _multi_scale_data(seed=13)
        sorted_data = tf.sort(tf.constant(data), axis=0)

        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            with self.subTest(q=q):
                np.testing.assert_allclose(
                    np.asarray(layer._percentile(sorted_data, q)),
                    np.percentile(data, q * 100, axis=0),
                    rtol=1e-4,
                    atol=1e-4,
                )

    def test_compute_robust_statistics_returns_per_feature_values(self):
        """min/max/median/IQR are computed along the batch axis."""
        layer = DistributionTransformLayer(transform_type="robust-scale")
        data = _multi_scale_data(seed=17)
        x_min, x_max, median, iqr = layer._compute_robust_statistics(tf.constant(data))

        np.testing.assert_allclose(
            np.asarray(x_min).ravel(), data.min(axis=0), rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(x_max).ravel(), data.max(axis=0), rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(median), np.median(data, axis=0), rtol=1e-4
        )
        expected_iqr = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)
        np.testing.assert_allclose(np.asarray(iqr), expected_iqr, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
