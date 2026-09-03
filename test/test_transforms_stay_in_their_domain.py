"""A transform outside its domain must return a number, not NaN or infinity.

`log` is defined for positive values. Every neighbouring transform in the layer
guards its own domain -- `sqrt` clamps at zero, `box-cox` at epsilon -- but
`log` handed negatives straight to `log()` and returned NaN. A NaN in a
preprocessor output is not a local problem: it spreads through every layer that
touches it and through every gradient taken from it, without a message.

`logit` clipped into `[epsilon, 1 - epsilon]`, which reads as correct and was
not: the default epsilon of 1e-10 is far below the float32 gap at 1.0, so
`1 - epsilon` rounded back to exactly 1.0, `1 - x` became 0, and any value at
or above 1 came out as +inf.
"""

import unittest

import numpy as np
import pytest
import tensorflow as tf

from kdp.layers.distribution_transform_layer import DistributionTransformLayer

# Values that sit outside the domain of at least one transform.
OUT_OF_DOMAIN = np.array(
    [[-1e6], [-5.0], [-0.5], [0.0], [0.5], [1.0], [5.0], [1e6]],
    dtype=np.float32,
)

TRANSFORMS = [
    "log",
    "sqrt",
    "box-cox",
    "yeo-johnson",
    "arcsinh",
    "cube-root",
    "logit",
    "quantile",
    "robust-scale",
    "min-max",
]


@pytest.mark.layers
class TestTransformsStayInTheirDomain(unittest.TestCase):
    """No transform may answer with NaN or infinity."""

    def test_every_transform_returns_finite_numbers(self) -> None:
        offenders = {}
        for name in TRANSFORMS:
            result = np.asarray(
                DistributionTransformLayer(transform_type=name)(
                    tf.constant(OUT_OF_DOMAIN),
                ),
            )
            if not np.isfinite(result).all():
                offenders[name] = result.reshape(-1).tolist()
        self.assertEqual(offenders, {}, f"NaN or infinity in the output: {offenders}")

    def test_log_still_matches_its_definition_where_it_is_defined(self) -> None:
        values = np.linspace(0.05, 5.0, 40, dtype=np.float32).reshape(-1, 1)
        result = np.asarray(
            DistributionTransformLayer(transform_type="log")(tf.constant(values)),
        )
        np.testing.assert_allclose(result, np.log(values + 1e-10), atol=1e-4)

    def test_logit_still_matches_its_definition_inside_the_unit_interval(self) -> None:
        values = np.linspace(0.05, 0.95, 40, dtype=np.float32).reshape(-1, 1)
        result = np.asarray(
            DistributionTransformLayer(transform_type="logit")(tf.constant(values)),
        )
        np.testing.assert_allclose(result, np.log(values / (1 - values)), atol=5e-3)

    def test_logit_is_symmetric_at_both_ends(self) -> None:
        """The clip has to bite on both sides, not only the low one."""
        result = np.asarray(
            DistributionTransformLayer(transform_type="logit")(
                tf.constant(np.array([[0.0], [1.0]], dtype=np.float32)),
            ),
        ).reshape(-1)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result[0], -result[1], rtol=1e-5)

    def test_arcsinh_is_odd_and_matches_its_definition(self) -> None:
        """`log(x + sqrt(x*x + 1))` lost the whole negative tail to float32.

        At x = -1e6 the square is 1e12, where float32 cannot hold the +1; its
        root is exactly 1e6, the sum is exactly zero, and the logarithm is
        -infinity. The positive side was unaffected, so the transform was quietly
        one-sided.
        """
        values = np.array(
            [[-1e6], [-1e3], [-1.0], [0.0], [1.0], [1e3], [1e6]],
            dtype=np.float32,
        )
        result = np.asarray(
            DistributionTransformLayer(transform_type="arcsinh")(tf.constant(values)),
        ).reshape(-1)
        np.testing.assert_allclose(result, np.arcsinh(values.reshape(-1)), rtol=1e-5)
        np.testing.assert_allclose(result, -result[::-1], rtol=1e-5)

    def test_a_negative_column_survives_the_log_transform_in_a_model(self) -> None:
        """The same thing, through the layer as a model would call it."""
        column = np.linspace(-50.0, 50.0, 100, dtype=np.float32).reshape(-1, 1)
        result = np.asarray(
            DistributionTransformLayer(transform_type="log")(tf.constant(column)),
        )
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(
            len(np.unique(result)),
            10,
            "the positive half of the column was flattened too",
        )


if __name__ == "__main__":
    unittest.main()
