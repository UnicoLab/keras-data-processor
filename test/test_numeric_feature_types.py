"""Tests pinning what each numeric feature type actually does.

The documentation described `FLOAT_NORMALIZED` as "0-1 normalization" and
`FLOAT_RESCALED` as "robust scaling". Neither was true: the first standardises
to zero mean and unit variance, and the second multiplies by a `scale` that
defaults to 1.0, so it is a no-op unless the caller sets it.
"""

import tempfile
import unittest
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kdp import FeatureType, PreprocessingModel
from kdp.features import NumericalFeature

MEAN, SD, ROWS = 50.0, 10.0, 500


def _dataset(directory):
    """A column with a known mean and spread."""
    rng = np.random.default_rng(0)
    values = rng.normal(MEAN, SD, ROWS)
    csv_path = directory / "numbers.csv"
    pd.DataFrame({"x": values}).to_csv(csv_path, index=False)
    return csv_path, values


def _apply(tmp_path, spec, probe):
    keras.backend.clear_session()
    csv_path, values = _dataset(tmp_path)
    preprocessor = PreprocessingModel(
        path_data=str(csv_path),
        features_specs={"x": spec},
        features_stats_path=str(tmp_path / "stats.json"),
        overwrite_stats=True,
    )
    preprocessor.build_preprocessor()
    output = preprocessor.model({"x": tf.constant([[probe]], dtype=tf.float32)})
    return np.asarray(output), values


@pytest.mark.unit
class TestNumericFeatureTypes(unittest.TestCase):
    """One test per documented claim."""

    def test_normalized_is_a_z_score_not_zero_to_one(self):
        """The column mean maps to ~0 and one SD above it maps to ~1.

        A 0-1 normalisation would instead put the minimum at 0 and the maximum
        at 1, and never produce a negative number.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _, values = _dataset(tmp_path)

            at_mean, _ = _apply(
                tmp_path, FeatureType.FLOAT_NORMALIZED, float(values.mean())
            )
            one_sd_up, _ = _apply(
                tmp_path,
                FeatureType.FLOAT_NORMALIZED,
                float(values.mean() + values.std()),
            )
            below, _ = _apply(
                tmp_path, FeatureType.FLOAT_NORMALIZED, float(values.min())
            )

        self.assertAlmostEqual(float(at_mean[0][0]), 0.0, places=3)
        self.assertAlmostEqual(float(one_sd_up[0][0]), 1.0, places=1)
        # Below the mean the output is negative, which 0-1 scaling never is.
        self.assertLess(float(below[0][0]), 0.0)

    def test_float_is_an_alias_for_normalized(self):
        """Both produce the same number for the same input."""
        probe = 73.0
        with tempfile.TemporaryDirectory() as tmp:
            plain, _ = _apply(Path(tmp), FeatureType.FLOAT, probe)
        with tempfile.TemporaryDirectory() as tmp:
            normalized, _ = _apply(Path(tmp), FeatureType.FLOAT_NORMALIZED, probe)
        np.testing.assert_allclose(plain, normalized, rtol=1e-5)

    def test_rescaled_without_scale_is_the_identity(self):
        """This is the defect the docs hid: nothing happens by default."""
        with tempfile.TemporaryDirectory() as tmp:
            output, _ = _apply(Path(tmp), FeatureType.FLOAT_RESCALED, 100.0)
        self.assertAlmostEqual(float(output[0][0]), 100.0, places=4)

    def test_rescaled_honours_an_explicit_scale(self):
        """Setting `scale` is what makes the type useful."""
        with tempfile.TemporaryDirectory() as tmp:
            output, _ = _apply(
                Path(tmp),
                NumericalFeature(
                    name="x", feature_type=FeatureType.FLOAT_RESCALED, scale=0.01
                ),
                100.0,
            )
        self.assertAlmostEqual(float(output[0][0]), 1.0, places=4)

    def test_discretized_is_one_hot_over_num_bins(self):
        """Width follows `num_bins`, and exactly one bin fires."""
        with tempfile.TemporaryDirectory() as tmp:
            output, _ = _apply(Path(tmp), FeatureType.FLOAT_DISCRETIZED, 50.0)
        self.assertEqual(output.shape[-1], 10)
        self.assertAlmostEqual(float(output.sum()), 1.0, places=4)

        with tempfile.TemporaryDirectory() as tmp:
            five, _ = _apply(
                Path(tmp),
                NumericalFeature(
                    name="x", feature_type=FeatureType.FLOAT_DISCRETIZED, num_bins=5
                ),
                50.0,
            )
        self.assertEqual(five.shape[-1], 5)


if __name__ == "__main__":
    unittest.main()
