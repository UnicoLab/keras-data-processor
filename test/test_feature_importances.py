"""Tests for `get_feature_importances`.

The method is documented as showing which features matter most. It returned a
description of each weight tensor -- shape, dtype, layer name -- so the
documented `sorted(importances.items(), key=lambda x: x[1])` raised
`TypeError: '<' not supported between instances of 'dict' and 'dict'`.

The importances are a per-row softmax, so they only exist once data is run
through the model; that is what the optional `data` argument is for.
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

SPECS = {
    "age": FeatureType.FLOAT_NORMALIZED,
    "income": FeatureType.FLOAT_RESCALED,
    "education": FeatureType.STRING_CATEGORICAL,
}


def _dataset(directory, rows: int = 200):
    """Write a CSV covering the three features used here."""
    rng = np.random.default_rng(4)
    csv_path = directory / "customers.csv"
    pd.DataFrame(
        {
            "age": rng.normal(40, 10, rows),
            "income": rng.normal(70_000, 15_000, rows),
            "education": rng.choice(["bsc", "msc"], rows),
        }
    ).to_csv(csv_path, index=False)
    return csv_path


def _build(tmp_path, **kwargs):
    keras.backend.clear_session()
    preprocessor = PreprocessingModel(
        path_data=str(_dataset(tmp_path)),
        features_specs=dict(SPECS),
        features_stats_path=str(tmp_path / "stats.json"),
        overwrite_stats=True,
        **kwargs,
    )
    preprocessor.build_preprocessor()
    return preprocessor


BATCH = {
    "age": tf.constant([[35.0], [50.0]]),
    "income": tf.constant([[70_000.0], [90_000.0]]),
    "education": tf.constant([["bsc"], ["msc"]]),
}


@pytest.mark.unit
class TestFeatureImportances(unittest.TestCase):
    """What the method returns, with and without data."""

    def test_without_selection_it_is_empty(self):
        """Nothing to report when the layer was never added."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(Path(tmp))
            self.assertEqual(preprocessor.get_feature_importances(), {})

    def test_without_data_it_describes_the_tensors(self):
        """The original behaviour is preserved for callers that relied on it."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(Path(tmp), feature_selection_placement="all_features")
            described = preprocessor.get_feature_importances()
        self.assertTrue(described)
        for value in described.values():
            self.assertIn("shape", value)

    def test_with_data_it_returns_sortable_numbers(self):
        """The documented sort used to raise TypeError on dict values."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(Path(tmp), feature_selection_placement="all_features")
            importances = preprocessor.get_feature_importances(BATCH)

        self.assertTrue(importances)
        for value in importances.values():
            self.assertIsInstance(value, float)
        # This is the exact expression the documentation shows.
        ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:3]
        self.assertEqual(len(ranked), min(3, len(importances)))

    def test_importances_are_currently_uninformative(self):
        """Each feature gets its own selector over a single feature.

        A softmax across one element is 1.0 by definition, so every weight is
        1.0 regardless of the data. The selection layer still applies a learned
        gated residual transform, but the weights cannot rank anything. This
        test records that limitation so a real implementation has to update it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(Path(tmp), feature_selection_placement="all_features")
            importances = preprocessor.get_feature_importances(BATCH)
            other = preprocessor.get_feature_importances(
                {
                    "age": tf.constant([[99.0]]),
                    "income": tf.constant([[1.0]]),
                    "education": tf.constant([["msc"]]),
                }
            )

        self.assertTrue(all(v == pytest.approx(1.0) for v in importances.values()))
        # Wildly different input, identical weights: they carry no signal.
        self.assertEqual(sorted(importances.values()), sorted(other.values()))


if __name__ == "__main__":
    unittest.main()
