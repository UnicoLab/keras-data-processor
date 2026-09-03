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

    def test_importances_rank_features_against_each_other(self):
        """Every feature used to score exactly 1.0, whatever the data.

        Selection wrapped each feature in its own `VariableSelection` with
        `nr_features=1`, and a softmax over one element is 1.0 by definition.
        The gating worked but the numbers ranked nothing, and the documentation
        had to say so. One softmax across all the selected features gives each
        one a share.
        """
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(Path(tmp), feature_selection_placement="all_features")
            importances = preprocessor.get_feature_importances(BATCH)

        self.assertEqual(sorted(importances), ["age", "education", "income"])
        self.assertAlmostEqual(sum(importances.values()), 1.0, places=4)
        # Shares, not a constant.
        self.assertGreater(len({round(v, 6) for v in importances.values()}), 1)

    def test_a_lone_feature_keeps_a_weight_of_one(self):
        """With nothing to compare against, all of the weight is its own."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rng = np.random.default_rng(0)
            csv_path = directory / "one.csv"
            pd.DataFrame({"age": rng.normal(40, 8, 200)}).to_csv(csv_path, index=False)
            keras.backend.clear_session()
            preprocessor = PreprocessingModel(
                path_data=str(csv_path),
                features_specs={"age": FeatureType.FLOAT_NORMALIZED},
                features_stats_path=str(directory / "stats.json"),
                overwrite_stats=True,
                feature_selection_placement="all_features",
            )
            preprocessor.build_preprocessor()
            importances = preprocessor.get_feature_importances(
                {"age": tf.constant([[40.0]])},
            )

        self.assertEqual(importances, {"age": pytest.approx(1.0)})

    def test_selection_does_not_change_the_output_width(self):
        """Scoring rescales the features; it must not reshape the model."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(
                Path(tmp),
                feature_selection_placement="all_features",
                feature_selection_units=8,
            )
            width = preprocessor.model.output_shape[-1]
        # Three features, each `feature_selection_units` wide.
        self.assertEqual(width, 3 * 8)


if __name__ == "__main__":
    unittest.main()
