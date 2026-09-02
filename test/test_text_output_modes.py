"""Tests for the text output modes TextVectorization supports.

`output_sequence_length` is only meaningful for the "int" output mode. KDP used
to default it unconditionally, which made every other mode unreachable: Keras
rejects the combination outright.
"""

import tempfile
import unittest
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kdp import PreprocessingModel
from kdp.features import FeatureType, TextFeature


def _dataset(directory, rows: int = 150):
    """Write a small text column with a stable vocabulary."""
    rng = np.random.default_rng(3)
    csv_path = directory / "text.csv"
    pd.DataFrame(
        {
            "bio": rng.choice(
                ["the hello world foo", "data science rocks the world"], rows
            )
        }
    ).to_csv(csv_path, index=False)
    return csv_path


def _build(tmp_path, **kwargs):
    keras.backend.clear_session()
    preprocessor = PreprocessingModel(
        path_data=str(_dataset(tmp_path)),
        features_specs={
            "bio": TextFeature(name="bio", feature_type=FeatureType.TEXT, **kwargs),
        },
        features_stats_path=str(tmp_path / "stats.json"),
        overwrite_stats=True,
    )
    preprocessor.build_preprocessor()
    return preprocessor


@pytest.mark.unit
class TestTextOutputModes(unittest.TestCase):
    """Every mode TextVectorization can serve from a fixed vocabulary."""

    def test_int_mode_is_padded_to_the_default_length(self):
        """The default stays a 35-token integer sequence."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(Path(tmp))
            output = preprocessor.model({"bio": tf.constant([["the hello world foo"]])})
        self.assertEqual(int(output.shape[-1]), 35)

    def test_explicit_sequence_length_is_respected(self):
        """Passing it overrides the default rather than being ignored."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(Path(tmp), output_sequence_length=10)
            output = preprocessor.model({"bio": tf.constant([["the hello world foo"]])})
        self.assertEqual(int(output.shape[-1]), 10)

    def test_multi_hot_mode_builds(self):
        """This used to raise because the default length was injected anyway."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(Path(tmp), output_mode="multi_hot")
            output = np.asarray(
                preprocessor.model({"bio": tf.constant([["the hello world foo"]])})
            )
        # One column per vocabulary entry, and it is an indicator vector.
        self.assertNotEqual(output.shape[-1], 35)
        self.assertTrue(set(np.unique(output)).issubset({0.0, 1.0}))

    def test_count_mode_builds(self):
        """Same defect, same fix."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(Path(tmp), output_mode="count")
            output = np.asarray(
                preprocessor.model({"bio": tf.constant([["the hello world foo"]])})
            )
        self.assertNotEqual(output.shape[-1], 35)
        self.assertGreater(output.sum(), 0)

    def test_stop_words_are_removed(self):
        """A stop word must not survive into the encoded sequence."""
        with tempfile.TemporaryDirectory() as tmp:
            plain = _build(Path(tmp))
            plain_out = np.asarray(
                plain.model({"bio": tf.constant([["the hello world foo"]])})
            )
        with tempfile.TemporaryDirectory() as tmp:
            filtered = _build(Path(tmp), stop_words=["the"])
            filtered_out = np.asarray(
                filtered.model({"bio": tf.constant([["the hello world foo"]])})
            )
        self.assertFalse(np.allclose(plain_out, filtered_out))


if __name__ == "__main__":
    unittest.main()
