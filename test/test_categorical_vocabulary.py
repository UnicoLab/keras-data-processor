"""What happens to a categorical feature whose column has nothing in it.

An empty vocabulary used to be swapped for `["<UNK>"]`. That made a string
feature encode every real value to the out-of-vocabulary slot -- a constant
column, produced silently -- and made an integer feature crash inside Keras
with "invalid literal for int() with base 10", an error naming neither the
feature nor the cause. Both are now a single error that says which feature is
empty and what to do about it.

The check has to stay narrow: a column of empty strings has the vocabulary
`[""]` and a column with one repeated value has a vocabulary of length one.
Neither is empty, both are legitimate, and neither may be rejected.
"""

import tempfile
import unittest
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kdp import CategoryEncodingOptions, FeatureType, PreprocessingModel
from kdp.features import CategoricalFeature

ROWS = 120


def _build(directory, column, specs, name="data"):
    """Build a preprocessor over a single-column frame."""
    keras.backend.clear_session()
    csv_path = Path(directory) / f"{name}.csv"
    pd.DataFrame({"cat": column}).to_csv(csv_path, index=False)
    preprocessor = PreprocessingModel(
        path_data=str(csv_path),
        features_specs=specs,
        features_stats_path=str(Path(directory) / f"{name}.json"),
        overwrite_stats=True,
    )
    preprocessor.build_preprocessor()
    return preprocessor


@pytest.mark.unit
class TestAnEmptyColumnIsReported(unittest.TestCase):
    """No values at all is an error that names the feature."""

    def test_a_string_column_with_no_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            with pytest.raises(ValueError) as caught:
                _build(
                    directory,
                    pd.Series([], dtype=str),
                    {"cat": FeatureType.STRING_CATEGORICAL},
                )
        message = str(caught.value)
        assert "cat" in message
        assert "empty" in message.lower()

    def test_an_integer_column_with_no_rows(self):
        """This one used to fail inside Keras, naming neither cause nor feature."""
        with tempfile.TemporaryDirectory() as directory:
            with pytest.raises(ValueError) as caught:
                _build(
                    directory,
                    pd.Series([], dtype="int64"),
                    {"cat": FeatureType.INTEGER_CATEGORICAL},
                )
        message = str(caught.value)
        assert "cat" in message
        assert "invalid literal" not in message

    def test_the_message_names_the_way_out(self):
        """It recommends hashing, so hashing has to be the thing that works."""
        with tempfile.TemporaryDirectory() as directory:
            with pytest.raises(ValueError) as caught:
                _build(
                    directory,
                    pd.Series([], dtype=str),
                    {"cat": FeatureType.STRING_CATEGORICAL},
                )
            assert "HASHING" in str(caught.value)

            hashed = _build(
                directory,
                pd.Series([], dtype=str),
                {
                    "cat": CategoricalFeature(
                        name="cat",
                        feature_type=FeatureType.STRING_CATEGORICAL,
                        category_encoding=CategoryEncodingOptions.HASHING,
                        hash_bucket_size=16,
                    ),
                },
                name="hashed",
            )
            out = hashed.model(
                {"cat": tf.constant([["a"], ["b"]], dtype=tf.string)},
                training=False,
            )
        assert int(out.shape[0]) == 2


@pytest.mark.unit
class TestAPresentButUninformativeColumnStillBuilds(unittest.TestCase):
    """A one-entry vocabulary is not an empty one, and must not be rejected."""

    def test_a_column_of_empty_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            built = _build(
                directory,
                [""] * ROWS,
                {"cat": FeatureType.STRING_CATEGORICAL},
            )
            out = built.model(
                {"cat": tf.constant([[""], [""]], dtype=tf.string)},
                training=False,
            )
        assert int(out.shape[0]) == 2

    def test_a_constant_column(self):
        with tempfile.TemporaryDirectory() as directory:
            built = _build(
                directory,
                ["US"] * ROWS,
                {"cat": FeatureType.STRING_CATEGORICAL},
            )
            out = built.model(
                {"cat": tf.constant([["US"], ["US"]], dtype=tf.string)},
                training=False,
            )
        assert int(out.shape[0]) == 2

    def test_empty_strings_mixed_with_real_categories(self):
        """The empties join the vocabulary; the real categories keep their identity."""
        rng = np.random.default_rng(3)
        column = rng.choice(["", "alpha", "beta"], ROWS).tolist()
        column[:3] = ["", "alpha", "beta"]
        with tempfile.TemporaryDirectory() as directory:
            built = _build(
                directory,
                column,
                {"cat": FeatureType.STRING_CATEGORICAL},
            )
            out = built.model(
                {"cat": tf.constant([[""], ["alpha"], ["beta"]], dtype=tf.string)},
                training=False,
            ).numpy()
        # Three known categories, three distinct encodings. The old `["<UNK>"]`
        # fallback collapsed every value onto one row here.
        assert len({tuple(np.round(row, 6)) for row in out}) == 3
