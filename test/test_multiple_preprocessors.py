"""Tests for combining several preprocessors in one Keras graph.

Every built model was named "preprocessor". Keras requires operation names to
be unique within a graph, so a two-tower architecture -- a user branch and an
item branch, which the integration guide demonstrates -- raised
`ValueError: The name "preprocessor" is used 2 times in the model`.
"""

import tempfile
import unittest
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import pytest

from kdp import FeatureType, PreprocessingModel


def _csv(directory, column: str, rows: int = 120):
    """Each preprocessor needs its own directory: the reader globs *.csv."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(6)
    path = directory / "data.csv"
    pd.DataFrame({column: rng.normal(40, 10, rows)}).to_csv(path, index=False)
    return path


def _build(tmp_path, column: str, name: str | None = None):
    kwargs = {} if name is None else {"name": name}
    preprocessor = PreprocessingModel(
        path_data=str(_csv(tmp_path / column, column)),
        features_specs={column: FeatureType.FLOAT_NORMALIZED},
        features_stats_path=str(tmp_path / f"{column}.json"),
        overwrite_stats=True,
        **kwargs,
    )
    return preprocessor.build_preprocessor()["model"]


@pytest.mark.unit
class TestMultiplePreprocessorsInOneGraph(unittest.TestCase):
    """Two preprocessors, one model."""

    def test_default_name_is_preprocessor(self):
        """The name stays the same for anyone relying on it."""
        with tempfile.TemporaryDirectory() as tmp:
            keras.backend.clear_session()
            model = _build(Path(tmp), "age")
        self.assertEqual(model.name, "preprocessor")

    def test_two_towers_build_with_distinct_names(self):
        """This is the recommendation-system shape from the integration guide."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            keras.backend.clear_session()
            user_model = _build(tmp_path, "age", name="user_preprocessor")
            item_model = _build(tmp_path, "price", name="item_preprocessor")

            user_inputs = {"age": keras.Input(shape=(1,), name="age")}
            item_inputs = {"price": keras.Input(shape=(1,), name="price")}
            score = keras.layers.Dot(axes=1)(
                [user_model(user_inputs), item_model(item_inputs)]
            )
            combined = keras.Model(inputs={**user_inputs, **item_inputs}, outputs=score)

        self.assertEqual(combined.output_shape, (None, 1))
        self.assertNotEqual(user_model.name, item_model.name)

    def test_duplicate_names_still_collide(self):
        """Keras' own guarantee is unchanged; the fix is that a name can differ."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            keras.backend.clear_session()
            first = _build(tmp_path, "age", name="same")
            second = _build(tmp_path, "price", name="same")

            first_inputs = {"age": keras.Input(shape=(1,), name="age")}
            second_inputs = {"price": keras.Input(shape=(1,), name="price")}
            with self.assertRaises(ValueError):
                keras.Model(
                    inputs={**first_inputs, **second_inputs},
                    outputs=keras.layers.Dot(axes=1)(
                        [first(first_inputs), second(second_inputs)]
                    ),
                )


if __name__ == "__main__":
    unittest.main()
