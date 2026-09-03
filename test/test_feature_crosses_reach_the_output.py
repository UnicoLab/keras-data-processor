"""A cross has to appear in the output, and mean what a hashed cross means.

`feature_crosses` built every cross it was given and then dropped it:
`_group_features_by_type` looks each processed feature up in `features_specs`,
a cross is configured with `feature_crosses` rather than by declaring a column,
so the lookup found nothing and skipped it. In the default output mode the
option was inert -- the model built, ran, and had exactly the width it would
have had with no crosses at all.

The existing tests asserted that the model built and that its output was not
None, which was true throughout. These measure the cross itself: that it widens
the output, that its value depends on both columns and on nothing else, and
that it stays inside the number of bins it was given.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kdp import FeatureType, PreprocessingModel
from kdp.processor import OutputModeOptions

COLOURS = ["red", "green", "blue"]
SHAPES = ["circle", "square"]

SPECS = {
    "colour": FeatureType.STRING_CATEGORICAL,
    "shape": FeatureType.STRING_CATEGORICAL,
    "size": FeatureType.FLOAT_NORMALIZED,
}


def _frame(rows: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "colour": [COLOURS[i % len(COLOURS)] for i in range(rows)],
            "shape": [SHAPES[i % len(SHAPES)] for i in range(rows)],
            "size": np.linspace(0.0, 10.0, rows),
        },
    )


def _call(model: tf.keras.Model, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        model({name: tf.constant(frame[name].values.reshape(-1, 1)) for name in frame}),
    )


@pytest.mark.unit
class TestFeatureCrossesReachTheOutput(unittest.TestCase):
    """The cross has to survive as far as the tensor the caller receives."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.frame = _frame()
        self.data = self.directory / "data.csv"
        self.frame.to_csv(self.data, index=False)

    def _build(self, **kwargs) -> tf.keras.Model:
        model = PreprocessingModel(
            path_data=str(self.data),
            features_stats_path=str(self.directory / f"stats_{len(kwargs)}.json"),
            features_specs=SPECS,
            overwrite_stats=True,
            **kwargs,
        )
        return model.build_preprocessor()["model"]

    def test_a_cross_makes_the_output_wider(self) -> None:
        """One crossed column per cross, on top of the declared features."""
        plain = _call(self._build(), self.frame).shape[1]
        crossed = _call(
            self._build(feature_crosses=[("colour", "shape", 16)]),
            self.frame,
        ).shape[1]
        self.assertEqual(
            crossed,
            plain + 1,
            "the cross never reached the concatenated output",
        )

    def test_two_crosses_add_two_columns(self) -> None:
        plain = _call(self._build(), self.frame).shape[1]
        crossed = _call(
            self._build(
                feature_crosses=[("colour", "shape", 16), ("shape", "colour", 8)],
            ),
            self.frame,
        ).shape[1]
        self.assertEqual(crossed, plain + 2)

    def test_the_cross_is_a_function_of_both_columns_and_of_nothing_else(self) -> None:
        """Every distinct pair gets its own bin; other columns do not move it."""
        width = _call(self._build(), self.frame).shape[1]
        model = self._build(feature_crosses=[("colour", "shape", 64)])
        probe = pd.DataFrame(
            {
                "colour": ["red", "red", "green", "green", "red"],
                "shape": ["circle", "square", "circle", "square", "circle"],
                # The last row repeats the first pair with a different size.
                "size": [1.0, 1.0, 1.0, 1.0, 9.0],
            },
        )
        crossed = _call(model, probe)[:, width:]

        self.assertEqual(
            len({tuple(row) for row in crossed[:4]}),
            4,
            "distinct (colour, shape) pairs collided into one bin",
        )
        np.testing.assert_allclose(
            crossed[0],
            crossed[4],
            err_msg="the cross moved when a column it does not cross changed",
        )

    def test_the_cross_stays_inside_the_number_of_bins(self) -> None:
        rows = 120
        frame = pd.DataFrame(
            {
                "colour": [f"colour_{index}" for index in range(rows)],
                "shape": [f"shape_{index % 7}" for index in range(rows)],
                "size": np.linspace(0.0, 1.0, rows),
            },
        )
        data = self.directory / "wide.csv"
        frame.to_csv(data, index=False)

        def build(**kwargs) -> tf.keras.Model:
            model = PreprocessingModel(
                path_data=str(data),
                features_stats_path=str(self.directory / f"wide_{len(kwargs)}.json"),
                features_specs=SPECS,
                overwrite_stats=True,
                **kwargs,
            )
            return model.build_preprocessor()["model"]

        width = _call(build(), frame).shape[1]
        crossed = _call(build(feature_crosses=[("colour", "shape", 5)]), frame)[
            :,
            width:,
        ]
        self.assertGreaterEqual(crossed.min(), 0.0)
        self.assertLess(crossed.max(), 5.0)
        self.assertGreater(
            len(np.unique(crossed)),
            1,
            "every pair hashed to the same bin",
        )

    def test_the_cross_is_a_key_in_dict_mode(self) -> None:
        model = self._build(
            feature_crosses=[("colour", "shape", 16)],
            output_mode=OutputModeOptions.DICT,
        )
        outputs = model(
            {
                name: tf.constant(self.frame[name].values.reshape(-1, 1))
                for name in self.frame
            },
        )
        self.assertIn("colour_x_shape", outputs)


@pytest.mark.unit
class TestFeatureCrossesRefuseWhatTheyCannotDo(unittest.TestCase):
    """A cross that cannot run has to say so while the config is still in view."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.data = self.directory / "data.csv"
        _frame().to_csv(self.data, index=False)

    def _build(self, crosses: list) -> tf.keras.Model:
        model = PreprocessingModel(
            path_data=str(self.data),
            features_stats_path=str(self.directory / "stats.json"),
            features_specs=SPECS,
            overwrite_stats=True,
            feature_crosses=crosses,
        )
        return model.build_preprocessor()["model"]

    def test_crossing_a_float_column_is_refused_at_build_time(self) -> None:
        """`HashedCrossing` takes integers and strings; floats used to build.

        The model came back looking correct -- summary, output width, the lot --
        and raised on the first batch it was ever given.
        """
        with self.assertRaises(ValueError) as raised:
            self._build([("colour", "size", 8)])
        message = str(raised.exception)
        self.assertIn("size", message)
        self.assertIn("float32", message)

    def test_crossing_an_undeclared_feature_names_it(self) -> None:
        with self.assertRaises(ValueError) as raised:
            self._build([("colour", "not_a_column", 8)])
        self.assertIn("not_a_column", str(raised.exception))
        self.assertIn("features_specs", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
