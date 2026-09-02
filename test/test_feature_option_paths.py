"""Feature-level options, each built and run.

The other sweeps vary the model-level switches. This one varies what goes on a
*feature*: encodings, text output modes, per-feature embeddings and the time
series configs. Four of these did not work at all, and none of them failed
loudly: `tf_idf` could not build, `use_embedding` was accepted and discarded,
`wavelet_transform_config` crashed on the shape the pipeline feeds it, and
`calendar_feature_config` tried to cast a date to a float.
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
from kdp.features import NumericalFeature, TextFeature, TimeSeriesFeature

ROWS = 300


def _frame():
    """One frame covering every column these tests need."""
    rng = np.random.default_rng(19)
    return pd.DataFrame(
        {
            "num": rng.normal(50, 10, ROWS),
            "txt": rng.choice(
                ["great value here", "poor quality item", "average product"], ROWS
            ),
            "series": np.linspace(1, 100, ROWS) + rng.normal(0, 1, ROWS),
            "dt": pd.date_range("2020-01-01", periods=ROWS).strftime("%Y-%m-%d"),
        },
    )


def _build(directory, specs):
    """A built preprocessor over those specs."""
    keras.backend.clear_session()
    csv_path = Path(directory) / "data.csv"
    _frame().to_csv(csv_path, index=False)
    preprocessor = PreprocessingModel(
        path_data=str(csv_path),
        features_specs=specs,
        features_stats_path=str(Path(directory) / "stats.json"),
        overwrite_stats=True,
    )
    preprocessor.build_preprocessor()
    return preprocessor


@pytest.mark.unit
class TestTextOutputModesBuildAndRun(unittest.TestCase):
    """`tf_idf` is exported and documented, and could not build."""

    def test_tf_idf_produces_weighted_tokens(self):
        """Keras needs `idf_weights` beside a vocabulary, which KDP had none of."""
        with tempfile.TemporaryDirectory() as tmp:
            model = _build(
                tmp,
                {
                    "txt": TextFeature(
                        name="txt",
                        feature_type=FeatureType.TEXT,
                        output_mode="tf_idf",
                        max_tokens=32,
                    ),
                },
            )
            output = model.model(
                {"txt": tf.constant([["great value here"], ["poor quality item"]])},
                training=False,
            ).numpy()

        self.assertEqual(output.shape[0], 2)
        # Weighted, not a plain count: the entries are not all 0 or 1.
        self.assertTrue(((output > 0) & (output != 1.0)).any())

    def test_the_two_documents_differ(self):
        """Different text has to produce different weights."""
        with tempfile.TemporaryDirectory() as tmp:
            model = _build(
                tmp,
                {
                    "txt": TextFeature(
                        name="txt",
                        feature_type=FeatureType.TEXT,
                        output_mode="tf_idf",
                        max_tokens=32,
                    ),
                },
            )
            output = model.model(
                {"txt": tf.constant([["great value here"], ["poor quality item"]])},
                training=False,
            ).numpy()
        self.assertFalse(np.allclose(output[0], output[1]))


@pytest.mark.unit
class TestPerFeatureNumericalEmbedding(unittest.TestCase):
    """`use_embedding` was stored on the feature and read nowhere."""

    def test_the_feature_gets_the_width_it_asked_for(self):
        """Without this the feature came through at its original width of one."""
        with tempfile.TemporaryDirectory() as tmp:
            model = _build(
                tmp,
                {
                    "num": NumericalFeature(
                        name="num",
                        feature_type=FeatureType.FLOAT_NORMALIZED,
                        use_embedding=True,
                        embedding_dim=6,
                    ),
                },
            )
        self.assertEqual(model.model.output_shape[-1], 6)

    def test_a_feature_that_did_not_ask_is_untouched(self):
        """The flag is per feature, not a model-wide switch."""
        with tempfile.TemporaryDirectory() as tmp:
            model = _build(tmp, {"num": FeatureType.FLOAT_NORMALIZED})
        self.assertEqual(model.model.output_shape[-1], 1)


@pytest.mark.unit
class TestCalendarFeatures(unittest.TestCase):
    """Calendar features read date strings, and the pipeline cast them to float."""

    def test_a_calendar_feature_builds_and_runs(self):
        """This raised "Cast string to float is not supported"."""
        with tempfile.TemporaryDirectory() as tmp:
            model = _build(
                tmp,
                {
                    "dt": TimeSeriesFeature(
                        name="dt",
                        calendar_feature_config={
                            "features": ["month", "day_of_week"],
                        },
                    ),
                },
            )
            output = model.model(
                {"dt": tf.constant([["2021-05-04"], ["2022-11-30"]])},
                training=False,
            ).numpy()

        self.assertEqual(output.shape, (2, 2))
        # Two different dates must not encode identically.
        self.assertFalse(np.allclose(output[0], output[1]))

    def test_the_feature_declares_a_string_column(self):
        """The input dtype is what decides whether the cast is inserted."""
        feature = TimeSeriesFeature(
            name="dt",
            calendar_feature_config={"features": ["month"]},
        )
        self.assertEqual(feature.dtype, tf.string)

    def test_calendar_cannot_be_mixed_with_numeric_configs(self):
        """One column cannot be both a date string and a number."""
        with self.assertRaises(ValueError) as caught:
            TimeSeriesFeature(
                name="dt",
                calendar_feature_config={"features": ["month"]},
                lag_config={"lags": [1]},
            )
        self.assertIn("lag_config", str(caught.exception))


@pytest.mark.unit
class TestWaveletFeatures(unittest.TestCase):
    """A wavelet needs a window of history, and got a single column."""

    def test_a_lone_wavelet_config_is_rejected(self):
        """It used to emit a constant column of zeros -- no information at all."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as caught:
                _build(
                    tmp,
                    {
                        "series": TimeSeriesFeature(
                            name="series",
                            wavelet_transform_config={"levels": 2},
                        ),
                    },
                )
        self.assertIn("two time steps", str(caught.exception))

    def test_a_wavelet_after_lags_carries_information(self):
        """Lags widen the feature, which gives the wavelet a window."""
        with tempfile.TemporaryDirectory() as tmp:
            model = _build(
                tmp,
                {
                    "series": TimeSeriesFeature(
                        name="series",
                        lag_config={"lags": [1, 2, 3]},
                        wavelet_transform_config={"levels": 2},
                    ),
                },
            )
            output = model.model(
                {"series": tf.constant([[10.0], [20.0], [30.0], [40.0], [50.0]])},
                training=False,
            ).numpy()

        self.assertEqual(output.shape[0], 5)
        self.assertGreater(len(set(np.round(output.ravel(), 4))), 1)


if __name__ == "__main__":
    unittest.main()
