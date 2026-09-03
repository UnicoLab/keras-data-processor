"""A saved preprocessor has to come back with the numbers it went in with.

Serialization is where this library has lost data silently: the model saves
without complaint, and the loss shows up only when someone needs the model
back. `output_mode="tf_idf"` was the last case -- Keras writes a `tf_idf`
vectorizer's IDF weights as layer variables and, on load, assigns them to a
layer that has no vocabulary yet and therefore no such variable, which surfaces
as `object of type 'bool' has no len()`. It happens with `TextVectorization`
alone, with no KDP in the picture; passing the vocabulary and the weights to
the constructor puts both in the config, where loading reads them.

Rather than testing that one case, this walks every option that changes what
gets built and checks the round trip is exact.
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
from kdp.features import (
    CategoricalFeature,
    DateFeature,
    NumericalFeature,
    PassthroughFeature,
    TextFeature,
)
from kdp.processor import OutputModeOptions

ROWS = 120

BASE = {
    "num1": FeatureType.FLOAT_NORMALIZED,
    "num2": FeatureType.FLOAT_RESCALED,
    "cat1": FeatureType.STRING_CATEGORICAL,
    "cat2": FeatureType.INTEGER_CATEGORICAL,
    "txt": FeatureType.TEXT,
    "date": FeatureType.DATE,
}

# (name, feature overrides, model options)
CONFIGURATIONS = [
    ("plain", {}, {}),
    (
        "one_hot",
        {
            "cat1": CategoricalFeature(
                name="cat1",
                feature_type=FeatureType.STRING_CATEGORICAL,
                category_encoding="ONE_HOT_ENCODING",
            ),
        },
        {},
    ),
    (
        "hashing",
        {
            "cat1": CategoricalFeature(
                name="cat1",
                feature_type=FeatureType.STRING_CATEGORICAL,
                category_encoding="HASHING",
                hash_bucket_size=8,
                salt=7,
            ),
        },
        {},
    ),
    (
        "discretized",
        {
            "num1": NumericalFeature(
                name="num1",
                feature_type=FeatureType.FLOAT_DISCRETIZED,
                bin_boundaries=[-1.0, 0.0, 1.0],
            ),
        },
        {},
    ),
    (
        "numeric_embedding",
        {
            "num1": NumericalFeature(
                name="num1",
                feature_type=FeatureType.FLOAT_NORMALIZED,
                use_embedding=True,
                embedding_dim=6,
            ),
        },
        {},
    ),
    (
        "distribution_aware",
        {
            "num1": NumericalFeature(
                name="num1",
                feature_type=FeatureType.FLOAT_NORMALIZED,
                use_distribution_aware=True,
            ),
        },
        {},
    ),
    (
        "text_multi_hot",
        {"txt": TextFeature(name="txt", output_mode="multi_hot")},
        {},
    ),
    ("text_tf_idf", {"txt": TextFeature(name="txt", output_mode="tf_idf")}, {}),
    (
        "text_ngrams",
        {"txt": TextFeature(name="txt", output_mode="multi_hot", ngrams=2)},
        {},
    ),
    (
        "text_capped",
        {"txt": TextFeature(name="txt", output_mode="multi_hot", max_tokens=4)},
        {},
    ),
    ("date_season", {"date": DateFeature(name="date", add_season=True)}, {}),
    ("passthrough", {"num2": PassthroughFeature(name="num2", dtype=tf.float32)}, {}),
    ("crosses", {}, {"feature_crosses": [("cat1", "cat2", 8)]}),
    (
        "attention",
        {},
        {
            "tabular_attention": True,
            "tabular_attention_heads": 2,
            "tabular_attention_dim": 16,
        },
    ),
    (
        "transformer_categorical",
        {},
        {
            "transfo_nr_blocks": 2,
            "transfo_nr_heads": 2,
            "transfo_ff_units": 8,
            "transfo_placement": "categorical",
        },
    ),
    (
        "transformer_all_features",
        {},
        {
            "transfo_nr_blocks": 2,
            "transfo_nr_heads": 2,
            "transfo_ff_units": 8,
            "transfo_placement": "all_features",
        },
    ),
    (
        "feature_selection",
        {},
        {"feature_selection_placement": "all_features", "feature_selection_units": 8},
    ),
    ("feature_moe", {}, {"use_feature_moe": True, "feature_moe_num_experts": 3}),
    (
        "global_numerical_embedding",
        {},
        {"use_global_numerical_embedding": True, "global_embedding_dim": 6},
    ),
    ("dict_mode", {}, {"output_mode": OutputModeOptions.DICT}),
    (
        "dict_mode_feature_moe",
        {},
        {
            "output_mode": OutputModeOptions.DICT,
            "use_feature_moe": True,
            "feature_moe_num_experts": 3,
        },
    ),
]


def _frame() -> pd.DataFrame:
    generator = np.random.default_rng(41)
    return pd.DataFrame(
        {
            "num1": generator.normal(0.0, 1.0, ROWS),
            "num2": generator.uniform(0.0, 10.0, ROWS),
            "cat1": generator.choice(["a", "b", "c"], ROWS),
            "cat2": generator.integers(0, 5, ROWS),
            "txt": generator.choice(["red car fast", "blue sky wide"], ROWS),
            "date": pd.date_range("2020-01-01", periods=ROWS, freq="3D").strftime(
                "%Y-%m-%d",
            ),
        },
    )


def _feed(frame: pd.DataFrame) -> dict:
    batch = {}
    for column in frame.columns:
        values = frame[column].to_numpy()
        if values.dtype.kind == "f":
            batch[column] = tf.constant(values.reshape(-1, 1), dtype=tf.float32)
        elif values.dtype.kind in "iu":
            batch[column] = tf.constant(values.reshape(-1, 1), dtype=tf.int32)
        else:
            batch[column] = tf.constant(values.reshape(-1, 1))
    return batch


def _flatten(result) -> np.ndarray:
    if isinstance(result, dict):
        return np.concatenate(
            [np.asarray(result[key]).reshape(ROWS, -1) for key in sorted(result)],
            axis=1,
        )
    return np.asarray(result)


@pytest.mark.integration
class TestEveryConfigurationReloads(unittest.TestCase):
    """Save, load, and compare the numbers, for every option that builds."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp())
        cls.frame = _frame()
        cls.data = cls.directory / "data.csv"
        cls.frame.to_csv(cls.data, index=False)

    def _round_trip(self, name: str, overrides: dict, options: dict) -> None:
        specs = dict(BASE)
        specs.update(overrides)

        keras.backend.clear_session()
        preprocessor = PreprocessingModel(
            path_data=str(self.data),
            features_stats_path=str(self.directory / f"stats_{name}.json"),
            features_specs=specs,
            overwrite_stats=True,
            **options,
        )
        preprocessor.build_preprocessor()

        batch = _feed(self.frame)
        before = _flatten(preprocessor.model(batch))

        path = self.directory / f"{name}.keras"
        preprocessor.model.save(path)
        after = _flatten(keras.models.load_model(path)(batch))

        self.assertEqual(before.shape, after.shape, name)
        np.testing.assert_allclose(before, after, atol=1e-5, err_msg=name)

    def test_every_configuration_round_trips(self) -> None:
        for name, overrides, options in CONFIGURATIONS:
            with self.subTest(configuration=name):
                self._round_trip(name, overrides, options)


@pytest.mark.unit
class TestTfIdfKeepsItsWeights(unittest.TestCase):
    """The rebuilt vectorizer has to compute what the adapted one computed."""

    def test_it_matches_a_keras_layer_adapted_on_the_same_data(self) -> None:
        directory = Path(tempfile.mkdtemp())
        sentences = [
            "red car fast",
            "blue sky wide",
            "red red red",
            "green grass here",
        ]
        frame = pd.DataFrame(
            {"txt": np.random.default_rng(41).choice(sentences, 200)},
        )
        data = directory / "data.csv"
        frame.to_csv(data, index=False)

        preprocessor = PreprocessingModel(
            path_data=str(data),
            features_stats_path=str(directory / "stats.json"),
            features_specs={"txt": TextFeature(name="txt", output_mode="tf_idf")},
            overwrite_stats=True,
            output_mode=OutputModeOptions.DICT,
        )
        preprocessor.build_preprocessor()

        probe = tf.constant(np.array(sentences).reshape(-1, 1))
        ours = np.asarray(preprocessor.model({"txt": probe})["txt"])

        reference = keras.layers.TextVectorization(output_mode="tf_idf")
        reference.adapt(tf.constant(frame["txt"].to_numpy()))
        theirs = np.asarray(reference(tf.reshape(probe, [-1])))

        self.assertEqual(ours.shape, theirs.shape)
        np.testing.assert_allclose(ours, theirs, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
