"""Round-trip serialization tests for the preprocessing model.

These tests guard the "deploy your preprocessing together with your model"
promise: every KDP layer must be registered with the Keras serialization
registry and must rebuild its sub-layers eagerly, otherwise
``PreprocessingModel.load_model`` raises instead of returning a usable model.
"""

import keras
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kdp import PreprocessingModel
from kdp.features import FeatureType

FEATURES_SPECS = {
    "age": FeatureType.FLOAT_NORMALIZED,
    "income": FeatureType.FLOAT_RESCALED,
    # Ten one-hot columns next to the one-column features above, so every
    # configuration below is exercised on a feature set of mixed widths --
    # the case where the mixture of experts used to slice mid-feature and
    # where the padding it now inserts has to survive the round trip.
    "score": FeatureType.FLOAT_DISCRETIZED,
    "occupation": FeatureType.STRING_CATEGORICAL,
    "description": FeatureType.TEXT,
    "signup_date": FeatureType.DATE,
}

SAMPLE_BATCH = {
    "age": tf.constant([[0.5], [1.5]]),
    "income": tf.constant([[100.0], [200.0]]),
    "score": tf.constant([[42.0], [77.0]]),
    "occupation": tf.constant([["engineer"], ["doctor"]]),
    "description": tf.constant([["hello world"], ["lorem ipsum"]]),
    "signup_date": tf.constant([["2021-05-04"], ["2022-11-30"]]),
}

# Each entry is a different preprocessing configuration that must survive a
# save/load round-trip.
ROUND_TRIP_CONFIGS = [
    ("basic", {}),
    ("distribution_aware", {"use_distribution_aware": True}),
    ("tabular_attention", {"tabular_attention": True}),
    ("transformer_blocks", {"transfo_nr_blocks": 1}),
    ("feature_selection", {"feature_selection_placement": "all_features"}),
    ("numerical_embedding", {"use_advanced_numerical_embedding": True}),
    ("global_numerical_embedding", {"use_global_numerical_embedding": True}),
    ("feature_moe", {"use_feature_moe": True, "feature_moe_num_experts": 2}),
    (
        "feature_moe_sparse",
        {
            "use_feature_moe": True,
            "feature_moe_num_experts": 4,
            "feature_moe_sparsity": 2,
        },
    ),
    (
        "feature_moe_predefined",
        {
            "use_feature_moe": True,
            "feature_moe_num_experts": 2,
            "feature_moe_routing": "predefined",
            "feature_moe_assignments": {
                "age": 0,
                "income": 1,
                "score": 0,
                "occupation": 1,
                "description": 0,
                "signup_date": 1,
            },
        },
    ),
    (
        "multi_resolution_attention",
        {
            "tabular_attention": True,
            "tabular_attention_placement": "multi_resolution",
        },
    ),
    ("dict_output", {"output_mode": "dict"}),
    (
        "moe_dict_output",
        {
            "use_feature_moe": True,
            "feature_moe_num_experts": 2,
            "output_mode": "dict",
        },
    ),
]


def _write_dataset(directory: Path) -> Path:
    """Write a small CSV covering every feature type used by these tests."""
    rng = np.random.default_rng(42)
    csv_path = directory / "data.csv"
    pd.DataFrame(
        {
            "age": rng.normal(size=120).astype("float32"),
            "income": (rng.random(120) * 1000).astype("float32"),
            "score": (rng.random(120) * 100).astype("float32"),
            "occupation": rng.choice(["engineer", "doctor", "teacher"], 120),
            "description": rng.choice(
                ["hello world", "lorem ipsum", "foo bar baz"], 120
            ),
            "signup_date": pd.date_range("2020-01-01", periods=120).strftime(
                "%Y-%m-%d"
            ),
        }
    ).to_csv(csv_path, index=False)
    return csv_path


def _assert_same_output(expected, actual) -> None:
    """Assert two model outputs match, handling both concat and dict modes."""
    if isinstance(expected, dict):
        assert set(expected) == set(actual)
        for key in expected:
            np.testing.assert_allclose(
                np.asarray(expected[key]), np.asarray(actual[key]), atol=1e-5
            )
    else:
        np.testing.assert_allclose(np.asarray(expected), np.asarray(actual), atol=1e-5)


def _config_id(entry):
    """Name a case, whether it is a plain tuple or carries marks."""
    values = entry.values if isinstance(entry, type(pytest.param(0))) else entry
    return values[0]


@pytest.mark.parametrize(
    "config_name,config",
    ROUND_TRIP_CONFIGS,
    ids=[_config_id(entry) for entry in ROUND_TRIP_CONFIGS],
)
def test_model_survives_save_load_round_trip(config_name, config, tmp_path):
    """A saved preprocessor reloads and produces byte-identical predictions."""
    keras.backend.clear_session()
    csv_path = _write_dataset(tmp_path)

    preprocessor = PreprocessingModel(
        path_data=str(csv_path),
        features_specs=dict(FEATURES_SPECS),
        features_stats_path=str(tmp_path / f"stats_{config_name}.json"),
        overwrite_stats=True,
        **config,
    )
    preprocessor.build_preprocessor()
    expected = preprocessor.model(SAMPLE_BATCH)

    save_dir = tmp_path / f"model_{config_name}"
    preprocessor.save_model(str(save_dir))

    loaded_model, metadata = PreprocessingModel.load_model(str(save_dir))
    _assert_same_output(expected, loaded_model(SAMPLE_BATCH))
    assert "output_mode" in metadata


class TestLayerRegistration(unittest.TestCase):
    """Every KDP layer must be discoverable by the Keras deserializer."""

    def test_all_kdp_layers_are_registered(self):
        """Each exported layer resolves through the Keras custom object registry."""
        from keras.saving import get_registered_name, get_registered_object

        import kdp.layers as kdp_layers

        unregistered = []
        for name in kdp_layers.__all__:
            obj = getattr(kdp_layers, name)
            if not (isinstance(obj, type) and issubclass(obj, keras.layers.Layer)):
                continue  # e.g. the DistributionType enum
            # An unregistered class serializes under its bare name and cannot be
            # looked up again; a registered one gets a "package>Name" key.
            registered_name = get_registered_name(obj)
            if get_registered_object(registered_name) is not obj:
                unregistered.append(name)

        self.assertEqual(
            unregistered,
            [],
            f"Layers missing @register_keras_serializable: {unregistered}",
        )

    def test_layer_registration_survives_reserialization(self):
        """A registered layer can be serialized and deserialized standalone."""
        from kdp.layers import DateParsingLayer

        layer = DateParsingLayer(date_format="YYYY-MM-DD", name="parse")
        restored = keras.layers.deserialize(keras.layers.serialize(layer))
        self.assertIsInstance(restored, DateParsingLayer)
        self.assertEqual(restored.date_format, "YYYY-MM-DD")


class TestSaveModelErrors(unittest.TestCase):
    """Failure modes of save/load should stay explicit."""

    def test_save_before_build_raises(self):
        """Saving an unbuilt preprocessor is an error, not a silent no-op."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_dataset(Path(tmp))
            preprocessor = PreprocessingModel(
                path_data=str(csv_path),
                features_specs={"age": FeatureType.FLOAT_NORMALIZED},
                features_stats_path=str(Path(tmp) / "stats.json"),
                overwrite_stats=True,
            )
            with self.assertRaises(ValueError):
                preprocessor.save_model(str(Path(tmp) / "out"))

    def test_load_from_missing_directory_raises(self):
        """Loading a non-existent model directory raises a clear error."""
        with self.assertRaises(ValueError):
            PreprocessingModel.load_model("/nonexistent/path/to/model")


def _config_keys_the_constructor_rejects() -> list[tuple[str, str, list[str]]]:
    """Classes whose `get_config` writes a key `__init__` will not take.

    Keras rebuilds a layer by calling `from_config`, which hands the stored
    config straight to `__init__`; anything the constructor does not name
    reaches `Layer.__init__` as an unexpected keyword and Keras raises. So a
    single extra key in `get_config` makes every saved model containing that
    layer impossible to load, and nothing notices until someone tries.
    `TextPreprocessingLayer` stored two patterns it derives from `stop_words`
    and could not be reloaded at all.
    """
    import ast

    root = Path(__file__).resolve().parent.parent / "kdp"
    findings = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            members = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
            if "__init__" not in members or "get_config" not in members:
                continue
            init, config = members["__init__"], members["get_config"]
            accepted = {a.arg for a in init.args.args if a.arg != "self"}
            accepted |= {a.arg for a in init.args.kwonlyargs}

            written = set()
            for node in ast.walk(config):
                if isinstance(node, ast.Dict):
                    written |= {
                        key.value
                        for key in node.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    }
                if isinstance(node, ast.Assign) and isinstance(
                    node.targets[0],
                    ast.Subscript,
                ):
                    index = node.targets[0].slice
                    if isinstance(index, ast.Constant) and isinstance(
                        index.value,
                        str,
                    ):
                        written.add(index.value)

            extra = sorted(written - accepted)
            if extra:
                findings.append(
                    (str(path.relative_to(root.parent)), cls.name, extra),
                )
    return findings


@pytest.mark.unit
class TestEveryConfigKeyCanBeRead(unittest.TestCase):
    """A config a constructor cannot read is a model that cannot be loaded."""

    def test_no_class_writes_a_config_key_its_constructor_rejects(self):
        findings = _config_keys_the_constructor_rejects()
        message = "\n".join(
            f"  {path}: {cls}.get_config writes {keys}, which __init__ "
            "does not accept -- a saved model carrying this layer will not load."
            for path, cls, keys in findings
        )
        self.assertEqual(findings, [], f"\n{message}")

    def test_the_text_layer_survives_a_round_trip(self):
        """The layer that could not be reloaded, and the old configs of it."""
        from kdp.layers.text_preprocessing_layer import TextPreprocessingLayer

        probe = tf.constant([["The quick, brown fox!"], ["A lazy dog."]])
        inputs = keras.Input(shape=(1,), dtype=tf.string)
        model = keras.Model(
            inputs,
            TextPreprocessingLayer(stop_words=["the", "a"])(inputs),
        )
        before = np.asarray(model(probe))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.keras"
            model.save(path)
            reloaded = keras.saving.load_model(path)
        np.testing.assert_array_equal(np.asarray(reloaded(probe)), before)

        # A config written by an earlier release still loads.
        legacy = {
            "name": "text",
            "stop_words": ["the"],
            "punctuation_pattern": "ignored",
            "stop_words_pattern": "the",
        }
        self.assertEqual(
            TextPreprocessingLayer.from_config(legacy).stop_words,
            ["the"],
        )
