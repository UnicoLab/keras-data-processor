"""Tests that every Feature-MoE option reaches the layer in concat mode.

`_apply_feature_moe`, the path taken by the default `output_mode="concat"`,
constructed `FeatureMoE` with only `num_experts`, `expert_dim` and `routing`.
`feature_moe_hidden_dims`, `feature_moe_sparsity`, `feature_moe_freeze_experts`
and `feature_moe_dropout` were therefore ignored, and
`feature_moe_routing="predefined"` raised, because the assignments and feature
names never arrived. The dict-mode branch passed them all along.
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

COLUMNS = ("n1", "n2")
BATCH = {"n1": tf.constant([[50.0]]), "n2": tf.constant([[5.0]])}


def _dataset(directory, rows: int = 200):
    """Two numeric columns is enough to route between two experts."""
    rng = np.random.default_rng(0)
    csv_path = directory / "moe.csv"
    pd.DataFrame({c: rng.normal(50, 10, rows) for c in COLUMNS}).to_csv(
        csv_path, index=False
    )
    return csv_path


def _build(tmp_path, **kwargs):
    keras.backend.clear_session()
    preprocessor = PreprocessingModel(
        path_data=str(_dataset(tmp_path)),
        features_specs={c: FeatureType.FLOAT_NORMALIZED for c in COLUMNS},
        features_stats_path=str(tmp_path / "stats.json"),
        overwrite_stats=True,
        use_feature_moe=True,
        **{"feature_moe_num_experts": 2, "feature_moe_expert_dim": 16, **kwargs},
    )
    preprocessor.build_preprocessor()
    return preprocessor


@pytest.mark.unit
class TestFeatureMoEOptionsInConcatMode(unittest.TestCase):
    """Concat mode is the default, so these are the common path."""

    def test_predefined_routing_builds(self):
        """This raised: feature_names and assignments were never passed."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(
                Path(tmp),
                feature_moe_routing="predefined",
                feature_moe_assignments={"n1": 0, "n2": 1},
            )
            output = preprocessor.model(BATCH)
        self.assertEqual(int(output.shape[-1]), len(COLUMNS) * 16)

    def test_predefined_routing_still_requires_assignments(self):
        """The layer's own validation must keep working."""
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            _build(Path(tmp), feature_moe_routing="predefined")

    def test_features_are_split_at_their_real_boundaries(self):
        """The split used to be an equal division of the concatenated width.

        `processed_features_dims` was written as a flat `{name: dim}` mapping
        and read as a nested one keyed by "numeric"/"categorical", so the
        lookup always missed and the fallback cut `concat_all` into equal
        parts. With features of unequal width every slice the router saw
        spanned several real features.
        """
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rng = np.random.default_rng(0)
            csv_path = directory / "mixed.csv"
            pd.DataFrame(
                {
                    "narrow": rng.normal(50, 10, 200),
                    "wide": rng.normal(100, 25, 200),
                },
            ).to_csv(csv_path, index=False)

            keras.backend.clear_session()
            preprocessor = PreprocessingModel(
                path_data=str(csv_path),
                features_specs={
                    "narrow": FeatureType.FLOAT_NORMALIZED,  # one column
                    "wide": FeatureType.FLOAT_DISCRETIZED,  # ten one-hot columns
                },
                features_stats_path=str(directory / "stats.json"),
                overwrite_stats=True,
                use_feature_moe=True,
                feature_moe_num_experts=2,
                feature_moe_expert_dim=8,
            )
            preprocessor.build_preprocessor()
            dims = preprocessor.model.get_layer("split_layer").feature_dims

        # The real widths, in the order the features were concatenated -- not
        # two equal halves of eleven.
        self.assertEqual(sorted(dims), [1, 10])

    def test_the_split_matches_the_names_given_to_the_router(self):
        """Predefined routing names features, so the slices must line up."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rng = np.random.default_rng(0)
            csv_path = directory / "mixed.csv"
            pd.DataFrame(
                {
                    "narrow": rng.normal(50, 10, 200),
                    "wide": rng.normal(100, 25, 200),
                },
            ).to_csv(csv_path, index=False)

            keras.backend.clear_session()
            preprocessor = PreprocessingModel(
                path_data=str(csv_path),
                features_specs={
                    "narrow": FeatureType.FLOAT_NORMALIZED,
                    "wide": FeatureType.FLOAT_DISCRETIZED,
                },
                features_stats_path=str(directory / "stats.json"),
                overwrite_stats=True,
                use_feature_moe=True,
                feature_moe_num_experts=2,
            )
            preprocessor.build_preprocessor()
            dims = preprocessor.model.get_layer("split_layer").feature_dims
            names = preprocessor.model.get_layer("feature_moe_concat").feature_names
            widths = preprocessor.processed_features_dims

        self.assertEqual(len(names), len(dims))
        self.assertEqual([widths[name] for name in names], list(dims))
        self.assertEqual(widths, {"narrow": 1, "wide": 10})

    def test_unequal_widths_are_padded_rather_than_rejected(self):
        """Stacking needs one width; padding keeps each feature whole."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rng = np.random.default_rng(0)
            csv_path = directory / "mixed.csv"
            pd.DataFrame(
                {
                    "narrow": rng.normal(50, 10, 200),
                    "wide": rng.normal(100, 25, 200),
                },
            ).to_csv(csv_path, index=False)

            keras.backend.clear_session()
            preprocessor = PreprocessingModel(
                path_data=str(csv_path),
                features_specs={
                    "narrow": FeatureType.FLOAT_NORMALIZED,
                    "wide": FeatureType.FLOAT_DISCRETIZED,
                },
                features_stats_path=str(directory / "stats.json"),
                overwrite_stats=True,
                use_feature_moe=True,
                feature_moe_num_experts=2,
                feature_moe_expert_dim=8,
            )
            preprocessor.build_preprocessor()
            padded = [
                layer.name
                for layer in preprocessor.model.layers
                if layer.name.startswith("moe_pad_")
            ]
            output = preprocessor.model(
                {
                    "narrow": tf.constant([[50.0]]),
                    "wide": tf.constant([[100.0]]),
                },
            )

        # Every feature gets a padding layer, the widest one as a no-op, so
        # they all sit at the same depth in the graph.
        self.assertEqual(sorted(padded), ["moe_pad_narrow", "moe_pad_wide"])
        self.assertEqual(int(output.shape[-1]), 2 * 8)

    def test_a_uniform_width_model_is_not_padded(self):
        """Padding must not appear where every feature is already one width."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(Path(tmp), feature_moe_num_experts=2)
            padded = [
                layer.name
                for layer in preprocessor.model.layers
                if layer.name.startswith("moe_pad_")
            ]
        self.assertEqual(padded, [])

    def test_global_numerical_embedding_is_rejected(self):
        """It merges every numeric feature, leaving no slices to route."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as caught:
                _build(Path(tmp), use_global_numerical_embedding=True)
        self.assertIn("global embedding", str(caught.exception))

    def test_an_unassigned_feature_is_rejected(self):
        """It used to be zeroed out: an all-zero routing row erases the feature."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as caught:
                _build(
                    Path(tmp),
                    feature_moe_routing="predefined",
                    feature_moe_assignments={"n1": 0},
                )
        self.assertIn("n2", str(caught.exception))

    def test_an_out_of_range_expert_is_rejected(self):
        """Routing to an expert that does not exist is a configuration error."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as caught:
                _build(
                    Path(tmp),
                    feature_moe_routing="predefined",
                    feature_moe_assignments={"n1": 0, "n2": 7},
                )
        self.assertIn("out of range", str(caught.exception))

    def test_an_unknown_feature_name_is_rejected(self):
        """Naming a feature the mixture never sees is a silent no-op otherwise."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as caught:
                _build(
                    Path(tmp),
                    feature_moe_routing="predefined",
                    feature_moe_assignments={"n1": 0, "n2": 1, "absent": 0},
                )
        self.assertIn("absent", str(caught.exception))

    def test_weighted_assignments_are_accepted(self):
        """A feature may be split across experts with explicit weights."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(
                Path(tmp),
                feature_moe_routing="predefined",
                feature_moe_assignments={"n1": {0: 0.7, 1: 0.3}, "n2": 1},
            )
            output = preprocessor.model(BATCH)
        self.assertEqual(int(output.shape[-1]), len(COLUMNS) * 16)

    def test_every_routed_feature_keeps_a_signal(self):
        """The regression this guards: a routed feature must not come out zero."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(
                Path(tmp),
                feature_moe_routing="predefined",
                feature_moe_assignments={"n1": 0, "n2": 1},
                feature_moe_use_residual=False,
            )
            output = preprocessor.model(BATCH).numpy().reshape(len(COLUMNS), 16)
        for index, column in enumerate(COLUMNS):
            self.assertGreater(
                float(np.abs(output[index]).sum()),
                0.0,
                f"{column} was zeroed out by the router",
            )

    def test_hidden_dims_change_the_expert_networks(self):
        """Ignored before, so the parameter count did not move."""
        with tempfile.TemporaryDirectory() as tmp:
            plain = _build(Path(tmp)).model.count_params()
        with tempfile.TemporaryDirectory() as tmp:
            deeper = _build(
                Path(tmp), feature_moe_hidden_dims=[32, 32]
            ).model.count_params()
        self.assertGreater(deeper, plain)

    def test_freezing_experts_removes_them_from_training(self):
        """`freeze_experts` never reached the layer either."""
        with tempfile.TemporaryDirectory() as tmp:
            trainable = _build(Path(tmp)).model
        with tempfile.TemporaryDirectory() as tmp:
            frozen = _build(Path(tmp), feature_moe_freeze_experts=True).model

        def count_trainable(model):
            return int(sum(np.prod(w.shape) for w in model.trainable_weights))

        self.assertLess(count_trainable(frozen), count_trainable(trainable))

    def test_learned_routing_is_unchanged(self):
        """The default path must behave as before."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(Path(tmp))
            output = preprocessor.model(BATCH)
        self.assertEqual(int(output.shape[-1]), len(COLUMNS) * 16)


if __name__ == "__main__":
    unittest.main()


@pytest.mark.unit
class TestFeatureMoEResidual(unittest.TestCase):
    """`feature_moe_use_residual` was stored on the model and read nowhere."""

    @staticmethod
    def _residual_layers(model) -> list[str]:
        """Names of the Add layers that implement the residual."""
        return [layer.name for layer in model.layers if "moe_residual" in layer.name]

    def test_residual_is_added_when_widths_match(self):
        """A width-preserving expert can be added to its input."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(Path(tmp), feature_moe_expert_dim=1)
        self.assertTrue(self._residual_layers(preprocessor.model))

    def test_no_residual_when_the_expert_changes_the_width(self):
        """There is nothing to add a 1-wide feature to a 16-wide output."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(Path(tmp), feature_moe_expert_dim=16)
        self.assertEqual(self._residual_layers(preprocessor.model), [])

    def test_disabling_it_removes_the_connection(self):
        """The flag now actually controls something."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(
                Path(tmp), feature_moe_expert_dim=1, feature_moe_use_residual=False
            )
        self.assertEqual(self._residual_layers(preprocessor.model), [])

    def test_the_model_still_runs_with_a_residual(self):
        """The connection must not break the forward pass."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(Path(tmp), feature_moe_expert_dim=1)
            output = preprocessor.model(BATCH)
        self.assertEqual(int(output.shape[-1]), len(COLUMNS))


@pytest.mark.unit
class TestAddFeatureMoeToModel(unittest.TestCase):
    """The module-level helper, which nothing exercised.

    `add_feature_moe_to_model` stacked the feature outputs without padding
    them, so it worked only when every feature happened to be the same width --
    which preprocessing rarely produces. `PreprocessingModel` pads for exactly
    this reason and this now does the same.
    """

    @staticmethod
    def _model(widths, **kwargs):
        from kdp.moe import add_feature_moe_to_model

        keras.backend.clear_session()
        names = tuple("abcde"[: len(widths)])
        inputs = {name: keras.Input(shape=(1,), name=name) for name in names}
        outputs = [
            keras.layers.Dense(width, name=f"preprocessed_{name}")(inputs[name])
            for name, width in zip(names, widths, strict=True)
        ]
        base = keras.Model(inputs=list(inputs.values()), outputs=outputs)
        kwargs.setdefault("num_experts", 2)
        kwargs.setdefault("expert_dim", 4)
        return names, add_feature_moe_to_model(base, inputs, **kwargs)

    def test_features_of_different_widths(self):
        for widths in ([3, 3, 3], [2, 3, 5], [1, 1, 10]):
            with self.subTest(widths=widths):
                names, model = self._model(widths)
                probe = {
                    name: tf.constant([[float(index + 1)]])
                    for index, name in enumerate(names)
                }
                outputs = model(probe)
                self.assertEqual(len(outputs), len(names))
                for output in outputs:
                    self.assertEqual(int(output.shape[-1]), 4)

    def test_every_feature_still_reaches_its_own_output(self):
        """Padding must not let a feature's signal go missing."""
        names, model = self._model([2, 3, 5])
        baseline_input = {name: tf.constant([[1.0]]) for name in names}
        baseline = [np.asarray(t) for t in model(baseline_input)]
        for index, name in enumerate(names):
            moved = dict(baseline_input)
            moved[name] = tf.constant([[9.0]])
            output = np.asarray(model(moved)[index])
            self.assertFalse(np.allclose(output, baseline[index]), name)

    def test_the_result_survives_a_round_trip(self):
        names, model = self._model([2, 3, 5])
        probe = {
            name: tf.constant([[float(index + 1)]]) for index, name in enumerate(names)
        }
        before = [np.asarray(t) for t in model(probe)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "moe.keras"
            model.save(path)
            after = [np.asarray(t) for t in keras.saving.load_model(path)(probe)]
        for expected, actual in zip(before, after, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-5)
