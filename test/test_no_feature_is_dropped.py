"""Every declared feature has to reach the output.

Perturb one input column and watch the output tensor: a column whose change
moves nothing is being dropped somewhere in the graph, silently, because the
model builds and runs either way. That is the shape of the bug where the
mixture of experts multiplied an unassigned feature by an all-zero routing row
and it vanished, and of the one where the mixture cut the concatenated tensor
into equal parts and every "feature" it routed was a slice spanning several
real ones.

These are slow -- each case builds a model -- so the matrix is deliberately
small and each entry covers a different path through `_prepare_outputs`.
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

ROWS = 200

SPECS = {
    "num_norm": FeatureType.FLOAT_NORMALIZED,
    "num_resc": FeatureType.FLOAT_RESCALED,
    "num_disc": FeatureType.FLOAT_DISCRETIZED,
    "cat_str": FeatureType.STRING_CATEGORICAL,
    "cat_int": FeatureType.INTEGER_CATEGORICAL,
    "txt": FeatureType.TEXT,
    "dt": FeatureType.DATE,
}

# A baseline row and a row differing only in the feature under test.
BASELINE = {
    "num_norm": [[50.0]],
    "num_resc": [[5.0]],
    "num_disc": [[100.0]],
    "cat_str": [["paris"]],
    "cat_int": [[1]],
    "txt": [["great value here"]],
    "dt": [["2020-06-15"]],
}
PERTURBED = {
    "num_norm": [[85.0]],
    "num_resc": [[19.0]],
    "num_disc": [[180.0]],
    "cat_str": [["cairo"]],
    "cat_int": [[5]],
    "txt": [["poor quality item"]],
    "dt": [["2023-11-02"]],
}
STRING_COLUMNS = {"cat_str", "txt", "dt"}


def _column(name, rng):
    """A column of the right shape for the feature type."""
    generators = {
        "num_norm": lambda: rng.normal(50, 10, ROWS),
        "num_resc": lambda: rng.normal(5, 2, ROWS),
        "num_disc": lambda: rng.normal(100, 25, ROWS),
        "cat_str": lambda: rng.choice(["paris", "lima", "oslo", "cairo"], ROWS),
        "cat_int": lambda: rng.integers(0, 6, ROWS),
        "txt": lambda: rng.choice(
            ["great value here", "poor quality item", "average product"], ROWS
        ),
        "dt": lambda: pd.date_range("2020-01-01", periods=ROWS).strftime("%Y-%m-%d"),
    }
    return generators[name]()


def _tensors(mapping):
    """Turn plain values into the tensors the built model expects."""
    tensors = {}
    for name, values in mapping.items():
        if name in STRING_COLUMNS:
            tensors[name] = tf.constant(values, dtype=tf.string)
        elif name == "cat_int":
            tensors[name] = tf.constant(values, dtype=tf.int32)
        else:
            tensors[name] = tf.constant(values, dtype=tf.float32)
    return tensors


def _flat(output):
    """Flatten either output mode into one vector."""
    if isinstance(output, dict):
        parts = [tf.reshape(tf.cast(v, tf.float32), [-1]) for v in output.values()]
        return tf.concat(parts, axis=0).numpy()
    return tf.reshape(tf.cast(output, tf.float32), [-1]).numpy()


def _dropped_features(names, **options):
    """Names whose perturbation leaves the output identical."""
    keras.backend.clear_session()
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as directory:
        csv_path = Path(directory) / "data.csv"
        pd.DataFrame({n: _column(n, rng) for n in names}).to_csv(csv_path, index=False)
        model = PreprocessingModel(
            path_data=str(csv_path),
            features_specs={n: SPECS[n] for n in names},
            features_stats_path=str(Path(directory) / "stats.json"),
            overwrite_stats=True,
            **options,
        )
        model.build_preprocessor()

        baseline = {n: BASELINE[n] for n in names}
        reference = _flat(model.model(_tensors(baseline), training=False))

        dropped = []
        for name in names:
            changed = dict(baseline)
            changed[name] = PERTURBED[name]
            other = _flat(model.model(_tensors(changed), training=False))
            if other.shape == reference.shape and np.allclose(
                other, reference, atol=1e-7, rtol=0
            ):
                dropped.append(name)
    return dropped


ALL_FEATURES = list(SPECS)
NUMERIC_FEATURES = ["num_norm", "num_resc", "num_disc"]


@pytest.mark.unit
class TestEveryFeatureReachesTheOutput(unittest.TestCase):
    """One case per distinct path through the output preparation."""

    def test_concat_mode(self):
        """The default path."""
        self.assertEqual(_dropped_features(ALL_FEATURES), [])

    def test_dict_mode(self):
        """Per-feature outputs rather than one tensor."""
        self.assertEqual(_dropped_features(ALL_FEATURES, output_mode="dict"), [])

    def test_distribution_aware_encoding(self):
        """Each numeric column is transformed by its detected shape."""
        self.assertEqual(
            _dropped_features(ALL_FEATURES, use_distribution_aware=True), []
        )

    def test_tabular_attention(self):
        """Attention mixes features; none may be attenuated to nothing."""
        self.assertEqual(
            _dropped_features(
                ALL_FEATURES,
                tabular_attention=True,
                tabular_attention_placement="all_features",
            ),
            [],
        )

    def test_transformer_blocks(self):
        """The same, through the transformer path."""
        self.assertEqual(
            _dropped_features(
                ALL_FEATURES,
                transfo_nr_blocks=1,
                transfo_placement="all_features",
            ),
            [],
        )

    def test_feature_selection(self):
        """A gate of zero would drop its feature."""
        self.assertEqual(
            _dropped_features(ALL_FEATURES, feature_selection_placement="all_features"),
            [],
        )

    def test_advanced_numerical_embedding(self):
        """This combination did not build at all: rank 3 met rank 2."""
        self.assertEqual(
            _dropped_features(ALL_FEATURES, use_advanced_numerical_embedding=True), []
        )


@pytest.mark.unit
class TestFeatureMoEKeepsEveryFeature(unittest.TestCase):
    """The mixture is where features have actually gone missing."""

    def test_learned_routing(self):
        """Widths differ here: one column, one column, ten."""
        self.assertEqual(
            _dropped_features(
                NUMERIC_FEATURES, use_feature_moe=True, feature_moe_num_experts=3
            ),
            [],
        )

    def test_sparse_routing(self):
        """Routing to a single expert must still carry every feature."""
        self.assertEqual(
            _dropped_features(
                NUMERIC_FEATURES,
                use_feature_moe=True,
                feature_moe_num_experts=4,
                feature_moe_sparsity=1,
            ),
            [],
        )

    def test_predefined_routing(self):
        """The assignments name features, so the slices must match them."""
        self.assertEqual(
            _dropped_features(
                NUMERIC_FEATURES,
                use_feature_moe=True,
                feature_moe_num_experts=2,
                feature_moe_routing="predefined",
                feature_moe_assignments={
                    "num_norm": 0,
                    "num_resc": 1,
                    "num_disc": 0,
                },
            ),
            [],
        )

    def test_dict_mode(self):
        """This built and then raised on the first batch."""
        self.assertEqual(
            _dropped_features(
                NUMERIC_FEATURES,
                use_feature_moe=True,
                output_mode="dict",
                feature_moe_num_experts=2,
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
