"""Rows in, rows out -- in order, and independent of the batch.

Three things a preprocessor must not do quietly: return fewer rows than it was
given, return them shifted so row i of the output is row j of the input, or let
one row's output depend on which other rows share its batch. The last one is
the one that bites in production: a record scored alone and the same record
scored in a batch must come out the same, or training and serving disagree.

The mixture of experts did exactly that -- it routed on the batch mean, so one
row's values moved every other row's output.
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
BATCH = 12

SPECS = {
    "num": FeatureType.FLOAT_NORMALIZED,
    "disc": FeatureType.FLOAT_DISCRETIZED,
    "cat": FeatureType.STRING_CATEGORICAL,
    "txt": FeatureType.TEXT,
    "dt": FeatureType.DATE,
}
STRING_COLUMNS = {"cat", "txt", "dt"}
ALL_FEATURES = list(SPECS)


def _training_columns(rng, names):
    """Enough rows for the statistics pass."""
    generators = {
        "num": lambda: rng.normal(50, 10, ROWS),
        "disc": lambda: rng.normal(100, 25, ROWS),
        "cat": lambda: rng.choice(["paris", "lima", "oslo"], ROWS),
        "txt": lambda: rng.choice(["good one", "bad one", "fine one"], ROWS),
        "dt": lambda: pd.date_range("2020-01-01", periods=ROWS).strftime("%Y-%m-%d"),
    }
    return {name: generators[name]() for name in names}


def _batch(names):
    """A batch whose every row differs from every other."""
    values = {}
    for name in names:
        if name == "num":
            values[name] = [[float(50 + i)] for i in range(BATCH)]
        elif name == "disc":
            values[name] = [[float(60 + 4 * i)] for i in range(BATCH)]
        elif name == "cat":
            values[name] = [[["paris", "lima", "oslo"][i % 3]] for i in range(BATCH)]
        elif name == "txt":
            values[name] = [
                [["good one", "bad one", "fine one"][i % 3]] for i in range(BATCH)
            ]
        else:
            values[name] = [[f"2021-{1 + i % 12:02d}-15"] for i in range(BATCH)]
    return values


def _tensors(mapping):
    """Tensors of the right dtype for the built model."""
    return {
        name: tf.constant(
            values, dtype=tf.string if name in STRING_COLUMNS else tf.float32
        )
        for name, values in mapping.items()
    }


def _rows(output):
    """The output as a 2-D array, whichever output mode is in use."""
    if isinstance(output, dict):
        parts = [tf.cast(value, tf.float32) for value in output.values()]
        parts = [tf.reshape(part, (tf.shape(part)[0], -1)) for part in parts]
        return tf.concat(parts, axis=-1).numpy()
    values = tf.cast(output, tf.float32)
    return tf.reshape(values, (tf.shape(values)[0], -1)).numpy()


def _model(names, directory, **options):
    """A built preprocessor over those features."""
    keras.backend.clear_session()
    rng = np.random.default_rng(5)
    csv_path = Path(directory) / "data.csv"
    pd.DataFrame(_training_columns(rng, names)).to_csv(csv_path, index=False)
    preprocessor = PreprocessingModel(
        path_data=str(csv_path),
        features_specs={name: SPECS[name] for name in names},
        features_stats_path=str(Path(directory) / "stats.json"),
        overwrite_stats=True,
        **options,
    )
    preprocessor.build_preprocessor()
    return preprocessor


class RowIntegrityChecks:
    """The three checks, shared by every configuration below."""

    names = ALL_FEATURES
    options: dict = {}

    def test_every_row_comes_back(self):
        """Nothing may be dropped between input and output."""
        with tempfile.TemporaryDirectory() as tmp:
            model = _model(self.names, tmp, **self.options)
            output = _rows(model.model(_tensors(_batch(self.names)), training=False))
        self.assertEqual(output.shape[0], BATCH)

    def test_changing_one_row_moves_only_that_row(self):
        """A shift, or cross-row leakage, shows up here."""
        with tempfile.TemporaryDirectory() as tmp:
            model = _model(self.names, tmp, **self.options)
            batch = _batch(self.names)
            before = _rows(model.model(_tensors(batch), training=False))

            changed = {key: [list(row) for row in rows] for key, rows in batch.items()}
            changed[self.names[0]][5] = (
                [[999.0]][0]
                if self.names[0] not in STRING_COLUMNS
                else [["something else entirely"]][0]
            )
            after = _rows(model.model(_tensors(changed), training=False))

        moved = [
            index
            for index in range(BATCH)
            if not np.allclose(after[index], before[index], atol=1e-6)
        ]
        self.assertEqual(moved, [5])

    def test_a_row_scored_alone_matches_the_batch(self):
        """Batching is an implementation detail; it must not change values."""
        with tempfile.TemporaryDirectory() as tmp:
            model = _model(self.names, tmp, **self.options)
            batch = _batch(self.names)
            batched = _rows(model.model(_tensors(batch), training=False))
            single = {key: [rows[3]] for key, rows in batch.items()}
            alone = _rows(model.model(_tensors(single), training=False))
        np.testing.assert_allclose(alone[0], batched[3], atol=1e-5)


@pytest.mark.unit
class TestConcatMode(RowIntegrityChecks, unittest.TestCase):
    """The default output mode."""


@pytest.mark.unit
class TestDictMode(RowIntegrityChecks, unittest.TestCase):
    """One tensor per feature."""

    options = {"output_mode": "dict"}


@pytest.mark.unit
class TestTabularAttention(RowIntegrityChecks, unittest.TestCase):
    """Attention crosses features; it must not cross rows."""

    options = {"tabular_attention": True, "tabular_attention_placement": "all_features"}


@pytest.mark.unit
class TestTransformerBlocks(RowIntegrityChecks, unittest.TestCase):
    """The same question for the transformer path."""

    options = {"transfo_nr_blocks": 1, "transfo_placement": "all_features"}


@pytest.mark.unit
class TestFeatureMoE(RowIntegrityChecks, unittest.TestCase):
    """Routing used to be computed from the batch mean."""

    names = ["num", "disc"]
    options = {"use_feature_moe": True, "feature_moe_num_experts": 3}


if __name__ == "__main__":
    unittest.main()
