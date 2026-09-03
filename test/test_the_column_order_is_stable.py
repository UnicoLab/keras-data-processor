"""Column N of the output has to be the same feature on every build.

Features are preprocessed in parallel batches, and the concatenated output was
assembled by walking `processed_features` -- a dict keyed in whichever order the
workers happened to finish. Building the same configuration twice therefore laid
the columns out differently: five distinct layouts in six builds of the same six
features.

Nothing raised, and the numbers were right for the layout each build chose. The
damage lands on anyone who trains on one build and serves from another: every
feature is read as a different one. Reloading a saved `.keras` file was always
safe, because the order is baked into its graph.
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
NUMERIC = [f"num{index}" for index in range(6)]

SPECS = {name: FeatureType.FLOAT_NORMALIZED for name in NUMERIC}
SPECS["cat_string"] = FeatureType.STRING_CATEGORICAL
SPECS["cat_integer"] = FeatureType.INTEGER_CATEGORICAL


def _frame() -> pd.DataFrame:
    generator = np.random.default_rng(67)
    frame = pd.DataFrame({name: generator.normal(0.0, 1.0, ROWS) for name in NUMERIC})
    frame["cat_string"] = generator.choice(["a", "b", "c"], ROWS)
    frame["cat_integer"] = generator.integers(0, 4, ROWS)
    return frame


class _Built:
    """One build of the preprocessor, with a way to locate each feature."""

    def __init__(self, directory: Path, first: bool, **options) -> None:
        keras.backend.clear_session()
        self.preprocessor = PreprocessingModel(
            path_data=str(directory / "data.csv"),
            features_stats_path=str(directory / "stats.json"),
            features_specs=SPECS,
            overwrite_stats=first,
            **options,
        )
        self.preprocessor.build_preprocessor()

    @property
    def names(self) -> tuple:
        return tuple(self.preprocessor._concat_feature_names)

    def columns_moved_by(self, feature: str) -> tuple:
        """Which output columns respond to a change in one numeric feature."""
        baseline = {name: tf.constant(np.zeros((3, 1)), np.float32) for name in NUMERIC}
        baseline["cat_string"] = tf.constant(np.array(["a", "a", "a"]).reshape(-1, 1))
        baseline["cat_integer"] = tf.constant(np.zeros((3, 1), dtype=np.int32))
        before = np.asarray(self.preprocessor.model(baseline))

        bumped = dict(baseline)
        bumped[feature] = tf.constant(np.full((3, 1), 7.0), np.float32)
        after = np.asarray(self.preprocessor.model(bumped))

        return tuple(np.where(np.abs(after - before).max(axis=0) > 1e-6)[0].tolist())


@pytest.mark.integration
class TestTheColumnOrderIsStable(unittest.TestCase):
    """The same configuration has to produce the same layout every time."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp())
        _frame().to_csv(cls.directory / "data.csv", index=False)

    def test_the_reported_order_does_not_move(self) -> None:
        orders = {_Built(self.directory, first=(run == 0)).names for run in range(5)}
        self.assertEqual(len(orders), 1, f"the feature order moved: {orders}")

    def test_it_is_the_order_the_features_were_declared_in(self) -> None:
        built = _Built(self.directory, first=True)
        # Numeric features are concatenated before categorical ones, and within
        # each block the declaration order holds.
        self.assertEqual(built.names[: len(NUMERIC)], tuple(NUMERIC))
        self.assertEqual(
            set(built.names[len(NUMERIC) :]),
            {"cat_string", "cat_integer"},
        )

    def test_each_feature_lands_in_the_same_columns_every_time(self) -> None:
        layouts = set()
        for run in range(4):
            built = _Built(self.directory, first=(run == 0))
            layouts.add(
                tuple((name, built.columns_moved_by(name)) for name in NUMERIC),
            )
        self.assertEqual(len(layouts), 1, f"the column layout moved: {layouts}")

    def test_a_numeric_feature_sits_at_its_declared_index(self) -> None:
        built = _Built(self.directory, first=True)
        for index, name in enumerate(NUMERIC):
            with self.subTest(feature=name):
                self.assertEqual(built.columns_moved_by(name), (index,))

    def test_the_mixture_of_experts_keeps_the_same_order(self) -> None:
        orders = {
            _Built(
                self.directory,
                first=(run == 0),
                use_feature_moe=True,
                feature_moe_num_experts=2,
                feature_moe_expert_dim=4,
            ).names
            for run in range(4)
        }
        self.assertEqual(len(orders), 1, f"the feature order moved: {orders}")

    def test_a_cross_comes_after_the_declared_features(self) -> None:
        built = _Built(
            self.directory,
            first=True,
            feature_crosses=[("cat_string", "cat_integer", 8)],
        )
        self.assertEqual(built.names[-1], "cat_string_x_cat_integer")
        self.assertEqual(built.names[: len(NUMERIC)], tuple(NUMERIC))


if __name__ == "__main__":
    unittest.main()
