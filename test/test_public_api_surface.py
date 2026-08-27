"""Tests for the top-level entry points users are pointed at.

`auto_configure`, `DatasetStatistics` and `PreprocessingModel.predict` are all
documented as standalone starting points, and each had a defect that only shows
up when they are used that way rather than through the processor's own wiring.
"""

import unittest

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from kdp import DatasetStatistics, PreprocessingModel, auto_configure
from kdp.features import FeatureType

SPECS = {
    "age": FeatureType.FLOAT_NORMALIZED,
    "city": FeatureType.STRING_CATEGORICAL,
    "bio": FeatureType.TEXT,
    "joined": FeatureType.DATE,
}


def _write_dataset(directory, rows: int = 120):
    """Write a CSV covering the four feature types used here."""
    rng = np.random.default_rng(11)
    csv_path = directory / "data.csv"
    pd.DataFrame(
        {
            "age": rng.normal(40, 12, rows),
            "city": rng.choice(["paris", "tokyo", "lima"], rows),
            "bio": rng.choice(["hello world", "data science"], rows),
            "joined": pd.date_range("2020-01-01", periods=rows).strftime("%Y-%m-%d"),
        }
    ).to_csv(csv_path, index=False)
    return csv_path


class TestDatasetStatisticsFromSpecs(unittest.TestCase):
    """features_specs alone must be enough to compute statistics."""

    def test_specs_populate_the_per_type_feature_lists(
        self,
    ):
        """The lists used to stay empty, so no accumulator was ever created."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stats = DatasetStatistics(
                path_data=str(_write_dataset(tmp_path)),
                features_specs=dict(SPECS),
                features_stats_path=str(tmp_path / "stats.json"),
                overwrite_stats=True,
            )
            self.assertEqual(stats.numeric_features, ["age"])
            self.assertEqual(stats.categorical_features, ["city"])
            self.assertEqual(stats.text_features, ["bio"])
            self.assertEqual(stats.date_features, ["joined"])

    def test_main_returns_statistics_rather_than_an_empty_dict(self):
        """A statistics run driven by specs produces real numbers."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            computed = DatasetStatistics(
                path_data=str(_write_dataset(tmp_path)),
                features_specs=dict(SPECS),
                features_stats_path=str(tmp_path / "stats.json"),
                overwrite_stats=True,
            ).main()

        self.assertIn("numeric_stats", computed)
        self.assertIn("categorical_stats", computed)
        self.assertIn("age", computed["numeric_stats"])
        self.assertGreater(computed["numeric_stats"]["age"]["count"], 0)
        self.assertIn("paris", computed["categorical_stats"]["city"]["vocab"])

    def test_explicit_lists_still_win(self):
        """Passing the lists directly overrides derivation from the specs."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stats = DatasetStatistics(
                path_data=str(_write_dataset(tmp_path)),
                features_specs=dict(SPECS),
                numeric_features=["age"],
                features_stats_path=str(tmp_path / "stats.json"),
                overwrite_stats=True,
            )
            self.assertEqual(stats.numeric_features, ["age"])
            self.assertEqual(stats.categorical_features, [])


class TestAutoConfigure(unittest.TestCase):
    """auto_configure is the headline "let KDP configure itself" entry point."""

    def test_returns_per_feature_recommendations(self):
        """It used to return an empty recommendation set for every dataset."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = auto_configure(
                str(_write_dataset(tmp_path)),
                features_specs=dict(SPECS),
                stats_path=str(tmp_path / "stats.json"),
                overwrite_stats=True,
            )

        self.assertIn("features", config["recommendations"])
        self.assertIn("age", config["recommendations"]["features"])
        self.assertIn(
            "FLOAT_NORMALIZED",
            config["recommendations"]["features"]["age"]["preprocessing"],
        )
        self.assertIn("global_config", config["recommendations"])
        self.assertGreater(len(config["code_snippet"]), 0)


class TestPredictAcceptsPlainPython(unittest.TestCase):
    """predict() has to take the dictionaries the docs show."""

    def test_predict_with_numpy_string_columns(self):
        """Keras rejects NumPy string arrays, so they must be converted first."""
        import tempfile
        from pathlib import Path

        keras.backend.clear_session()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            preprocessor = PreprocessingModel(
                path_data=str(_write_dataset(tmp_path)),
                features_specs=dict(SPECS),
                features_stats_path=str(tmp_path / "stats.json"),
                overwrite_stats=True,
            )
            preprocessor.build_preprocessor()

            result = preprocessor.predict(
                {
                    "age": np.array([[35.0]], dtype=np.float32),
                    "city": np.array([["paris"]]),
                    "bio": np.array([["hello world"]]),
                    "joined": np.array([["2021-06-15"]]),
                }
            )

        self.assertEqual(np.asarray(result).shape[0], 1)

    def test_build_result_is_callable_with_plain_lists(self):
        """The returned CallableDict converts plain Python values too."""
        import tempfile
        from pathlib import Path

        keras.backend.clear_session()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = PreprocessingModel(
                path_data=str(_write_dataset(tmp_path)),
                features_specs=dict(SPECS),
                features_stats_path=str(tmp_path / "stats.json"),
                overwrite_stats=True,
            ).build_preprocessor()

            output = result(
                {
                    "age": [[35.0]],
                    "city": [["paris"]],
                    "bio": [["hello world"]],
                    "joined": [["2021-06-15"]],
                }
            )

        self.assertEqual(int(tf.shape(output)[0]), 1)


if __name__ == "__main__":
    unittest.main()
