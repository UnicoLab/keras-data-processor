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


class TestLegacyStatsFileDetection(unittest.TestCase):
    """Statistics written before the bytes fix must not be trusted."""

    def test_repr_encoded_vocabulary_is_detected(self):
        """`b'paris'` is a byte repr, not a category."""
        self.assertTrue(
            DatasetStatistics._has_repr_encoded_vocabulary(
                {"categorical_stats": {"city": {"vocab": ["b'paris'", "b'lima'"]}}}
            )
        )

    def test_correct_vocabulary_is_left_alone(self):
        """A properly decoded vocabulary is not mistaken for the old format."""
        self.assertFalse(
            DatasetStatistics._has_repr_encoded_vocabulary(
                {"categorical_stats": {"city": {"vocab": ["paris", "lima"]}}}
            )
        )

    def test_single_letter_b_category_is_not_a_false_positive(self):
        """A category that merely starts with "b" is still a category."""
        self.assertFalse(
            DatasetStatistics._has_repr_encoded_vocabulary(
                {"categorical_stats": {"c": {"vocab": ["b", "berlin", "b'"]}}}
            )
        )

    def test_legacy_file_is_recomputed_rather_than_reused(self):
        """Reusing it would map every category to the OOV slot."""
        import json
        import tempfile
        from pathlib import Path

        keras.backend.clear_session()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = _write_dataset(tmp_path)
            stats_path = tmp_path / "features_stats.json"
            # Exactly what the previous release wrote.
            stats_path.write_text(
                json.dumps(
                    {
                        "categorical_stats": {
                            "city": {
                                "size": 3,
                                "vocab": ["b'paris'", "b'tokyo'", "b'lima'"],
                                "dtype": "string",
                            }
                        }
                    }
                )
            )

            preprocessor = PreprocessingModel(
                path_data=str(csv_path),
                features_specs={"city": FeatureType.STRING_CATEGORICAL},
                features_stats_path=str(stats_path),
            )
            preprocessor.build_preprocessor()

            output = np.asarray(
                preprocessor.model(
                    {"city": tf.constant([["paris"], ["tokyo"], ["UNSEEN"]])}
                )
            )

        # Known categories must not collapse onto the out-of-vocabulary vector.
        self.assertFalse(np.allclose(output[0], output[2]))
        self.assertFalse(np.allclose(output[0], output[1]))


class TestStatisticsAreOnlyComputedWhenNeeded(unittest.TestCase):
    """A model whose features learn nothing from the data needs no dataset."""

    def test_hashing_only_model_builds_without_any_data(self):
        """Hashing sizes its own buckets, so no statistics pass is required."""
        from kdp.features import CategoricalFeature, CategoryEncodingOptions

        keras.backend.clear_session()
        preprocessor = PreprocessingModel(
            features_specs={
                "city": CategoricalFeature(
                    name="city",
                    feature_type=FeatureType.STRING_CATEGORICAL,
                    category_encoding=CategoryEncodingOptions.HASHING,
                    hash_bucket_size=32,
                ),
            },
        )
        preprocessor.build_preprocessor()
        output = preprocessor.model({"city": tf.constant([["paris"], ["lima"]])})
        self.assertEqual(tuple(output.shape), (2, 32))

    def test_numeric_feature_without_data_raises_a_clear_error(self):
        """The old failure was a pathlib TypeError from deep inside the stats pass."""
        keras.backend.clear_session()
        with self.assertRaises(ValueError) as ctx:
            PreprocessingModel(
                features_specs={"age": FeatureType.FLOAT_NORMALIZED},
            ).build_preprocessor()
        self.assertIn("path_data", str(ctx.exception))

    def test_hashing_without_explicit_bucket_size_still_needs_statistics(self):
        """Without a bucket count the size is derived from the vocabulary."""
        from kdp.features import CategoricalFeature, CategoryEncodingOptions

        keras.backend.clear_session()
        with self.assertRaises(ValueError):
            PreprocessingModel(
                features_specs={
                    "city": CategoricalFeature(
                        name="city",
                        feature_type=FeatureType.STRING_CATEGORICAL,
                        category_encoding=CategoryEncodingOptions.HASHING,
                    ),
                },
            ).build_preprocessor()


if __name__ == "__main__":
    unittest.main()
