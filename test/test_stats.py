import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from kdp.stats import (
    CategoricalAccumulator,
    CategoricalFeature,
    DatasetStatistics,
    FeatureType,
    NumericalFeature,
    WelfordAccumulator,
)


class TestWelfordAccumulator(unittest.TestCase):
    """Unit tests for the WelfordAccumulator class."""

    def setUp(self) -> None:
        """Set up test cases for WelfordAccumulator."""
        self.accumulator = WelfordAccumulator()

    def test_initial_state(self):
        """Ensure initial state is correctly set."""
        self.assertEqual(self.accumulator.n.numpy(), 0.0)
        self.assertEqual(self.accumulator.mean.numpy(), 0.0)
        self.assertEqual(self.accumulator.M2.numpy(), 0.0)

    def test_update_single_value(self):
        """Test updating the accumulator with a single value."""
        self.accumulator.update(tf.constant([5.0]))
        self.assertEqual(self.accumulator.count.numpy(), 1)
        self.assertEqual(self.accumulator.mean.numpy(), 5.0)

    def test_update_multiple_values(self):
        """Test updating the accumulator with multiple values."""
        values = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
        self.accumulator.update(values)
        self.assertAlmostEqual(self.accumulator.mean.numpy(), 3.0, places=5)
        self.assertEqual(self.accumulator.count.numpy(), 5)
        self.assertAlmostEqual(self.accumulator.variance.numpy(), 2.5, places=5)

    def test_variance_n_equals_1(self):
        """Verify that variance is 0 when n equals 1."""
        self.accumulator.update(tf.constant([5.0]))
        self.assertEqual(self.accumulator.variance, 0.0)


class TestCategoricalAccumulator(unittest.TestCase):
    """Unit tests for the CategoricalAccumulator class."""

    def setUp(self) -> None:
        """Set up test cases for CategoricalAccumulator."""
        self.accumulator = CategoricalAccumulator()

    def test_initial_state(self):
        """Ensure initial state is correctly set."""
        self.assertEqual(len(self.accumulator.values.numpy()), 0)
        self.assertEqual(len(self.accumulator.int_values.numpy()), 0)

    def test_update_string_values(self):
        """Test updating the accumulator with string values."""
        self.accumulator.update(tf.constant(["apple", "banana"]))
        self.assertIn(
            "apple",
            [_bytes.decode("utf-8") for _bytes in self.accumulator.get_unique_values()],
        )
        self.assertIn(
            "banana",
            [_bytes.decode("utf-8") for _bytes in self.accumulator.get_unique_values()],
        )

    def test_update_int_values(self):
        """Test updating the accumulator with integer values."""
        self.accumulator.update(tf.constant([1, 2, 2]))
        unique_values = [int(_el) for _el in self.accumulator.get_unique_values()]
        self.assertIn(1, unique_values)
        self.assertIn(2, unique_values)
        self.assertEqual(len(unique_values), 2)

    def test_update_unsupported_dtype(self):
        """Test updating the accumulator with an unsupported data type."""
        with self.assertRaises(ValueError):
            self.accumulator.update(tf.constant([1.0, 2.0], dtype=tf.float32))


# class TestDatasetStatistics(unittest.TestCase):
#     """Unit tests for the DatasetStatistics class."""

#     def setUp(self) -> None:
#         """Set up test cases for DatasetStatistics."""
#         self.path_data = "path/to/dataset"
#         self.batch_size = 2
#         self.stats = DatasetStatistics(
#             path_data=self.path_data,
#             numeric_cols=["num_feature"],
#             categorical_cols=["cat_feature"],
#             batch_size=self.batch_size,
#         )

#     @patch("pathlib.Path.open")
#     @patch("json.dump")
#     def test_save_stats(self, mock_json_dump, mock_open):
#         """Test saving feature stats to a file."""
#         self.stats.features_stats = {
#             "numeric_stats": {
#                 "feat_a": {"mean": 0.46489500999450684, "count": 10.0, "var": 1.3447670936584473},
#             },
#             "categorical_stats": {
#                 "feat_b": {"size": 5, "vocab": [0, 1, 2, 3, 4]},
#             },
#         }
#         self.stats._save_stats()
#         mock_open.assert_called_once()
#         mock_json_dump.assert_called_once()

#     @patch("pathlib.Path.is_file", return_value=True)
#     @patch("pathlib.Path.open", new_callable=MagicMock)
#     @patch("json.load")
#     def test_load_stats_file_exists(self, mock_json_load, mock_open, mock_is_file):
#         """Test loading stats from an existing file."""
#         self.stats._load_stats()
#         mock_open.assert_called_once()
#         mock_json_load.assert_called_once()

#     @patch("pathlib.Path.is_file", return_value=False)
#     def test_load_stats_file_not_exists(self, mock_is_file):
#         """Test behavior when stats file does not exist."""
#         self.stats.features_stats = {}
#         loaded_stats = self.stats._load_stats()
#         self.assertEqual(loaded_stats, {})

#     # Continuing from the previous TestDatasetStatistics class...

#     def test_parse_features_specs(self):
#         """Test parsing of feature specifications."""
#         features_specs = {
#             "num_feature": FeatureType.FLOAT.value,
#             "cat_feature_int": FeatureType.INTEGER_CATEGORICAL.value,
#             "cat_feature_str": FeatureType.STRING_CATEGORICAL.value,
#         }
#         stats = DatasetStatistics(
#             path_data=self.path_data,
#             features_specs=features_specs,
#             batch_size=self.batch_size,
#         )
#         stats._parse_features_specs()
#         self.assertIn("num_feature", stats.numeric_cols)
#         self.assertIn("cat_feature_int", stats.categorical_cols)
#         self.assertIn("cat_feature_str", stats.categorical_cols)

#     @patch("tensorflow.data.experimental.make_csv_dataset")
#     def test_read_data_into_dataset(self, mock_make_csv_dataset):
#         """Test reading data into a tf.data.Dataset."""
#         mock_make_csv_dataset.return_value = tf.data.Dataset.from_tensor_slices(
#             {"num_feature": tf.constant([1.0, 2.0]), "cat_feature": tf.constant(["apple", "banana"])}
#         )
#         dataset = self.stats._read_data_into_dataset()
#         self.assertIsInstance(dataset, tf.data.Dataset)

#     @patch("tensorflow.data.Dataset.take")
#     def test_infer_feature_dtypes(self, mock_dataset_take):
#         """Test inferring feature data types from a dataset."""
#         batch = {
#             "num_feature": tf.constant([1.0, 2.0], dtype=tf.float32),
#             "cat_feature": tf.constant(["apple", "banana"], dtype=tf.string),
#         }
#         mock_dataset_take.return_value = iter([batch])
#         self.stats.categorical_cols = ["cat_feature"]
#         self.stats.numeric_cols = ["num_feature"]
#         self.stats._infer_feature_dtypes(tf.data.Dataset.from_tensor_slices(batch))
#         self.assertEqual(self.stats.features_dtypes["num_feature"], tf.float32)
#         self.assertEqual(self.stats.features_dtypes["cat_feature"], tf.string)

#     def test_process_batch(self):
#         """Test processing a single batch of data."""
#         batch = {
#             "num_feature": tf.constant([1.0, 2.0], dtype=tf.float32),
#             "cat_feature": tf.constant(["apple", "banana"], dtype=tf.string),
#         }
#         self.stats.numeric_cols = ["num_feature"]
#         self.stats.categorical_cols = ["cat_feature"]
#         self.stats.numeric_stats = {"num_feature": WelfordAccumulator()}
#         self.stats.categorical_stats = {"cat_feature": CategoricalAccumulator()}
#         self.stats._process_batch(batch)
#         self.assertNotEqual(self.stats.numeric_stats["num_feature"].count.numpy(), 0)
#         self.assertNotEqual(len(self.stats.categorical_stats["cat_feature"].get_unique_values()), 0)

#     def test_compute_final_statistics(self):
#         """Test computation of final statistics."""
#         # Simulate data processing
#         self.stats.numeric_stats = {"num_feature": WelfordAccumulator()}
#         self.stats.categorical_stats = {"cat_feature": CategoricalAccumulator()}
#         self.stats.numeric_stats["num_feature"].update(tf.constant([1.0, 2.0]))
#         self.stats.categorical_stats["cat_feature"].update(tf.constant(["apple", "banana"]))

#         final_stats = self.stats._compute_final_statistics()
#         self.assertIn("num_feature", final_stats["numeric_stats"])
#         self.assertIn("cat_feature", final_stats["categorical_stats"])
#         self.assertEqual(final_stats["numeric_stats"]["num_feature"]["count"], 2)
#         self.assertIn("apple", final_stats["categorical_stats"]["cat_feature"]["vocab"])


class TestDatasetStatisticsPandasComparison(unittest.TestCase):
    @staticmethod
    def generate_fake_data(features_scope: dict, num_rows: int = 10) -> pd.DataFrame:
        """Generate a dummy dataset with a given number of rows.

        Args:
            features_scope: A dictionary with the features and their types.
            num_rows: The number of rows to generate.
        """
        data = {}
        for feature, spec in features_scope.items():
            _feat_type = spec.feature_type
            if _feat_type in {FeatureType.FLOAT, FeatureType.FLOAT.value}:
                data[feature] = np.random.randn(
                    num_rows
                )  # Normal distribution for floats
            elif _feat_type in {
                FeatureType.INTEGER_CATEGORICAL,
                FeatureType.INTEGER_CATEGORICAL.value,
            }:
                data[feature] = np.random.randint(
                    0, 5, size=num_rows
                )  # Integer categories from 0 to 4
            elif _feat_type in {
                FeatureType.STRING_CATEGORICAL,
                FeatureType.STRING_CATEGORICAL.value,
            }:
                categories = ["cat", "dog", "fish", "bird"]
                data[feature] = np.random.choice(
                    categories, size=num_rows
                )  # Randomly chosen string categories
        # Create the DataFrame
        df = pd.DataFrame(data)
        return df

    @classmethod
    def setUpClass(cls):
        # create the temp file in setUp method if you want a fresh directory for each test.
        # This is useful if you don't want to share state between tests.
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_file = Path(cls.temp_dir.name)

        # prepare the PATH_LOCAL_TRAIN_DATA
        cls._path_data = Path("data/rawdata.csv")
        cls._path_data = cls.temp_file / cls._path_data
        cls._path_data.parent.mkdir(exist_ok=True, parents=True)

        # this part is handled by the preprocessor to normalize the stored and passed features
        # so when it arrives to the stats module, it is already normalized this way
        cls.features_scope = {
            "feat_a": NumericalFeature(name="feat_a", feature_type=FeatureType.FLOAT),
            "feat_b": CategoricalFeature(
                name="feat_b", feature_type=FeatureType.INTEGER_CATEGORICAL
            ),
            "feat_c": CategoricalFeature(
                name="feat_c", feature_type=FeatureType.STRING_CATEGORICAL
            ),
        }
        cls.numeric_features = ["feat_a"]
        cls.categorical_features = ["feat_b", "feat_c"]
        # building fake data
        cls.df = cls.generate_fake_data(
            features_scope=cls.features_scope,
            num_rows=10,
        )
        cls.df.to_csv(cls._path_data, index=False)

        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        # Remove the temporary file after the test is done
        cls.temp_dir.cleanup()

    def test_stats_pandas_vs_tf(self):
        _data_stats = DatasetStatistics(
            path_data=self._path_data,
            features_specs=self.features_scope,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            # Without this the stats land in ./features_stats.json, and a later
            # test that asserts a missing-data error instead finds that file
            # and passes silently.
            features_stats_path=str(Path(self.temp_dir.name) / "features_stats.json"),
        )
        stats = _data_stats.main()
        print(f"{stats=}")
        # numerical stats comparison
        self.assertAlmostEqual(
            self.df["feat_a"].mean(), stats["numeric_stats"]["feat_a"]["mean"], places=5
        )
        self.assertAlmostEqual(
            self.df["feat_a"].var(), stats["numeric_stats"]["feat_a"]["var"], places=5
        )
        # categorical data comparison
        self.assertEqual(
            self.df["feat_b"].nunique(), stats["categorical_stats"]["feat_b"]["size"]
        )
        self.assertEqual(
            list(np.unique(self.df["feat_b"])),
            stats["categorical_stats"]["feat_b"]["vocab"],
        )


# ============= NEW TESTS ==============
# class TestDatasetStatistics(unittest.TestCase):
#     def setUp(self):
#         self.path_data = "dummy/path"
#         self.features_stats_path = "dummy/stats/path"
#         self.dataset_statistics = DatasetStatistics(
#             path_data=self.path_data,
#             features_stats_path=self.features_stats_path
#         )

#     @patch("kdp.stats.DatasetStatistics.Path.open", new_callable=MagicMock)
#     @patch("kdp.stats.DatasetStatistics.json.dump")
#     def test_save_stats(self, mock_json_dump, mock_open):
#         self.dataset_statistics.features_stats = {
#             "numeric_stats": {"num_feature": {"mean": 0.5, "var": 0.25, "count": 2, "dtype": tf.float32}},
#             "categorical_stats": {"cat_feature": {"size": 2, "vocab": ["a", "b"], "dtype": tf.string}}
#         }
#         self.dataset_statistics._save_stats()
#         mock_open.assert_called_once()
#         mock_json_dump.assert_called_once()

#     @patch("kdp.stats.DatasetStatistics.Path.is_file", return_value=True)
#     @patch("kdp.stats.DatasetStatistics.Path.open", new_callable=MagicMock)
#     @patch("kdp.stats.DatasetStatistics.json.load")
#     def test_load_stats_file_exists(self, mock_json_load, mock_open, mock_is_file):
#         self.dataset_statistics._load_stats()
#         mock_open.assert_called_once()
#         mock_json_load.assert_called_once()

#     @patch("kdp.stats.DatasetStatistics.Path.is_file", return_value=False)
#     def test_load_stats_file_not_exists(self, mock_is_file):
#         loaded_stats = self.dataset_statistics._load_stats()
#         self.assertEqual(loaded_stats, {})

#     @patch("kdp.stats.DatasetStatistics.tf.data.experimental.make_csv_dataset")
#     def test_read_data_into_dataset(self, mock_make_csv_dataset):
#         mock_make_csv_dataset.return_value = tf.data.Dataset.from_tensor_slices({"num_feature": tf.constant([1.0, 2.0]), "cat_feature": tf.constant(["apple", "banana"])})
#         dataset = self.dataset_statistics._read_data_into_dataset()
#         self.assertIsInstance(dataset, tf.data.Dataset)

#     def test_process_batch(self):
#         batch = {"num_feature": tf.constant([1.0, 2.0], dtype=tf.float32), "cat_feature": tf.constant(["apple", "banana"], dtype=tf.string)}
#         self.dataset_statistics.numeric_features = ["num_feature"]
#         self.dataset_statistics.categorical_features = ["cat_feature"]
#         self.dataset_statistics.numeric_stats = {"num_feature": WelfordAccumulator()}
#         self.dataset_statistics.categorical_stats = {"cat_feature": CategoricalAccumulator()}
#         self.dataset_statistics._process_batch(batch)
#         # Ensure accumulators are called or updated accordingly. This requires the WelfordAccumulator and CategoricalAccumulator classes to support these operations.

#     def test_compute_final_statistics(self):
#         # Mock or simulate the statistical accumulators' state after processing some batches.
#         self.dataset_statistics.numeric_features = ["num_feature"]
#         self.dataset_statistics.categorical_features = ["cat_feature"]
#         self.dataset_statistics.numeric_stats = {"num_feature": WelfordAccumulator()}
#         self.dataset_statistics.categorical_stats = {"cat_feature": CategoricalAccumulator()}
#         # Simulate updates for simplicity. In practice, these would result from _process_batch calls.
#         self.dataset_statistics.numeric_stats["num_feature"].update(tf.constant([1.0, 2.0]))
#         self.dataset_statistics.categorical_stats["cat_feature"].update(tf.constant(["apple", "banana"]))
#         final_stats = self.dataset_statistics._compute_final_statistics()
#         # Assertions based on the expected final statistics structure and content.

if __name__ == "__main__":
    unittest.main()
