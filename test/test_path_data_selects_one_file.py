"""`path_data` pointing at a file has to mean that file.

The path was replaced with `<parent>/*.csv`, so naming one CSV computed the
statistics over every CSV beside it. Every example in the documentation --
`path_data="data/my_data.csv"` in the README included -- passes a single file,
and a project laid out as `data/train.csv` and `data/test.csv` therefore had its
test set folded into the training statistics. Nothing raised: the numbers came
back, drawn from the wrong rows.
"""

import json
import tempfile
import unittest
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kdp import FeatureType, PreprocessingModel
from kdp.stats import DatasetStatistics

ROWS = 400


def _write(directory, name, mean):
    """A CSV whose column sits at a known, distinctive mean."""
    frame = pd.DataFrame(
        {"x": np.random.default_rng(abs(hash(name)) % 2**32).normal(mean, 1.0, ROWS)},
    )
    path = Path(directory) / name
    frame.to_csv(path, index=False)
    return path, float(frame["x"].mean())


@pytest.mark.unit
class TestPathDataSelectsOneFile(unittest.TestCase):
    """What a file, a directory and a glob each select."""

    def test_naming_a_file_ignores_its_neighbours(self):
        with tempfile.TemporaryDirectory() as directory:
            train, train_mean = _write(directory, "train.csv", mean=10.0)
            _write(directory, "test.csv", mean=1000.0)

            stats_path = Path(directory) / "stats.json"
            keras.backend.clear_session()
            preprocessor = PreprocessingModel(
                path_data=str(train),
                features_specs={"x": FeatureType.FLOAT_NORMALIZED},
                features_stats_path=str(stats_path),
                overwrite_stats=True,
            )
            preprocessor.build_preprocessor()
            stats = json.loads(stats_path.read_text())["numeric_stats"]["x"]

        # Only train.csv: its own row count and its own mean, nowhere near the
        # 505 that averaging the two files would give.
        self.assertEqual(int(stats["count"]), ROWS)
        self.assertAlmostEqual(float(stats["mean"]), train_mean, places=3)

    def test_naming_a_directory_reads_every_csv_in_it(self):
        with tempfile.TemporaryDirectory() as directory:
            _write(directory, "a.csv", mean=10.0)
            _write(directory, "b.csv", mean=20.0)

            stats_path = Path(directory) / "stats.json"
            keras.backend.clear_session()
            preprocessor = PreprocessingModel(
                path_data=directory,
                features_specs={"x": FeatureType.FLOAT_NORMALIZED},
                features_stats_path=str(stats_path),
                overwrite_stats=True,
            )
            preprocessor.build_preprocessor()
            stats = json.loads(stats_path.read_text())["numeric_stats"]["x"]

        self.assertEqual(int(stats["count"]), 2 * ROWS)
        self.assertAlmostEqual(float(stats["mean"]), 15.0, places=1)

    def test_the_three_shapes_a_path_can_take(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = _write(directory, "one.csv", mean=5.0)
            self.assertEqual(
                DatasetStatistics._get_csv_file_pattern(str(path)),
                str(path),
            )
            self.assertEqual(
                DatasetStatistics._get_csv_file_pattern(directory),
                str(Path(directory) / "*.csv"),
            )
            glob = str(Path(directory) / "*.csv")
            self.assertEqual(DatasetStatistics._get_csv_file_pattern(glob), glob)

    def test_a_path_that_does_not_exist_is_reported(self):
        """`tf.data` answers a missing file with an empty dataset, and the
        statistics then come from nothing at all."""
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "absent.csv")
            with self.assertRaises(FileNotFoundError) as caught:
                DatasetStatistics._get_csv_file_pattern(missing)
            self.assertIn("absent.csv", str(caught.exception))

    def test_two_models_built_in_one_process_keep_their_own_statistics(self):
        """The symptom this surfaced as: the second model's numbers were the
        two datasets pooled together."""
        with tempfile.TemporaryDirectory() as directory:
            first, first_mean = _write(directory, "first.csv", mean=10.0)
            second, second_mean = _write(directory, "second.csv", mean=900.0)

            seen = []
            for index, (path, expected) in enumerate(
                ((first, first_mean), (second, second_mean)),
            ):
                stats_path = Path(directory) / f"stats{index}.json"
                keras.backend.clear_session()
                preprocessor = PreprocessingModel(
                    path_data=str(path),
                    features_specs={"x": FeatureType.FLOAT_NORMALIZED},
                    features_stats_path=str(stats_path),
                    overwrite_stats=True,
                )
                preprocessor.build_preprocessor()
                stats = json.loads(stats_path.read_text())["numeric_stats"]["x"]
                seen.append((int(stats["count"]), float(stats["mean"]), expected))

            for count, mean, expected in seen:
                self.assertEqual(count, ROWS)
                self.assertAlmostEqual(mean, expected, places=3)


@pytest.mark.unit
class TestLargeMagnitudeColumnsSurviveNormalisation(unittest.TestCase):
    """IDs, epoch timestamps and amounts in cents all live far from zero."""

    COLUMNS = {
        "ordinary": (50.0, 12.0),
        "large ids": (1e8, 1e3),
        "amounts in cents": (1e7, 5e2),
        "epoch seconds": (1.7e9, 1e6),
    }

    def test_a_column_far_from_zero_still_standardises(self):
        rng = np.random.default_rng(1)
        for label, (centre, spread) in self.COLUMNS.items():
            with self.subTest(column=label), tempfile.TemporaryDirectory() as directory:
                column = rng.normal(centre, spread, 2000)
                path = Path(directory) / "d.csv"
                pd.DataFrame({"x": column}).to_csv(path, index=False)

                keras.backend.clear_session()
                preprocessor = PreprocessingModel(
                    path_data=str(path),
                    features_specs={"x": FeatureType.FLOAT_NORMALIZED},
                    features_stats_path=str(Path(directory) / "s.json"),
                    overwrite_stats=True,
                )
                preprocessor.build_preprocessor()
                output = np.asarray(
                    preprocessor.model(
                        {"x": tf.constant(column.reshape(-1, 1), dtype=tf.float32)},
                        training=False,
                    ),
                )

                self.assertFalse(np.isnan(output).any(), label)
                # The whole point of FLOAT_NORMALIZED: zero mean, unit variance.
                # These columns used to come out as a constant.
                self.assertLess(abs(float(output.mean())), 0.05, label)
                self.assertAlmostEqual(float(output.std()), 1.0, delta=0.05, msg=label)
