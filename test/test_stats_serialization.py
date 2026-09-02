"""Tests for how statistics are written to and read back from JSON.

`_custom_serializer` decides what a statistics file contains. Its bytes branch
is the one that mattered: `str(b"paris")` yields `"b'paris'"`, and a vocabulary
saved that way loads without complaint and maps every category to the
out-of-vocabulary slot. Each branch is checked here, along with a full
save/load round trip.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kdp import DatasetStatistics
from kdp.features import FeatureType


@pytest.mark.unit
class TestCustomSerializer(unittest.TestCase):
    """One case per type the serializer knows about."""

    def _serialize(self, value):
        return DatasetStatistics._custom_serializer(value)

    def test_a_tensorflow_dtype_becomes_its_name(self):
        """`tf.string` has to survive as something JSON can hold."""
        self.assertEqual(self._serialize(tf.string), "string")
        self.assertEqual(self._serialize(tf.float32), "float32")

    def test_numpy_integers_become_python_ints(self):
        """`json` cannot encode a numpy scalar."""
        result = self._serialize(np.int64(7))
        self.assertIsInstance(result, int)
        self.assertEqual(result, 7)

    def test_numpy_floats_become_python_floats(self):
        """Same for the float family."""
        result = self._serialize(np.float32(2.5))
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 2.5)

    def test_bytes_are_decoded_not_repr_ed(self):
        """`str(b"paris")` gives "b'paris'", which corrupts a vocabulary."""
        self.assertEqual(self._serialize(b"paris"), "paris")
        self.assertNotIn("b'", self._serialize(b"paris"))

    def test_arrays_become_lists(self):
        """A vocabulary often arrives as an ndarray."""
        self.assertEqual(self._serialize(np.array([1, 2, 3])), [1, 2, 3])

    def test_an_unknown_type_is_refused(self):
        """Better to fail than to write something that will not load."""
        with self.assertRaises(TypeError):
            self._serialize(object())


@pytest.mark.integration
class TestStatisticsRoundTrip(unittest.TestCase):
    """What is written must be what comes back."""

    def _dataset(self, directory, rows: int = 150):
        rng = np.random.default_rng(3)
        csv_path = directory / "data.csv"
        pd.DataFrame(
            {
                "age": rng.normal(40, 10, rows),
                "city": rng.choice(["paris", "tokyo", "lima"], rows),
            }
        ).to_csv(csv_path, index=False)
        return csv_path

    def test_the_saved_file_holds_decoded_categories(self):
        """The bytes defect showed up only once the file was read back."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stats_path = tmp_path / "stats.json"
            DatasetStatistics(
                path_data=str(self._dataset(tmp_path)),
                features_specs={
                    "age": FeatureType.FLOAT_NORMALIZED,
                    "city": FeatureType.STRING_CATEGORICAL,
                },
                features_stats_path=str(stats_path),
                overwrite_stats=True,
            ).main()

            written = json.loads(stats_path.read_text())

        vocabulary = written["categorical_stats"]["city"]["vocab"]
        self.assertIn("paris", vocabulary)
        for entry in vocabulary:
            self.assertFalse(
                entry.startswith("b'"), f"{entry} was serialized as a byte repr"
            )

    def test_reloading_reuses_the_saved_statistics(self):
        """A second run with the same path must not recompute from scratch."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = self._dataset(tmp_path)
            stats_path = tmp_path / "stats.json"
            specs = {
                "age": FeatureType.FLOAT_NORMALIZED,
                "city": FeatureType.STRING_CATEGORICAL,
            }
            first = DatasetStatistics(
                path_data=str(csv_path),
                features_specs=dict(specs),
                features_stats_path=str(stats_path),
                overwrite_stats=True,
            ).main()

            second = DatasetStatistics(
                path_data=str(csv_path),
                features_specs=dict(specs),
                features_stats_path=str(stats_path),
            ).main()

        self.assertEqual(
            sorted(first["categorical_stats"]["city"]["vocab"]),
            sorted(second["categorical_stats"]["city"]["vocab"]),
        )
        self.assertAlmostEqual(
            first["numeric_stats"]["age"]["mean"],
            second["numeric_stats"]["age"]["mean"],
            places=5,
        )

    def test_numeric_statistics_match_the_data(self):
        """The numbers themselves, not just their presence."""
        rng = np.random.default_rng(3)
        rows = 150
        values = rng.normal(40, 10, rows)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "data.csv"
            pd.DataFrame(
                {"age": values, "city": rng.choice(["paris", "lima"], rows)}
            ).to_csv(csv_path, index=False)

            computed = DatasetStatistics(
                path_data=str(csv_path),
                features_specs={
                    "age": FeatureType.FLOAT_NORMALIZED,
                    "city": FeatureType.STRING_CATEGORICAL,
                },
                features_stats_path=str(tmp_path / "stats.json"),
                overwrite_stats=True,
            ).main()

        age = computed["numeric_stats"]["age"]
        self.assertAlmostEqual(age["mean"], float(values.mean()), places=3)
        # KDP reports the sample variance (ddof=1), which is the estimate you
        # want from a sample; numpy's `.var()` defaults to the population one.
        self.assertAlmostEqual(age["var"], float(values.var(ddof=1)), places=2)
        self.assertEqual(age["count"], rows)


if __name__ == "__main__":
    unittest.main()
