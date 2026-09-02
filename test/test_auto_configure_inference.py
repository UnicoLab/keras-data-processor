"""Tests for `auto_configure` used the way its docstring leads with.

`auto_configure("data.csv")` -- no `features_specs` -- handed `None` straight to
`DatasetStatistics`, which then had no per-type feature lists, computed nothing
and returned empty recommendations and empty statistics. Only a generic code
template came back. Feature types are now inferred from the data.

Separately, `ModelAdvisor` looked for date statistics under `date_stats` while
`DatasetStatistics` writes them under `date`, so a dataset with a date column
came back with no recommendation for that column at all.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kdp import auto_configure
from kdp.auto_config import _infer_features_specs
from kdp.features import FeatureType

ROWS = 300


def _dataset(directory):
    """One column per feature type the inference has to tell apart."""
    rng = np.random.default_rng(0)
    csv_path = directory / "data.csv"
    pd.DataFrame(
        {
            "age": rng.normal(40, 12, ROWS),
            "income": rng.lognormal(10, 1, ROWS),
            "rating": rng.integers(1, 6, ROWS),
            "city": rng.choice(["paris", "tokyo", "lima"], ROWS),
            "user_id": [f"u{i}" for i in range(ROWS)],
            "bio": rng.choice(
                ["hello world how are you today", "data science rocks the whole world"],
                ROWS,
            ),
            "joined": pd.date_range("2020-01-01", periods=ROWS).strftime("%Y-%m-%d"),
            "slashed": pd.date_range("2020-01-01", periods=ROWS).strftime("%Y/%m/%d"),
        }
    ).to_csv(csv_path, index=False)
    return csv_path


@pytest.mark.unit
class TestFeatureTypeInference(unittest.TestCase):
    """Each rule the inference documents."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.inferred = _infer_features_specs(_dataset(Path(self._tmp.name)))

    def tearDown(self):
        self._tmp.cleanup()

    def test_continuous_numbers_are_numeric(self):
        """Floats with many distinct values are quantities."""
        self.assertEqual(self.inferred["age"], FeatureType.FLOAT_NORMALIZED)
        self.assertEqual(self.inferred["income"], FeatureType.FLOAT_NORMALIZED)

    def test_small_whole_number_ranges_are_categorical(self):
        """A 1-5 rating is a code, not a magnitude."""
        self.assertEqual(self.inferred["rating"], FeatureType.INTEGER_CATEGORICAL)

    def test_repeating_labels_are_categorical(self):
        """A handful of city names."""
        self.assertEqual(self.inferred["city"], FeatureType.STRING_CATEGORICAL)

    def test_high_cardinality_single_tokens_stay_categorical(self):
        """An id column would otherwise build a vocabulary the size of the data."""
        self.assertEqual(self.inferred["user_id"], FeatureType.STRING_CATEGORICAL)

    def test_multi_word_values_are_text(self):
        """Prose gets text vectorization."""
        self.assertEqual(self.inferred["bio"], FeatureType.TEXT)

    def test_both_date_layouts_are_detected(self):
        """`YYYY-MM-DD` and `YYYY/MM/DD` are the two KDP parses."""
        self.assertEqual(self.inferred["joined"], FeatureType.DATE)
        self.assertEqual(self.inferred["slashed"], FeatureType.DATE)

    def test_every_column_is_classified(self):
        """A column with no type would silently vanish from the model."""
        self.assertEqual(len(self.inferred), 8)


@pytest.mark.integration
class TestAutoConfigureWithoutSpecs(unittest.TestCase):
    """The headline call: point it at a CSV and nothing else."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmp.name)
        self.config = auto_configure(
            str(_dataset(tmp_path)),
            stats_path=str(tmp_path / "stats.json"),
            overwrite_stats=True,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_recommendations_are_not_empty(self):
        """They were `{}` for every dataset."""
        self.assertTrue(self.config["recommendations"]["features"])

    def test_statistics_are_not_empty(self):
        """Nothing was computed, so nothing could be recommended."""
        self.assertTrue(self.config["statistics"])

    def test_every_column_gets_a_recommendation(self):
        """Including the date column, which was dropped by a key mismatch."""
        features = self.config["recommendations"]["features"]
        for column in ("age", "rating", "city", "user_id", "bio", "joined"):
            self.assertIn(column, features, f"no recommendation for {column}")

    def test_the_date_recommendation_names_real_options(self):
        """It used to suggest `extract` and `cyclical_encoding`, which do not exist."""
        date_rec = self.config["recommendations"]["features"]["joined"]
        self.assertEqual(date_rec["feature_type"], "DateFeature")
        self.assertNotIn("extract", date_rec["config"])
        self.assertNotIn("cyclical_encoding", date_rec["advanced_options"])
        # Only `format` and `add_season` are real.
        self.assertTrue(set(date_rec["config"]) <= {"format", "add_season"})

    def test_the_generated_snippet_executes(self):
        """It is presented as ready to use, so it has to run."""
        namespace = {}
        exec(  # noqa: S102 - executing the snippet is the point of the test
            compile(self.config["code_snippet"], "<snippet>", "exec"), namespace
        )
        self.assertIn("features", namespace)


if __name__ == "__main__":
    unittest.main()
