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


@pytest.mark.unit
class TestInferenceEdgeCases(unittest.TestCase):
    """The paths a real dataset hits that a tidy one does not."""

    def test_a_directory_of_csvs_is_accepted(self):
        """`path_data` may be a directory, as the statistics pass allows."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "shards"
            directory.mkdir()
            rng = np.random.default_rng(0)
            for shard in range(2):
                pd.DataFrame(
                    {"age": rng.normal(40, 10, 50), "city": rng.choice(["a", "b"], 50)}
                ).to_csv(directory / f"part_{shard}.csv", index=False)

            inferred = _infer_features_specs(directory)

        self.assertEqual(inferred["age"], FeatureType.FLOAT_NORMALIZED)
        self.assertEqual(inferred["city"], FeatureType.STRING_CATEGORICAL)

    def test_an_empty_directory_is_reported_clearly(self):
        """Better than a confusing error from deep inside pandas."""
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError) as ctx:
            _infer_features_specs(Path(tmp))
        self.assertIn("No CSV files", str(ctx.exception))

    def test_an_all_empty_column_defaults_to_numeric(self):
        """There is no evidence either way, so it must not crash."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "gaps.csv"
            pd.DataFrame(
                {"present": [1.5, 2.5, 3.5], "absent": [None, None, None]}
            ).to_csv(csv_path, index=False)
            inferred = _infer_features_specs(csv_path)

        self.assertEqual(inferred["absent"], FeatureType.FLOAT_NORMALIZED)
        self.assertEqual(inferred["present"], FeatureType.FLOAT_NORMALIZED)

    def test_a_column_with_gaps_is_still_classified(self):
        """Missing values must not decide the type."""
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "partial.csv"
            scores = rng.normal(50, 10, 60).tolist()
            scores[5] = scores[17] = None
            cities = rng.choice(["paris", "lima"], 60).tolist()
            cities[3] = None
            pd.DataFrame({"city": cities, "score": scores}).to_csv(
                csv_path, index=False
            )
            inferred = _infer_features_specs(csv_path)

        self.assertEqual(inferred["city"], FeatureType.STRING_CATEGORICAL)
        self.assertEqual(inferred["score"], FeatureType.FLOAT_NORMALIZED)

    def test_a_float_column_of_few_whole_values_reads_as_a_code(self):
        """`1.0, 3.0, 4.0` is a rating stored as float, not a measurement.

        The rule looks at whether the values are whole and how many distinct
        ones there are, not at the column's dtype, so a small integer range
        written as floats is still treated as categorical.
        """
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "codes.csv"
            pd.DataFrame({"score": [1.0, 3.0, 4.0, 1.0, 3.0]}).to_csv(
                csv_path, index=False
            )
            inferred = _infer_features_specs(csv_path)

        self.assertEqual(inferred["score"], FeatureType.INTEGER_CATEGORICAL)

    def test_explicit_specs_are_used_unchanged(self):
        """Inference must not override what the caller asked for."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = _dataset(tmp_path)
            config = auto_configure(
                str(csv_path),
                features_specs={"age": FeatureType.FLOAT_RESCALED},
                stats_path=str(tmp_path / "stats.json"),
                overwrite_stats=True,
            )
        # Only the column the caller named is analysed.
        self.assertEqual(list(config["recommendations"]["features"]), ["age"])


@pytest.mark.integration
class TestGeneratedSnippetUsesRealOptions(unittest.TestCase):
    """Every keyword the snippet writes must be one its class actually reads.

    The snippet is presented as ready to use, so a keyword the feature ignores
    is worse than an error: the reader believes the setting took effect. It
    emitted `embedding_dim` on `TextFeature`, which forwards to
    `TextVectorization` and has no such argument, and `date_format`,
    `output_format` and `extract` on `DateFeature`, which reads only `format`
    and `add_season`.
    """

    # What each class genuinely honours, established by running the model.
    ALLOWED = {
        "NumericalFeature": {
            "name",
            "feature_type",
            "preferred_distribution",
            "prefered_distribution",
            "use_embedding",
            "embedding_dim",
            "num_bins",
            "bin_boundaries",
            "scale",
        },
        "CategoricalFeature": {
            "name",
            "feature_type",
            "category_encoding",
            "embedding_size",
            "hash_bucket_size",
            "salt",
            "hash_with_embedding",
        },
        "TextFeature": {
            "name",
            "feature_type",
            "stop_words",
            "max_tokens",
            "output_sequence_length",
            "output_mode",
            "ngrams",
            "split",
            "standardize",
        },
        "DateFeature": {"name", "feature_type", "format", "add_season"},
    }

    def _snippet_for(self, directory):
        rng = np.random.default_rng(0)
        rows = 400
        csv_path = directory / "varied.csv"
        pd.DataFrame(
            {
                "amount": rng.lognormal(3, 1, rows),
                "score": rng.normal(0, 1, rows),
                "sparse": np.where(rng.random(rows) < 0.8, 0.0, rng.normal(5, 1, rows)),
                "small_cat": rng.choice(list("abc"), rows),
                "big_cat": rng.choice([f"v{i}" for i in range(150)], rows),
                "note": rng.choice(
                    ["short text here", "a much longer piece of prose about things"],
                    rows,
                ),
                "when": pd.date_range("2020-01-01", periods=rows).strftime("%Y-%m-%d"),
            }
        ).to_csv(csv_path, index=False)
        return auto_configure(
            str(csv_path),
            stats_path=str(directory / "stats.json"),
            overwrite_stats=True,
        )["code_snippet"]

    def test_every_emitted_keyword_is_honoured(self):
        """A varied dataset, so each feature class is exercised."""
        import ast

        with tempfile.TemporaryDirectory() as tmp:
            snippet = self._snippet_for(Path(tmp))

        offenders = []
        for node in ast.walk(ast.parse(snippet)):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            allowed = self.ALLOWED.get(node.func.id)
            if allowed is None:
                continue
            for keyword in node.keywords:
                if keyword.arg and keyword.arg not in allowed:
                    offenders.append(f"{node.func.id}({keyword.arg}=...)")

        self.assertEqual(offenders, [], f"snippet writes ignored options: {offenders}")

    def test_the_varied_snippet_executes(self):
        """Covering all four classes, not just the simple ones."""
        with tempfile.TemporaryDirectory() as tmp:
            snippet = self._snippet_for(Path(tmp))
        namespace = {}
        exec(  # noqa: S102 - running the snippet is the point
            compile(snippet, "<snippet>", "exec"), namespace
        )
        self.assertIn("features", namespace)
