"""The advisor's advice has to come from the data, and has to be runnable.

`ModelAdvisor` reads `skewness`, `kurtosis`, `min` and `max` off each numeric
feature's statistics, and the statistics carried none of them. Every column
therefore arrived with the neutral defaults -- skew 0, kurtosis 3 -- and came
back as "Normal distribution detected, standard normalization recommended"
whatever shape it actually had, while the rescaling factor derived from `min`
and `max` always worked out to exactly 1.

The generated snippet had the matching problem: it built a `PreprocessingModel`
with no `path_data`, and every configuration the advisor recommends needs the
data, so pasting the advice raised.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kdp.features import FeatureType
from kdp.model_advisor import recommend_model_configuration
from kdp.stats import DatasetStatistics

ROWS = 4000


def _statistics(directory: Path, frame: pd.DataFrame) -> dict:
    data = directory / "data.csv"
    frame.to_csv(data, index=False)
    return DatasetStatistics(
        path_data=str(data),
        features_stats_path=str(directory / "stats.json"),
        features_specs={
            name: FeatureType.STRING_CATEGORICAL
            if frame[name].dtype == object
            else FeatureType.FLOAT_NORMALIZED
            for name in frame
        },
        overwrite_stats=True,
    ).main()


@pytest.mark.unit
class TestTheAdvisorCanTellDistributionsApart(unittest.TestCase):
    """Three differently shaped columns must not get one answer."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp())
        generator = np.random.default_rng(19)
        cls.frame = pd.DataFrame(
            {
                "gaussian": generator.normal(0.0, 1.0, ROWS),
                "heavy_tailed": generator.exponential(5.0, ROWS),
                "uniform": generator.uniform(0.0, 1.0, ROWS),
                "label": generator.choice(["a", "b", "c"], ROWS),
            },
        )
        cls.statistics = _statistics(cls.directory, cls.frame)
        cls.recommendation = recommend_model_configuration(cls.statistics)

    def test_the_statistics_carry_the_numbers_the_advisor_reads(self) -> None:
        numeric = self.statistics["numeric_stats"]
        for name in ("gaussian", "heavy_tailed", "uniform"):
            with self.subTest(feature=name):
                for key in ("min", "max", "skewness", "kurtosis"):
                    self.assertIn(key, numeric[name])

    def test_the_moments_match_the_data(self) -> None:
        numeric = self.statistics["numeric_stats"]
        for name in ("gaussian", "heavy_tailed", "uniform"):
            column = self.frame[name].to_numpy()
            centered = column - column.mean()
            second = (centered**2).mean()
            with self.subTest(feature=name):
                self.assertAlmostEqual(
                    numeric[name]["skewness"],
                    (centered**3).mean() / second**1.5,
                    places=2,
                )
                self.assertAlmostEqual(
                    numeric[name]["kurtosis"],
                    (centered**4).mean() / second**2,
                    places=2,
                )
                self.assertAlmostEqual(numeric[name]["min"], column.min(), places=3)
                self.assertAlmostEqual(numeric[name]["max"], column.max(), places=3)

    def test_each_column_gets_the_advice_its_shape_calls_for(self) -> None:
        features = self.recommendation["features"]
        self.assertEqual(features["gaussian"]["detected_distribution"], "normal")
        self.assertEqual(
            features["heavy_tailed"]["detected_distribution"],
            "heavy_tailed",
        )
        self.assertNotEqual(
            features["uniform"]["detected_distribution"],
            features["heavy_tailed"]["detected_distribution"],
        )

    def test_the_advice_is_not_the_same_sentence_for_every_column(self) -> None:
        notes = {
            name: tuple(self.recommendation["features"][name].get("notes", ()))
            for name in ("gaussian", "heavy_tailed", "uniform")
        }
        self.assertEqual(
            len(set(notes.values())),
            3,
            f"three differently shaped columns got the same advice: {notes}",
        )


@pytest.mark.unit
class TestTheGeneratedSnippetRuns(unittest.TestCase):
    """Advice you cannot paste is not advice."""

    def test_the_snippet_builds_a_preprocessor(self) -> None:
        directory = Path(tempfile.mkdtemp())
        generator = np.random.default_rng(23)
        frame = pd.DataFrame(
            {
                "amount": generator.exponential(3.0, 500),
                "score": generator.normal(0.0, 1.0, 500),
                "label": generator.choice(["x", "y"], 500),
            },
        )
        statistics = _statistics(directory, frame)
        snippet = recommend_model_configuration(statistics)["code_snippet"]

        self.assertIn('path_data="your_data.csv"', snippet)

        runnable = snippet.replace(
            'path_data="your_data.csv"',
            f'path_data="{directory / "data.csv"}"',
        ).replace(
            "    features_specs=features,",
            "    features_specs=features,\n"
            f'    features_stats_path="{directory / "fresh.json"}",\n'
            "    overwrite_stats=True,",
        )
        namespace: dict = {}
        exec(compile(runnable, "<advisor snippet>", "exec"), namespace)  # noqa: S102
        result = namespace["model"].build_preprocessor()
        self.assertIsNotNone(result["model"])


if __name__ == "__main__":
    unittest.main()
