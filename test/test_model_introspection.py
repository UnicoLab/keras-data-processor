"""Tests for the introspection methods the documentation promised.

`get_timing_metrics`, `get_memory_usage` and `plot_model` were all shown in the
docs and none of them existed, so every page that used them ended in
`AttributeError`. The timing and memory numbers were in fact being measured all
along by the `_monitor_performance` decorator and then written only to a debug
log.
"""

import tempfile
import unittest
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import pytest

from kdp import FeatureType, PreprocessingModel

COLUMNS = {"num1": "numeric", "cat1": "categorical"}


def _dataset(directory: Path, rows: int = 200) -> Path:
    """A small mixed-type CSV, enough to exercise several build steps."""
    rng = np.random.default_rng(0)
    csv_path = directory / "data.csv"
    pd.DataFrame(
        {
            "num1": rng.normal(50, 10, rows),
            "cat1": rng.choice(["a", "b", "c"], rows),
        },
    ).to_csv(csv_path, index=False)
    return csv_path


def _preprocessor(directory: Path) -> PreprocessingModel:
    """An unbuilt preprocessor over that CSV."""
    keras.backend.clear_session()
    return PreprocessingModel(
        path_data=str(_dataset(directory)),
        features_specs={
            "num1": FeatureType.FLOAT_NORMALIZED,
            "cat1": FeatureType.STRING_CATEGORICAL,
        },
        features_stats_path=str(directory / "stats.json"),
        overwrite_stats=True,
    )


@pytest.mark.unit
class TestTimingAndMemoryMetrics(unittest.TestCase):
    """The numbers the optimization page prints."""

    def test_nothing_is_reported_before_a_build(self):
        """A fresh model has run no steps, so it has measured nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _preprocessor(Path(tmp))
            self.assertEqual(preprocessor.get_timing_metrics()["total_seconds"], 0)
            self.assertEqual(preprocessor.get_timing_metrics()["steps"], {})
            self.assertEqual(preprocessor.get_memory_usage()["peak_mb"], 0.0)

    def test_building_records_time_per_step(self):
        """The measurements used to be discarded after being logged."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _preprocessor(Path(tmp))
            preprocessor.build_preprocessor()
            metrics = preprocessor.get_timing_metrics()

        self.assertGreater(metrics["total_seconds"], 0)
        self.assertGreater(len(metrics["steps"]), 1)
        self.assertAlmostEqual(
            metrics["total_seconds"],
            sum(metrics["steps"].values()),
            places=6,
        )
        for seconds in metrics["steps"].values():
            self.assertGreaterEqual(seconds, 0)

    def test_memory_is_reported_per_step_in_megabytes(self):
        """CPU-only hosts report zeros; the shape of the answer is the point."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _preprocessor(Path(tmp))
            preprocessor.build_preprocessor()
            usage = preprocessor.get_memory_usage()

        self.assertIn("peak_mb", usage)
        self.assertEqual(sorted(usage["steps"]), sorted(usage["steps"]))
        self.assertGreaterEqual(usage["peak_mb"], 0.0)
        self.assertEqual(
            usage["peak_mb"],
            max(usage["steps"].values(), default=0.0),
        )

    def test_the_same_step_run_twice_accumulates(self):
        """A per-feature step runs once per feature and both count."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _preprocessor(Path(tmp))
            preprocessor.build_preprocessor()
            steps = preprocessor.get_timing_metrics()["steps"]

        self.assertIn("_add_input_signature", steps)
        self.assertGreater(steps["_add_input_signature"], 0)


@pytest.mark.unit
class TestPlotModel(unittest.TestCase):
    """Writing the architecture out as an image."""

    def test_an_unbuilt_model_says_so(self):
        """`plot_model` on a fresh preprocessor used to raise AttributeError."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _preprocessor(Path(tmp))
            with self.assertRaises(ValueError) as caught:
                preprocessor.plot_model(str(Path(tmp) / "unused.png"))
        self.assertIn("build_preprocessor", str(caught.exception))

    def test_a_diagram_is_written_or_the_missing_tool_is_named(self):
        """Keras prints and returns None when Graphviz is absent; we raise."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _preprocessor(Path(tmp))
            preprocessor.build_preprocessor()
            target = Path(tmp) / "architecture.png"
            try:
                preprocessor.plot_model(str(target))
            except ImportError as exc:
                self.assertIn("Graphviz", str(exc))
                self.skipTest("Graphviz is not installed on this host")
            self.assertTrue(target.exists())
            self.assertGreater(target.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
