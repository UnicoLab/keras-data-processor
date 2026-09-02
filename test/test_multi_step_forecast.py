"""Tests for `TimeSeriesInferenceFormatter.generate_multi_step_forecast`.

The method called `self._check_history_requirements(...)`, which does not
exist -- the real name is `_check_inference_data_sufficiency` -- so this public
entry point raised `AttributeError` on every call, whatever it was given. It
had no test, which is why nothing noticed.
"""

import tempfile
import unittest
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import pytest

from kdp import FeatureType, PreprocessingModel, TimeSeriesFeature
from kdp.time_series.inference import TimeSeriesInferenceFormatter

ROWS = 60


def _history():
    """Sixty ordered days, enough for a lag-7 lookback."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=ROWS).strftime("%Y-%m-%d"),
            "sales": np.linspace(10, 70, ROWS),
        }
    )


def _formatter(tmp_path, **feature_kwargs):
    keras.backend.clear_session()
    csv_path = tmp_path / "series.csv"
    _history().to_csv(csv_path, index=False)
    preprocessor = PreprocessingModel(
        path_data=str(csv_path),
        features_specs={
            "sales": TimeSeriesFeature(
                name="sales",
                sort_by="date",
                **{"lag_config": {"lags": [1, 7]}, **feature_kwargs},
            ),
            "date": FeatureType.DATE,
        },
        features_stats_path=str(tmp_path / "stats.json"),
        overwrite_stats=True,
    )
    preprocessor.build_preprocessor()
    return TimeSeriesInferenceFormatter(preprocessor)


FUTURE = pd.date_range("2023-03-02", periods=5).strftime("%Y-%m-%d").tolist()


@pytest.mark.time_series
class TestGenerateMultiStepForecast(unittest.TestCase):
    """The placeholder frame a caller fills in step by step."""

    def test_it_returns_one_row_per_future_date(self):
        """The whole method used to raise AttributeError."""
        with tempfile.TemporaryDirectory() as tmp:
            frame = _formatter(Path(tmp)).generate_multi_step_forecast(
                _history(), FUTURE
            )
        self.assertEqual(len(frame), len(FUTURE))

    def test_the_sort_column_carries_the_future_dates(self):
        """Each row is the step it stands for."""
        with tempfile.TemporaryDirectory() as tmp:
            frame = _formatter(Path(tmp)).generate_multi_step_forecast(
                _history(), FUTURE
            )
        self.assertEqual(frame["date"].tolist(), FUTURE)

    def test_the_target_column_is_empty(self):
        """The caller fills each value in from its own model."""
        with tempfile.TemporaryDirectory() as tmp:
            frame = _formatter(Path(tmp)).generate_multi_step_forecast(
                _history(), FUTURE
            )
        self.assertTrue(frame["sales"].isna().all())

    def test_steps_limits_the_frame(self):
        """Forecasting fewer steps than dates supplied."""
        with tempfile.TemporaryDirectory() as tmp:
            frame = _formatter(Path(tmp)).generate_multi_step_forecast(
                _history(), FUTURE, steps=3
            )
        self.assertEqual(len(frame), 3)
        self.assertEqual(frame["date"].tolist(), FUTURE[:3])

    def test_asking_for_more_steps_than_dates_raises(self):
        """Otherwise the extra rows would have no timestamp."""
        with tempfile.TemporaryDirectory() as tmp:
            formatter = _formatter(Path(tmp))
            with self.assertRaises(ValueError) as ctx:
                formatter.generate_multi_step_forecast(_history(), FUTURE, steps=99)
        self.assertIn("99", str(ctx.exception))

    def test_too_little_history_raises_with_the_requirement(self):
        """A lag-7 feature needs at least seven rows of context."""
        with tempfile.TemporaryDirectory() as tmp:
            formatter = _formatter(Path(tmp))
            with self.assertRaises(ValueError) as ctx:
                formatter.generate_multi_step_forecast(_history().head(2), FUTURE)
        self.assertIn("sales", str(ctx.exception))

    def test_it_accepts_a_dict_as_history(self):
        """`_convert_to_dict` handles both, so both must work."""
        with tempfile.TemporaryDirectory() as tmp:
            history = _history()
            frame = _formatter(Path(tmp)).generate_multi_step_forecast(
                {c: history[c].tolist() for c in history}, FUTURE
            )
        self.assertEqual(len(frame), len(FUTURE))

    def test_a_grouped_feature_carries_its_group(self):
        """With `group_by`, the caller names which series to forecast."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            keras.backend.clear_session()
            history = _history()
            history["store"] = "s1"
            csv_path = tmp_path / "grouped.csv"
            history.to_csv(csv_path, index=False)

            preprocessor = PreprocessingModel(
                path_data=str(csv_path),
                features_specs={
                    "sales": TimeSeriesFeature(
                        name="sales",
                        sort_by="date",
                        group_by="store",
                        lag_config={"lags": [1]},
                    ),
                    "date": FeatureType.DATE,
                    "store": FeatureType.STRING_CATEGORICAL,
                },
                features_stats_path=str(tmp_path / "stats.json"),
                overwrite_stats=True,
            )
            preprocessor.build_preprocessor()
            frame = TimeSeriesInferenceFormatter(
                preprocessor
            ).generate_multi_step_forecast(history, FUTURE, group_id="s1")

        self.assertEqual(len(frame), len(FUTURE))
        self.assertTrue((frame["store"] == "s1").all())


if __name__ == "__main__":
    unittest.main()
