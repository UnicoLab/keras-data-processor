"""Tests for how the time series formatter renders and orders rows.

Both defects here only appeared when a caller passed a real `pd.Timestamp` for
the new row while the history held date strings -- exactly what the
incremental-forecast example does.
"""

import datetime
import unittest

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kdp.inference.base import InferenceFormatter, _as_text


@pytest.mark.unit
class TestDateRendering(unittest.TestCase):
    """A date must reach the graph in the form DateParsingLayer can parse."""

    def setUp(self):
        self.formatter = InferenceFormatter.__new__(InferenceFormatter)

    def test_timestamp_loses_the_time_component(self):
        """`str(Timestamp)` gave "2023-03-01 00:00:00", which failed to parse."""
        self.assertEqual(_as_text(pd.Timestamp("2023-03-01")), "2023-03-01")

    def test_datetime_and_date_render_the_same_way(self):
        """Whatever the caller holds, the graph sees one format."""
        self.assertEqual(_as_text(datetime.datetime(2023, 3, 1, 13, 45)), "2023-03-01")
        self.assertEqual(_as_text(datetime.date(2023, 3, 1)), "2023-03-01")

    def test_numpy_datetime64_renders_the_same_way(self):
        """A frame column often yields datetime64 rather than Timestamp."""
        self.assertEqual(_as_text(np.datetime64("2023-03-01")), "2023-03-01")

    def test_plain_strings_are_untouched(self):
        """History rows already carry the right format."""
        self.assertEqual(_as_text("2023-03-01"), "2023-03-01")
        self.assertEqual(_as_text("Store_0"), "Store_0")

    def test_a_mixed_date_column_converts_consistently(self):
        """One Timestamp among strings must not produce two formats."""
        result = self.formatter._convert_to_tensors(
            {"date": ["2023-01-01", pd.Timestamp("2023-03-01")]}
        )["date"]
        self.assertEqual(
            [v.decode() for v in result.numpy().tolist()],
            ["2023-01-01", "2023-03-01"],
        )
        self.assertEqual(result.dtype, tf.string)


@pytest.mark.time_series
class TestChronologicalOrdering(unittest.TestCase):
    """The newest row must sort last so `[-1]` is the row being predicted."""

    def test_new_row_sorts_after_history(self):
        """Comparing a Timestamp against strings did not order chronologically."""
        import tempfile
        from pathlib import Path

        import keras

        from kdp import FeatureType, PreprocessingModel, TimeSeriesFeature
        from kdp.time_series.inference import TimeSeriesInferenceFormatter

        keras.backend.clear_session()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = 40
            frame = pd.DataFrame(
                {
                    "date": pd.date_range("2023-01-01", periods=rows).strftime(
                        "%Y-%m-%d"
                    ),
                    "sales": np.linspace(10, 50, rows),
                }
            )
            csv_path = tmp_path / "sales.csv"
            frame.to_csv(csv_path, index=False)

            preprocessor = PreprocessingModel(
                path_data=str(csv_path),
                features_specs={
                    "sales": TimeSeriesFeature(
                        name="sales", sort_by="date", lag_config={"lags": [1]}
                    ),
                    "date": FeatureType.DATE,
                },
                features_stats_path=str(tmp_path / "stats.json"),
                overwrite_stats=True,
            )
            preprocessor.build_preprocessor()

            formatter = TimeSeriesInferenceFormatter(preprocessor)
            formatted = formatter.format_for_incremental_prediction(
                frame,
                {"date": pd.Timestamp("2023-06-01"), "sales": np.nan},
                to_tensors=True,
            )
            dates = [v.decode() for v in formatted["date"].numpy().tolist()]

        self.assertEqual(dates[-1], "2023-06-01")
        self.assertEqual(dates, sorted(dates))


if __name__ == "__main__":
    unittest.main()
