"""A calendar time series feature has to survive the statistics pass.

The documented way to ask for calendar features is a `TimeSeriesFeature` whose
`calendar_feature_config` names the components to extract. Its column holds
dates, and everything it produces is derived from the string by
`CalendarFeatureLayer` -- there is no mean or variance to take. The statistics
pass fed it to the numeric accumulator anyway and died inside a `tf.function`
with "Cast string to double is not supported", so that whole documented
configuration could not be built at all.

The layer itself was covered; the path from `PreprocessingModel` to it was not.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kdp import FeatureType, PreprocessingModel
from kdp.features import TimeSeriesFeature
from kdp.processor import OutputModeOptions

ROWS = 70
COMPONENTS = [
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
]


def _frame() -> pd.DataFrame:
    # Nearly four years, so every component below actually varies.
    stamps = pd.date_range("2021-01-01", periods=ROWS, freq="20D")
    return pd.DataFrame(
        {
            "when": stamps.strftime("%Y-%m-%d"),
            "sales": np.arange(ROWS, dtype=float),
        },
    )


@pytest.mark.time_series
class TestCalendarFeaturesCanBeBuilt(unittest.TestCase):
    """From `features_specs` all the way to a tensor."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.frame = _frame()
        self.data = self.directory / "data.csv"
        self.frame.to_csv(self.data, index=False)

    def _run(self, **config) -> tuple[np.ndarray, int]:
        feature = TimeSeriesFeature(name="when", calendar_feature_config=config)
        preprocessor = PreprocessingModel(
            path_data=str(self.data),
            features_stats_path=str(self.directory / f"stats_{len(config)}.json"),
            features_specs={"when": feature, "sales": FeatureType.FLOAT_NORMALIZED},
            overwrite_stats=True,
            output_mode=OutputModeOptions.DICT,
        )
        preprocessor.build_preprocessor()
        produced = np.asarray(
            preprocessor.model(
                {
                    "when": tf.constant(self.frame["when"].values.reshape(-1, 1)),
                    "sales": tf.constant(
                        self.frame["sales"].values.reshape(-1, 1),
                        dtype=tf.float32,
                    ),
                },
            )["when"],
        )
        return produced, feature.get_output_dim()

    def test_the_documented_configuration_builds_and_runs(self) -> None:
        produced, declared = self._run(features=COMPONENTS)
        self.assertEqual(produced.shape, (ROWS, len(COMPONENTS)))
        self.assertEqual(produced.shape[1], declared)
        self.assertTrue(np.isfinite(produced).all())

    def test_no_component_comes_back_constant(self) -> None:
        produced, _ = self._run(features=COMPONENTS)
        constant = [
            name
            for index, name in enumerate(COMPONENTS)
            if produced[:, index].std() < 1e-12
        ]
        self.assertEqual(constant, [], f"constant calendar columns: {constant}")

    def test_the_raw_components_are_the_ones_pandas_reads(self) -> None:
        produced, _ = self._run(features=COMPONENTS, normalize=False)
        stamps = pd.to_datetime(self.frame["when"])
        expected = {
            "year": stamps.dt.year,
            "month": stamps.dt.month,
            "day": stamps.dt.day,
            "day_of_week": stamps.dt.dayofweek,
            "day_of_year": stamps.dt.dayofyear,
            "week_of_year": stamps.dt.isocalendar().week.astype(int),
            "is_weekend": stamps.dt.dayofweek.isin([5, 6]).astype(int),
            "is_month_start": stamps.dt.is_month_start.astype(int),
            "is_month_end": stamps.dt.is_month_end.astype(int),
        }
        for index, name in enumerate(COMPONENTS):
            with self.subTest(component=name):
                np.testing.assert_allclose(
                    produced[:, index],
                    expected[name].to_numpy().astype(float),
                    atol=1e-4,
                )

    def test_normalize_puts_them_in_the_unit_interval(self) -> None:
        produced, _ = self._run(features=COMPONENTS, normalize=True)
        self.assertGreaterEqual(produced.min(), 0.0)
        self.assertLessEqual(produced.max(), 1.0)

    def test_the_cyclical_components_can_be_asked_for(self) -> None:
        produced, declared = self._run(
            features=["month_sin", "month_cos", "day_of_week_sin", "day_of_week_cos"],
        )
        self.assertEqual(produced.shape, (ROWS, 4))
        self.assertEqual(produced.shape[1], declared)
        self.assertTrue(np.isfinite(produced).all())


if __name__ == "__main__":
    unittest.main()
