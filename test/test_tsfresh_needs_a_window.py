"""Statistics over a window of one are not statistics.

A `TimeSeriesFeature` hands one column per row to the layers it builds, and
`TSFreshFeatureLayer` clamps `window_size` to the number of steps it is given.
Over a single step the mean, the minimum and the maximum are all the value
itself and the standard deviation is zero, so a config asking for four
statistics returned the input column three times plus a column of zeros. The
width matched what `get_output_dim()` declared, the values were finite, and
none of it carried anything.

The wavelet already refused the same shape of input. This checks that tsfresh
does too, and that once it has a window its numbers are the ones pandas gets.
"""

import unittest

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kdp.features import TimeSeriesFeature
from kdp.layers.time_series.tsfresh_feature_layer import TSFreshFeatureLayer

ROWS = 96
SERIES = np.sin(np.arange(ROWS) / 6.0) * 10 + np.arange(ROWS) * 0.3 + 50


def _through(feature: TimeSeriesFeature) -> np.ndarray:
    data = tf.constant(SERIES.reshape(-1, 1), dtype=tf.float32)
    for layer in feature.build_layers():
        data = layer(data)
    return np.asarray(data)


@pytest.mark.time_series
class TestTsfreshNeedsAWindow(unittest.TestCase):
    """One step in, nothing to summarize."""

    def test_a_single_step_is_refused(self) -> None:
        with self.assertRaises(ValueError) as raised:
            TSFreshFeatureLayer(features=["mean", "std"])(
                tf.constant(np.arange(10.0).reshape(-1, 1), dtype=tf.float32),
            )
        message = str(raised.exception)
        self.assertIn("two time steps", message)
        self.assertIn("lag_config", message)

    def test_a_time_series_feature_without_a_window_is_refused(self) -> None:
        feature = TimeSeriesFeature(
            name="sales",
            sort_by="order",
            tsfresh_feature_config={
                "features": ["mean", "std", "min", "max"],
                "window_size": 8,
            },
        )
        with self.assertRaises(ValueError):
            _through(feature)

    def test_with_a_window_the_statistics_are_the_ones_pandas_gets(self) -> None:
        window = 8
        feature = TimeSeriesFeature(
            name="sales",
            sort_by="order",
            lag_config={
                "lags": list(range(1, window)),
                "drop_na": False,
                "fill_value": 0.0,
            },
            tsfresh_feature_config={
                "features": ["mean", "std", "min", "max"],
                "window_size": window,
            },
        )
        produced = _through(feature)
        self.assertEqual(produced.shape, (ROWS, 4))

        series = pd.Series(SERIES)
        rolling = series.rolling(window)
        expected = {
            0: rolling.mean(),
            1: rolling.std(ddof=0),
            2: rolling.min(),
            3: rolling.max(),
        }
        # Only the rows with a full window: earlier ones sit on the lag padding.
        for column, reference in expected.items():
            with self.subTest(statistic=column):
                np.testing.assert_allclose(
                    produced[window - 1 :, column],
                    reference.to_numpy()[window - 1 :],
                    atol=1e-2,
                )

    def test_the_statistics_are_not_four_copies_of_the_input(self) -> None:
        """What the degenerate window produced, stated as a check."""
        feature = TimeSeriesFeature(
            name="sales",
            sort_by="order",
            lag_config={"lags": [1, 2, 3, 4, 5], "drop_na": False, "fill_value": 0.0},
            tsfresh_feature_config={
                "features": ["mean", "std", "min", "max"],
                "window_size": 6,
            },
        )
        produced = _through(feature)
        self.assertGreater(
            produced[:, 1].std(),
            0.0,
            "the standard deviation column is constant",
        )
        distinct = {tuple(np.round(produced[:, i], 4)) for i in range(4)}
        self.assertEqual(len(distinct), 4, "two statistics produced the same column")


if __name__ == "__main__":
    unittest.main()
