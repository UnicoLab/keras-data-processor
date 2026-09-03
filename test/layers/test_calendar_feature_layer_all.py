"""Every calendar feature `CalendarFeatureLayer` accepts.

Twelve of the seventeen had no test. Two were broken: `week_of_year` returned
NaN for every row because `isocalendar()` yields a frame indexed by the
timestamps, so assigning it into a RangeIndex frame aligned on nothing; and
`hour`, `minute` and `second` passed validation but had no branch in the
non-cyclic path, so they raised when asked for by their plain names.
"""

import unittest

import keras
import numpy as np
import pytest
import tensorflow as tf

from kdp.layers import CalendarFeatureLayer

# 2021-01-01 Friday, year/quarter/month start; 2021-12-31 Friday, year end;
# 2021-03-31 quarter end; 2021-07-04 Sunday.
DATES = ["2021-01-01", "2021-06-15", "2021-12-31", "2021-03-31", "2021-07-04"]
CYCLIC = (
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "quarter",
    "hour",
    "minute",
    "second",
)
FLAGS = (
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
)


def _values(feature, dates=None, **kwargs):
    keras.backend.clear_session()
    batch = tf.constant([[d] for d in (dates or DATES)])
    return np.asarray(CalendarFeatureLayer(features=[feature], **kwargs)(batch)).ravel()


@pytest.mark.layers
class TestEveryCalendarFeature(unittest.TestCase):
    """Each documented name has to produce usable numbers."""

    def test_all_documented_features_build(self):
        """The full list from the documentation, one at a time."""
        for feature in (*CYCLIC, "year", *FLAGS):
            values = _values(feature)
            self.assertEqual(len(values), len(DATES), feature)
            self.assertTrue(np.isfinite(values).all(), f"{feature} produced NaN/inf")

    def test_week_of_year_is_not_nan(self):
        """It was NaN for every date because of index alignment."""
        values = _values("week_of_year")
        self.assertTrue(np.isfinite(values).all())

    def test_week_of_year_matches_iso(self):
        """Checked against pandas rather than a hand-written expectation."""
        import pandas as pd

        raw = _values("week_of_year", normalize=False)
        expected = pd.DatetimeIndex(DATES).isocalendar().week.to_numpy(dtype=float)
        np.testing.assert_array_equal(raw, expected)

    def test_time_components_read_the_clock(self):
        """These raised when asked for by their plain names."""
        stamps = ["2021-06-15 13:45:30", "2021-06-15 00:00:00", "2021-06-15 23:59:59"]
        for feature, expected in (
            ("hour", [13.0, 0.0, 23.0]),
            ("minute", [45.0, 0.0, 59.0]),
            ("second", [30.0, 0.0, 59.0]),
        ):
            values = _values(feature, dates=stamps, normalize=False)
            np.testing.assert_array_equal(values, expected, err_msg=feature)

    def test_normalisation_keeps_values_in_range(self):
        """Normalised features feed straight into a model."""
        for feature in CYCLIC:
            values = _values(feature, normalize=True)
            self.assertTrue(
                np.all((values >= -0.001) & (values <= 1.001)),
                f"{feature} left [0, 1]: {values}",
            )

    def test_boolean_flags_are_zero_or_one(self):
        """A flag that drifts off 0/1 is not a flag."""
        for feature in FLAGS:
            values = _values(feature)
            self.assertTrue(set(np.unique(values)) <= {0.0, 1.0}, feature)

    def test_flags_fire_on_the_right_dates(self):
        """Semantics, not just shape."""
        expectations = {
            # 2021-01-01 is the start of month, quarter and year.
            "is_month_start": [1, 0, 0, 0, 0],
            "is_quarter_start": [1, 0, 0, 0, 0],
            "is_year_start": [1, 0, 0, 0, 0],
            # 2021-12-31 and 2021-03-31 end their month and quarter.
            "is_month_end": [0, 0, 1, 1, 0],
            "is_quarter_end": [0, 0, 1, 1, 0],
            "is_year_end": [0, 0, 1, 0, 0],
            # 2021-07-04 is a Sunday.
            "is_weekend": [0, 0, 0, 0, 1],
        }
        for feature, expected in expectations.items():
            np.testing.assert_array_equal(
                _values(feature), np.array(expected, dtype=float), err_msg=feature
            )

    def test_sin_and_cos_halves_can_be_taken_separately(self):
        """Appending `_sin` or `_cos` selects one half of the pair."""
        for feature in ("month", "day_of_week", "hour"):
            sin = _values(f"{feature}_sin")
            cos = _values(f"{feature}_cos")
            self.assertTrue(np.isfinite(sin).all() and np.isfinite(cos).all())
            np.testing.assert_allclose(sin**2 + cos**2, 1.0, atol=1e-4)

    def test_several_features_concatenate(self):
        """The real use is a list, and width must follow it."""
        keras.backend.clear_session()
        features = ["month", "day_of_week", "is_weekend", "is_year_start"]
        out = np.asarray(
            CalendarFeatureLayer(features=features)(tf.constant([[d] for d in DATES]))
        )
        self.assertEqual(out.shape, (len(DATES), len(features)))

    def test_an_unknown_feature_is_rejected(self):
        """`is_holiday` is the one the docs used to suggest."""
        with self.assertRaises(ValueError):
            CalendarFeatureLayer(features=["is_holiday"])

    def test_config_round_trips(self):
        """A saved model must rebuild the same feature list."""
        layer = CalendarFeatureLayer(features=["month", "is_weekend"], normalize=False)
        restored = CalendarFeatureLayer.from_config(layer.get_config())
        self.assertEqual(restored.features, ["month", "is_weekend"])
        self.assertFalse(restored.normalize)


if __name__ == "__main__":
    unittest.main()
