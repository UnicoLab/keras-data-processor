"""Behavioural tests for the three date layers.

These carry the date encoding every `FeatureType.DATE` column goes through, and
their validation and edge-date paths were untested: a wrong day-of-week or a
silently accepted malformed date changes model input without failing anything.
"""

import unittest
from datetime import date

import keras
import numpy as np
import pytest
import tensorflow as tf

from kdp.layers.date_encoding_layer import DateEncodingLayer
from kdp.layers.date_parsing_layer import DateParsingLayer
from kdp.layers.season_layer import SeasonLayer


@pytest.mark.layers
class TestDateParsingLayer(unittest.TestCase):
    """Parsing turns a string into [year, month, day, day_of_week]."""

    def setUp(self):
        self.layer = DateParsingLayer()

    def _parse(self, *dates):
        return np.asarray(self.layer(tf.constant([[d] for d in dates])))

    def test_day_of_week_matches_the_calendar(self):
        """Zeller's congruence is easy to get subtly wrong, so check real dates.

        The layer returns 0 for Sunday through 6 for Saturday.
        """
        samples = [
            "2024-01-01",  # Monday
            "2024-02-29",  # Thursday, leap day
            "2023-12-31",  # Sunday
            "2000-02-29",  # Tuesday, century leap year
            "1999-12-31",  # Friday
        ]
        parsed = self._parse(*samples)
        for row, text in zip(parsed, samples):
            expected = (date.fromisoformat(text).weekday() + 1) % 7
            self.assertEqual(int(row[3]), expected, f"day of week for {text}")

    def test_components_are_extracted(self):
        """Year, month and day come back as integers."""
        year, month, day, _ = self._parse("2021-06-15")[0]
        self.assertEqual((int(year), int(month), int(day)), (2021, 6, 15))

    def test_both_separators_parse_identically(self):
        """`YYYY/MM/DD` is documented as equivalent."""
        np.testing.assert_array_equal(
            self._parse("2021-06-15"), self._parse("2021/06/15")
        )

    def test_a_single_row_batch_works(self):
        """Squeezing a (1, 1) batch to a scalar used to break map_fn."""
        self.assertEqual(self._parse("2021-06-15").shape, (1, 4))

    def test_month_boundaries_are_accepted(self):
        """January and December are inside the validated range."""
        parsed = self._parse("2021-01-01", "2021-12-31")
        self.assertEqual([int(r[1]) for r in parsed], [1, 12])

    def test_a_malformed_date_is_rejected(self):
        """Silently accepting it would feed nonsense into the model."""
        for bad in ("not-a-date", "2021-06", "20210615", ""):
            with self.assertRaises(tf.errors.InvalidArgumentError, msg=bad):
                self._parse(bad)

    def test_out_of_range_components_are_rejected(self):
        """Each documented bound has an assertion behind it."""
        for bad in ("2021-13-01", "2021-00-01", "2021-06-32", "2021-06-00"):
            with self.assertRaises(tf.errors.InvalidArgumentError, msg=bad):
                self._parse(bad)

    def test_year_bounds_are_enforced(self):
        """The layer documents 1000-2200."""
        for bad in ("0999-01-01", "2201-01-01"):
            with self.assertRaises(tf.errors.InvalidArgumentError, msg=bad):
                self._parse(bad)

    def test_the_format_survives_serialization(self):
        """A saved model has to rebuild the layer with the same format."""
        layer = DateParsingLayer(date_format="YYYY/MM/DD")
        restored = DateParsingLayer.from_config(layer.get_config())
        self.assertEqual(restored.date_format, "YYYY/MM/DD")


@pytest.mark.layers
class TestDateEncodingLayer(unittest.TestCase):
    """Encoding turns those components into sin/cos pairs."""

    def setUp(self):
        self.layer = DateEncodingLayer()

    def _encode(self, year, month, day, dow):
        return np.asarray(
            self.layer(tf.constant([[year, month, day, dow]], dtype=tf.int32))
        )[0]

    def test_output_is_eight_wide(self):
        """Four components, each a (sin, cos) pair."""
        self.assertEqual(self._encode(2021, 6, 15, 2).shape, (8,))

    def test_every_value_is_within_the_unit_circle(self):
        """sin and cos never leave [-1, 1]."""
        encoded = self._encode(2021, 6, 15, 2)
        self.assertTrue(np.all(encoded >= -1.0000001))
        self.assertTrue(np.all(encoded <= 1.0000001))

    def test_each_pair_lies_on_the_unit_circle(self):
        """sin^2 + cos^2 == 1 is what makes the encoding cyclic."""
        encoded = self._encode(2021, 6, 15, 2)
        for i in range(0, 8, 2):
            self.assertAlmostEqual(
                float(encoded[i] ** 2 + encoded[i + 1] ** 2), 1.0, places=4
            )

    def test_december_and_january_are_adjacent(self):
        """The whole point of cyclic encoding: month 12 sits next to month 1."""
        december = self._encode(2021, 12, 15, 2)[2:4]
        january = self._encode(2021, 1, 15, 2)[2:4]
        june = self._encode(2021, 6, 15, 2)[2:4]
        self.assertLess(
            np.linalg.norm(december - january), np.linalg.norm(december - june)
        )

    def test_the_same_month_encodes_the_same_way_in_any_year(self):
        """The month component must not drift with the year."""
        np.testing.assert_allclose(
            self._encode(2021, 6, 15, 2)[2:4],
            self._encode(1995, 6, 15, 2)[2:4],
            rtol=1e-5,
        )


@pytest.mark.layers
class TestSeasonLayer(unittest.TestCase):
    """The season layer appends a four-way one-hot."""

    def setUp(self):
        self.layer = SeasonLayer()

    def _season(self, month):
        out = np.asarray(
            self.layer(tf.constant([[2021, month, 15, 2]], dtype=tf.int32))
        )[0]
        return out[4:]

    def test_four_columns_are_appended(self):
        """Input is four components; output adds the season one-hot."""
        out = np.asarray(self.layer(tf.constant([[2021, 6, 15, 2]], dtype=tf.int32)))
        self.assertEqual(out.shape, (1, 8))

    def test_exactly_one_season_is_set(self):
        """It is a one-hot, not a distribution."""
        for month in range(1, 13):
            season = self._season(month)
            self.assertEqual(float(season.sum()), 1.0, f"month {month}")

    def test_months_in_the_same_season_agree(self):
        """Meteorological seasons run in three-month blocks."""
        for block in ((12, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)):
            encodings = [self._season(m) for m in block]
            for other in encodings[1:]:
                np.testing.assert_array_equal(encodings[0], other)

    def test_different_seasons_differ(self):
        """A winter month and a summer month must not collide."""
        self.assertFalse(np.array_equal(self._season(1), self._season(7)))


@pytest.mark.layers
class TestDateLayersCompose(unittest.TestCase):
    """The three run back to back inside the real pipeline."""

    def test_parse_encode_season_chain(self):
        """This is exactly the order `_add_pipeline_date` builds."""
        keras.backend.clear_session()
        parsed = DateParsingLayer()(tf.constant([["2021-06-15"], ["2021-12-31"]]))
        encoded = DateEncodingLayer()(parsed)
        seasoned = SeasonLayer()(parsed)
        self.assertEqual(tuple(encoded.shape), (2, 8))
        self.assertEqual(tuple(seasoned.shape), (2, 8))

    def test_the_chain_is_graph_safe(self):
        """Every layer has to trace into a functional model."""
        keras.backend.clear_session()
        inputs = keras.Input(shape=(1,), dtype=tf.string)
        model = keras.Model(inputs, DateEncodingLayer()(DateParsingLayer()(inputs)))
        self.assertEqual(tuple(model(tf.constant([["2021-06-15"]])).shape), (1, 8))


if __name__ == "__main__":
    unittest.main()
