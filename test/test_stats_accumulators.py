"""Tests for the accumulators that build the dataset statistics.

`TextAccumulator` and `DateAccumulator` were named in no test. They are what
turns a CSV column into the vocabulary a `TextVectorization` layer is adapted
with, and into the cyclical date statistics the date encoder reads, so a
regression in either one changes every model built from that data.
"""

import math
import unittest

import numpy as np
import pytest
import tensorflow as tf

from kdp.stats import DateAccumulator, TextAccumulator


@pytest.mark.unit
class TestTextAccumulator(unittest.TestCase):
    """The vocabulary side of the statistics."""

    def test_words_are_lowercased_and_deduplicated(self):
        """The vocabulary is case-insensitive, so `WORLD` and `world` are one."""
        accumulator = TextAccumulator()
        accumulator.update(tf.constant(["Hello   WORLD", "world of data"]))
        self.assertEqual(
            sorted(accumulator.get_unique_words()),
            ["data", "hello", "of", "world"],
        )

    def test_batches_accumulate(self):
        """Statistics are gathered batch by batch over the whole dataset."""
        accumulator = TextAccumulator()
        accumulator.update(tf.constant(["alpha beta"]))
        accumulator.update(tf.constant(["beta gamma"]))
        self.assertEqual(
            sorted(accumulator.get_unique_words()),
            ["alpha", "beta", "gamma"],
        )

    def test_runs_of_whitespace_do_not_create_empty_words(self):
        """An empty token poisons the vocabulary of the vectorizer."""
        accumulator = TextAccumulator()
        accumulator.update(tf.constant(["spaced    out\ttext"]))
        self.assertNotIn("", accumulator.get_unique_words())

    def test_words_come_back_as_str_not_bytes(self):
        """`bytes` here breaks Keras serialization of the vectorizer."""
        accumulator = TextAccumulator()
        accumulator.update(tf.constant(["some words"]))
        for word in accumulator.get_unique_words():
            self.assertIsInstance(word, str)

    def test_a_non_string_column_is_rejected(self):
        """Feeding a numeric column here is a configuration mistake."""
        accumulator = TextAccumulator()
        with self.assertRaises(ValueError):
            accumulator.update(tf.constant([1.0, 2.0]))

    def test_an_empty_accumulator_has_no_words(self):
        """Nothing read yet must not look like a one-word vocabulary."""
        self.assertEqual(TextAccumulator().get_unique_words(), [])


@pytest.mark.unit
class TestDateAccumulator(unittest.TestCase):
    """The cyclical date statistics."""

    DATES = ["2021-01-15", "2021-07-15", "2022-03-01"]

    def test_year_mean_matches_the_dates(self):
        """The year is accumulated as a plain number, not a cycle."""
        accumulator = DateAccumulator()
        accumulator.update(tf.constant(self.DATES))
        self.assertAlmostEqual(
            float(accumulator.mean()["year"]),
            (2021 + 2021 + 2022) / 3,
            places=3,
        )

    def test_month_is_encoded_on_a_circle(self):
        """January and July sit opposite each other, so the cosines cancel."""
        accumulator = DateAccumulator()
        accumulator.update(tf.constant(self.DATES))
        months = [1, 7, 3]
        self.assertAlmostEqual(
            float(accumulator.mean()["month_sin"]),
            float(np.mean([math.sin(2 * math.pi * m / 12) for m in months])),
            places=4,
        )
        self.assertAlmostEqual(
            float(accumulator.mean()["month_cos"]),
            float(np.mean([math.cos(2 * math.pi * m / 12) for m in months])),
            places=4,
        )

    def test_slashes_parse_like_dashes(self):
        """Both separators reach these accumulators from real CSV files."""
        dashed, slashed = DateAccumulator(), DateAccumulator()
        dashed.update(tf.constant(["2021-01-15", "2021-07-15"]))
        slashed.update(tf.constant(["2021/01/15", "2021/07/15"]))
        for key, value in dashed.mean().items():
            self.assertAlmostEqual(float(value), float(slashed.mean()[key]), places=4)

    def test_already_parsed_tensors_are_accepted(self):
        """The processor hands over parsed dates, not always strings."""
        accumulator = DateAccumulator()
        accumulator.update(
            tf.constant([[2021.0, 1.0, 15.0, 4.0], [2022.0, 7.0, 1.0, 4.0]]),
        )
        self.assertAlmostEqual(float(accumulator.mean()["year"]), 2021.5, places=3)

    def test_a_parsed_tensor_missing_components_is_rejected(self):
        """Silently reading column 3 of a 2-column tensor would be worse."""
        with self.assertRaises(ValueError):
            DateAccumulator().update(tf.constant([[2021.0, 1.0]]))

    def test_variance_is_reported_for_every_component(self):
        """The date encoder normalises with these, so none may be missing."""
        accumulator = DateAccumulator()
        accumulator.update(tf.constant(self.DATES))
        self.assertEqual(
            sorted(accumulator.variance()),
            ["day_of_week_cos", "day_of_week_sin", "month_cos", "month_sin", "year"],
        )
        for value in accumulator.variance().values():
            self.assertGreaterEqual(float(value), 0.0)

    def test_a_single_date_has_no_spread(self):
        """One observation cannot have variance, and must not report any."""
        accumulator = DateAccumulator()
        accumulator.update(tf.constant(["2021-01-15"]))
        self.assertAlmostEqual(float(accumulator.variance()["year"]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
