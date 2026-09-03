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

from kdp.stats import DateAccumulator, TextAccumulator, WelfordAccumulator


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


@pytest.mark.unit
class TestWelfordAccumulator(unittest.TestCase):
    """The mean and variance every normalised column is built from.

    This accumulated in float32 and subtracted the running mean from each raw
    value, so the deviations were computed at the magnitude of the values
    themselves. A column around 1e8 came back with a variance of *minus* 1.6e6
    against a true 1.0e6; one with a spread of 1e-4 was wrong by four orders of
    magnitude. `Normalization` divides by the root of that number, so the
    column reached the model as a constant and nothing raised.
    """

    CASES = {
        "ordinary": lambda rng: rng.normal(50, 12, 5000),
        "large magnitude": lambda rng: rng.normal(1e8, 1e3, 5000),
        "small magnitude": lambda rng: rng.normal(1e-6, 1e-7, 5000),
        "heavy tail": lambda rng: rng.standard_t(2, 5000) * 100,
        "constant": lambda rng: np.full(5000, 42.0),
    }

    @staticmethod
    def _accumulate(data, chunks=37, dtype=tf.float64):
        accumulator = WelfordAccumulator()
        for chunk in np.array_split(data, chunks):
            accumulator.update(tf.constant(chunk, dtype=dtype))
        return accumulator

    def test_mean_and_variance_match_numpy(self):
        rng = np.random.default_rng(3)
        for label, make in self.CASES.items():
            with self.subTest(case=label):
                data = make(rng)
                accumulator = self._accumulate(data)
                mean = float(accumulator.mean.numpy())
                variance = float(accumulator.variance.numpy())
                self.assertAlmostEqual(
                    mean / max(abs(float(np.mean(data))), 1e-30),
                    1.0 if np.mean(data) != 0 else mean,
                    places=6,
                )
                expected = float(np.var(data, ddof=1))
                if expected == 0.0:
                    self.assertAlmostEqual(variance, 0.0, places=12)
                else:
                    self.assertLess(abs(variance - expected) / expected, 1e-6, label)

    def test_variance_is_never_negative(self):
        """A negative variance is not a number a variance can be."""
        rng = np.random.default_rng(11)
        for label, make in self.CASES.items():
            with self.subTest(case=label):
                accumulator = self._accumulate(make(rng))
                self.assertGreaterEqual(float(accumulator.variance.numpy()), 0.0, label)

    def test_the_answer_does_not_depend_on_how_the_data_is_batched(self):
        """One row at a time or all at once has to give the same statistics."""
        data = np.random.default_rng(5).normal(1e7, 250, 900)
        results = [
            (
                float(self._accumulate(data, chunks=c).mean.numpy()),
                float(self._accumulate(data, chunks=c).variance.numpy()),
            )
            for c in (1, 7, 90, 900)
        ]
        for mean, variance in results[1:]:
            self.assertAlmostEqual(mean / results[0][0], 1.0, places=10)
            self.assertAlmostEqual(variance / results[0][1], 1.0, places=8)

    def test_an_empty_batch_changes_nothing(self):
        accumulator = WelfordAccumulator()
        accumulator.update(tf.constant(np.arange(10.0), dtype=tf.float64))
        before = (
            float(accumulator.n.numpy()),
            float(accumulator.mean.numpy()),
            float(accumulator.variance.numpy()),
        )
        accumulator.update(tf.constant([], dtype=tf.float64))
        after = (
            float(accumulator.n.numpy()),
            float(accumulator.mean.numpy()),
            float(accumulator.variance.numpy()),
        )
        self.assertEqual(before, after)

    def test_accumulators_do_not_share_state(self):
        first = WelfordAccumulator()
        first.update(tf.constant(np.zeros(100), dtype=tf.float64))
        second = WelfordAccumulator()
        second.update(tf.constant(np.full(50, 7.0), dtype=tf.float64))
        self.assertEqual(float(first.n.numpy()), 100.0)
        self.assertEqual(float(first.mean.numpy()), 0.0)
        self.assertEqual(float(second.n.numpy()), 50.0)
        self.assertAlmostEqual(float(second.mean.numpy()), 7.0, places=10)

    def test_merging_accumulators_equals_accumulating_the_whole(self):
        """Grouped statistics are pooled group by group through `merge`.

        The grouped time series path replaced each group with `count` copies of
        its own mean before combining, which keeps only the variance *between*
        the group means. Two groups of spread 25 whose means happened to be
        close reported 0.73 against a true 643.
        """
        rng = np.random.default_rng(9)
        parts = [
            rng.normal(10, 2, 300),
            rng.normal(1000, 50, 700),
            rng.normal(-5, 0.5, 120),
        ]
        whole = np.concatenate(parts)

        single = WelfordAccumulator()
        single.update(tf.constant(whole, dtype=tf.float64))

        pooled = WelfordAccumulator()
        for part in parts:
            group = WelfordAccumulator()
            group.update(tf.constant(part, dtype=tf.float64))
            pooled.merge(group)

        self.assertAlmostEqual(float(pooled.n.numpy()), float(len(whole)))
        self.assertAlmostEqual(
            float(pooled.mean.numpy()),
            float(single.mean.numpy()),
            places=8,
        )
        self.assertAlmostEqual(
            float(pooled.variance.numpy()) / float(single.variance.numpy()),
            1.0,
            places=9,
        )
        self.assertAlmostEqual(
            float(pooled.variance.numpy()) / float(np.var(whole, ddof=1)),
            1.0,
            places=9,
        )

    def test_merging_an_empty_accumulator_changes_nothing(self):
        accumulator = WelfordAccumulator()
        accumulator.update(tf.constant(np.arange(20.0), dtype=tf.float64))
        before = (
            float(accumulator.n.numpy()),
            float(accumulator.mean.numpy()),
            float(accumulator.variance.numpy()),
        )
        accumulator.merge(WelfordAccumulator())
        self.assertEqual(
            (
                float(accumulator.n.numpy()),
                float(accumulator.mean.numpy()),
                float(accumulator.variance.numpy()),
            ),
            before,
        )
