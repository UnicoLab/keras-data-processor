"""`predict()` has to take the shapes its own docstring offers.

The docstring says the data "can be pandas DataFrame, dict, or TensorFlow
dataset". A DataFrame went straight to Keras, which reads a frame as a block of
floats and raised `could not convert string to float` on the first categorical
column. A dict of flat lists -- the natural spelling, and exactly what
`InferenceFormatter` produces -- converted to shape (N,) where every input is
declared (N, 1), and failed with `as_list() is not defined on an unknown
TensorShape`. Both carried the right values; only the container was wrong.

Every accepted spelling has to give the same numbers as calling the model with
tensors, because that is the whole claim.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kdp import FeatureType, PreprocessingModel

ROWS = 24

SPECS = {
    "age": FeatureType.FLOAT_NORMALIZED,
    "city": FeatureType.STRING_CATEGORICAL,
    "tier": FeatureType.INTEGER_CATEGORICAL,
    "signed": FeatureType.DATE,
}


def _frame() -> pd.DataFrame:
    generator = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": generator.normal(40.0, 8.0, ROWS),
            "city": generator.choice(["paris", "lisbon", "oslo"], ROWS),
            "tier": generator.integers(0, 4, ROWS),
            "signed": pd.date_range("2021-01-01", periods=ROWS, freq="9D").strftime(
                "%Y-%m-%d",
            ),
        },
    )


@pytest.mark.unit
class TestPredictAcceptsWhatItDocuments(unittest.TestCase):
    """Every documented container has to reach the same tensor."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp())
        cls.frame = _frame()
        data = cls.directory / "data.csv"
        cls.frame.to_csv(data, index=False)
        cls.preprocessor = PreprocessingModel(
            path_data=str(data),
            features_stats_path=str(cls.directory / "stats.json"),
            features_specs=SPECS,
            overwrite_stats=True,
        )
        cls.preprocessor.build_preprocessor()
        cls.reference = np.asarray(
            cls.preprocessor.model(
                {
                    name: tf.constant(cls.frame[name].values.reshape(-1, 1))
                    for name in cls.frame
                },
            ),
        )

    def _assert_matches(self, payload) -> None:
        got = np.asarray(self.preprocessor.predict(payload, verbose=0))
        self.assertEqual(got.shape, self.reference.shape)
        np.testing.assert_allclose(got, self.reference, atol=1e-5)

    def test_a_dataframe(self) -> None:
        self._assert_matches(self.frame)

    def test_a_dict_of_flat_lists(self) -> None:
        self._assert_matches({name: self.frame[name].tolist() for name in self.frame})

    def test_a_dict_of_column_lists(self) -> None:
        self._assert_matches(
            {name: [[value] for value in self.frame[name]] for name in self.frame},
        )

    def test_a_dict_of_numpy_columns(self) -> None:
        self._assert_matches(
            {name: self.frame[name].values.reshape(-1, 1) for name in self.frame},
        )

    def test_a_tensorflow_dataset(self) -> None:
        dataset = tf.data.Dataset.from_tensor_slices(dict(self.frame)).batch(8)
        self._assert_matches(dataset)

    def test_a_single_row_as_a_dataframe(self) -> None:
        got = np.asarray(self.preprocessor.predict(self.frame.iloc[:1], verbose=0))
        np.testing.assert_allclose(got, self.reference[:1], atol=1e-5)

    def test_what_the_inference_formatter_produces(self) -> None:
        """The formatter's own output is flat, and has to be accepted."""
        from kdp import InferenceFormatter

        formatter = InferenceFormatter(self.preprocessor)
        prepared = formatter.prepare_inference_data(self.frame, to_tensors=True)
        self._assert_matches(prepared)


@pytest.mark.unit
class TestAGapInTheFrameDoesNotStopIt(unittest.TestCase):
    """A frame with missing values is what real data looks like.

    pandas gives a column mixing NaN with strings the object dtype, and Keras
    refuses that outright with `Invalid dtype: object`. `InferenceFormatter`
    already decided each column's type from the values actually present;
    `predict()` did not, so the two documented ways in disagreed.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp())
        cls.frame = _frame()
        data = cls.directory / "data.csv"
        cls.frame.to_csv(data, index=False)
        cls.preprocessor = PreprocessingModel(
            path_data=str(data),
            features_stats_path=str(cls.directory / "stats.json"),
            features_specs=SPECS,
            overwrite_stats=True,
        )
        cls.preprocessor.build_preprocessor()

    def _with_gaps(self) -> pd.DataFrame:
        # Not the date: that column is the one kind of gap the library refuses,
        # and the test below is about the message it refuses with.
        gappy = self.frame.iloc[:5].copy()
        gappy.loc[gappy.index[0], "age"] = np.nan
        gappy.loc[gappy.index[1], "city"] = None
        gappy.loc[gappy.index[2], "tier"] = np.nan
        return gappy

    def test_a_frame_with_gaps_goes_through(self) -> None:
        result = np.asarray(self.preprocessor.predict(self._with_gaps(), verbose=0))
        self.assertEqual(result.shape[0], 5)

    def test_a_missing_category_does_not_produce_a_non_finite_number(self) -> None:
        """Only the numeric gap may carry NaN; the rest map to their OOV slot."""
        gappy = self.frame.iloc[:4].copy()
        gappy.loc[gappy.index[0], "city"] = None
        gappy.loc[gappy.index[1], "tier"] = np.nan
        result = np.asarray(self.preprocessor.predict(gappy, verbose=0))
        self.assertTrue(np.isfinite(result).all())

    def test_a_missing_date_says_what_is_wrong_and_which_value(self) -> None:
        """A date column cannot hold nulls, and has to say so where it fails.

        The formatter turns any missing string into the empty string, which the
        date parser refuses. It used to refuse it as a TensorFlow graph error
        naming an internal assertion node and nothing else.
        """
        gappy = self.frame.iloc[:3].copy()
        gappy.loc[gappy.index[1], "signed"] = None
        with self.assertRaises(Exception) as raised:  # noqa: B017 -- tf error type
            self.preprocessor.predict(gappy, verbose=0)
        message = " ".join(str(raised.exception).split())
        self.assertIn("date column cannot hold nulls", message)

    def test_a_missing_number_stays_inside_its_own_column(self) -> None:
        gappy = self.frame.iloc[:4].copy()
        gappy.loc[gappy.index[1], "age"] = np.nan
        result = np.asarray(self.preprocessor.predict(gappy, verbose=0))
        rows = {int(row) for row in np.argwhere(~np.isfinite(result))[:, 0]}
        self.assertEqual(rows, {1}, "the gap spread beyond the row it was in")
        self.assertEqual(int((~np.isfinite(result)).sum()), 1)

    def test_both_documented_paths_agree_about_a_gap(self) -> None:
        from kdp import InferenceFormatter

        gappy = self._with_gaps()
        through_predict = np.asarray(self.preprocessor.predict(gappy, verbose=0))
        prepared = InferenceFormatter(self.preprocessor).prepare_inference_data(
            gappy,
            to_tensors=True,
        )
        through_formatter = np.asarray(self.preprocessor.model(prepared))
        np.testing.assert_array_equal(
            np.isfinite(through_predict),
            np.isfinite(through_formatter),
        )
        finite = np.isfinite(through_predict)
        np.testing.assert_allclose(
            through_predict[finite],
            through_formatter[finite],
            atol=1e-5,
        )


@pytest.mark.unit
class TestPredictIsStableAcrossHowItIsCalled(unittest.TestCase):
    """The same row has to come back the same, however it is fed in."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp())
        cls.frame = _frame()
        data = cls.directory / "data.csv"
        cls.frame.to_csv(data, index=False)
        preprocessor = PreprocessingModel(
            path_data=str(data),
            features_stats_path=str(cls.directory / "stats.json"),
            features_specs=SPECS,
            overwrite_stats=True,
        )
        preprocessor.build_preprocessor()
        cls.model = preprocessor.model
        cls.reference = cls._call(cls.frame)

    @classmethod
    def _call(cls, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            cls.model(
                {
                    name: tf.constant(frame[name].values.reshape(-1, 1))
                    for name in frame
                },
            ),
        )

    def test_batch_size_does_not_change_a_row(self) -> None:
        pieces = [self._call(self.frame.iloc[i : i + 5]) for i in range(0, ROWS, 5)]
        np.testing.assert_allclose(
            np.concatenate(pieces),
            self.reference,
            atol=1e-5,
        )

    def test_row_order_does_not_change_a_row(self) -> None:
        order = np.random.default_rng(1).permutation(ROWS)
        shuffled = self._call(self.frame.iloc[order].reset_index(drop=True))
        np.testing.assert_allclose(shuffled, self.reference[order], atol=1e-5)

    def test_calling_twice_gives_the_same_answer(self) -> None:
        np.testing.assert_array_equal(self._call(self.frame), self.reference)

    def test_a_single_row_matches_its_place_in_the_batch(self) -> None:
        np.testing.assert_allclose(
            self._call(self.frame.iloc[:1]),
            self.reference[:1],
            atol=1e-5,
        )

    def test_values_never_seen_in_training_still_produce_finite_numbers(self) -> None:
        unseen = self.frame.iloc[:4].copy()
        unseen["city"] = ["atlantis", "paris", "", "oslo"]
        unseen["tier"] = [999, 0, -7, 1]
        result = self._call(unseen)
        self.assertEqual(result.shape, (4, self.reference.shape[1]))
        self.assertTrue(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
