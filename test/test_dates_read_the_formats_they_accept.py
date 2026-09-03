"""A date column has to be read, or refused where the caller can see why.

The parser reads year, then month, then day, with `-` or `/` between them. It
never read the format string it was given, so a day-first or month-first format
was accepted at construction, ignored, and met again at the first batch as an
assertion buried inside a TensorFlow graph error -- far from the line that
caused it.

A time after the date is the other half: timestamps are what a great many date
columns actually hold, and "2021-06-15 13:45:00" had no way through this layer
at all.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kdp import FeatureType, PreprocessingModel
from kdp.features import DateFeature
from kdp.layers.date_parsing_layer import DateParsingLayer
from kdp.processor import OutputModeOptions

READABLE = ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")
UNREADABLE = ("%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "nonsense")


@pytest.mark.unit
class TestDateFormatsAreCheckedWhereTheyAreWritten(unittest.TestCase):
    """The format is judged at construction, not at the first batch."""

    def test_readable_formats_are_accepted(self) -> None:
        for date_format in READABLE:
            with self.subTest(date_format=date_format):
                DateFeature(
                    name="d",
                    feature_type=FeatureType.DATE,
                    format=date_format,
                )

    def test_unreadable_formats_are_refused_by_name(self) -> None:
        for date_format in UNREADABLE:
            with self.subTest(date_format=date_format):
                with self.assertRaises(ValueError) as raised:
                    DateFeature(
                        name="signed_on",
                        feature_type=FeatureType.DATE,
                        format=date_format,
                    )
                message = str(raised.exception)
                self.assertIn("signed_on", message)
                self.assertIn(date_format, message)

    def test_date_format_is_the_same_option_as_format(self) -> None:
        """Both spellings appear in the wild; only one was ever read."""
        feature = DateFeature(
            name="d",
            feature_type=FeatureType.DATE,
            date_format="%Y/%m/%d",
        )
        self.assertEqual(feature.kwargs.get("format"), "%Y/%m/%d")

        with self.assertRaises(ValueError):
            DateFeature(
                name="d",
                feature_type=FeatureType.DATE,
                date_format="%d/%m/%Y",
            )


@pytest.mark.layers
class TestTheParserReadsTimestamps(unittest.TestCase):
    """A time after the date is dropped, not rejected."""

    def test_a_date_with_a_time_parses_to_the_same_components(self) -> None:
        layer = DateParsingLayer()
        plain = np.asarray(layer(tf.constant([["2021-06-15"]])))
        spaced = np.asarray(layer(tf.constant([["2021-06-15 13:45:00"]])))
        iso = np.asarray(layer(tf.constant([["2021-06-15T13:45:00"]])))
        slashed = np.asarray(layer(tf.constant([["2021/06/15 13:45:00"]])))

        np.testing.assert_array_equal(plain, spaced)
        np.testing.assert_array_equal(plain, iso)
        np.testing.assert_array_equal(plain, slashed)
        # year, month, day, day-of-week
        np.testing.assert_array_equal(plain[0][:3], [2021, 6, 15])

    def test_a_column_of_timestamps_goes_through_a_model(self) -> None:
        directory = Path(tempfile.mkdtemp())
        stamps = pd.date_range("2020-01-01 06:00:00", periods=60, freq="37h")
        frame = pd.DataFrame(
            {
                "seen_at": stamps.strftime("%Y-%m-%d %H:%M:%S"),
                "n": np.arange(60.0),
            },
        )
        data = directory / "data.csv"
        frame.to_csv(data, index=False)

        preprocessor = PreprocessingModel(
            path_data=str(data),
            features_stats_path=str(directory / "stats.json"),
            features_specs={
                "seen_at": DateFeature(
                    name="seen_at",
                    feature_type=FeatureType.DATE,
                    format="%Y-%m-%d %H:%M:%S",
                ),
                "n": FeatureType.FLOAT_NORMALIZED,
            },
            overwrite_stats=True,
            output_mode=OutputModeOptions.DICT,
        )
        preprocessor.build_preprocessor()
        result = np.asarray(
            preprocessor.model(
                {
                    "seen_at": tf.constant(frame["seen_at"].values.reshape(-1, 1)),
                    "n": tf.constant(
                        frame["n"].values.reshape(-1, 1),
                        dtype=tf.float32,
                    ),
                },
            )["seen_at"],
        )
        self.assertEqual(result.shape, (60, 8))
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(
            len(np.unique(result, axis=0)),
            1,
            "every timestamp encoded to the same row",
        )


if __name__ == "__main__":
    unittest.main()
