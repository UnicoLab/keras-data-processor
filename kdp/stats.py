import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from loguru import logger

from kdp.features import (
    CategoricalFeature,
    FeatureSpaceConverter,
    FeatureType,
    NumericalFeature,
    TimeSeriesFeature,
)
from kdp.layers.date_parsing_layer import DateParsingLayer

MAX_WORKERS = os.cpu_count() or 4


class WelfordAccumulator:
    """Streaming mean and variance for a sequence of numbers.

    Accumulates in float64 and pools each batch with the Chan-Golub-LeVeque
    update. Both matter, and float32 was doing real damage: the deviations
    `x - mean` were computed at the magnitude of the values themselves, so a
    column of large numbers lost every significant digit to cancellation. A
    column around 1e8 with a spread of 1e3 came out with a variance of
    *minus* 1.6e6 against a true 1.0e6, and one with a spread of 1e-4 was
    wrong by four orders of magnitude.

    Nothing raised. `Normalization` divides by the square root of that
    variance, so the column arrived at the model as a constant -- every ID,
    epoch timestamp, amount in cents or sensor reading with a large offset
    silently flattened, its signal gone.

    Pooling each batch from its own mean keeps the subtraction at the scale of
    the spread rather than the values, and float64 leaves ten orders of
    magnitude of headroom on top. Together they hold the relative error near
    1e-13 on the cases that used to break.
    """

    def __init__(self):
        """Initializes the accumulators for the Welford algorithm."""
        self.n = tf.Variable(
            0.0,
            dtype=tf.float64,
            trainable=False,
        )
        self.mean = tf.Variable(
            0.0,
            dtype=tf.float64,
            trainable=False,
        )
        self.M2 = tf.Variable(
            0.0,
            dtype=tf.float64,
            trainable=False,
        )
        # Third and fourth central moments, pooled the same way as M2, and the
        # running extremes. The advisor reads skewness, kurtosis, min and max;
        # none of them were ever collected, so every column arrived at it with
        # the neutral defaults -- skew 0, kurtosis 3 -- and was reported as
        # "Normal distribution detected" whatever shape it had, while the
        # rescaling factor derived from min and max always came out as 1.
        self.M3 = tf.Variable(
            0.0,
            dtype=tf.float64,
            trainable=False,
        )
        self.M4 = tf.Variable(
            0.0,
            dtype=tf.float64,
            trainable=False,
        )
        self.minimum = tf.Variable(
            float("inf"),
            dtype=tf.float64,
            trainable=False,
        )
        self.maximum = tf.Variable(
            float("-inf"),
            dtype=tf.float64,
            trainable=False,
        )
        self.var = tf.Variable(
            0.0,
            dtype=tf.float64,
            trainable=False,
        )

    @tf.function
    def update(self, values: tf.Tensor) -> None:
        """Pool a batch of values into the running mean and variance.

        Args:
            values: The new values to add to the accumulators.
        """
        values = tf.cast(tf.reshape(values, [-1]), tf.float64)
        batch_count = tf.cast(tf.size(values), tf.float64)

        # An empty batch has to leave everything untouched, and both divisors
        # below can be zero when it does. Flooring them at one keeps the
        # arithmetic finite; every numerator carries `batch_count`, so an empty
        # batch contributes nothing either way.
        safe_batch_count = tf.maximum(batch_count, 1.0)
        batch_mean = tf.reduce_sum(values) / safe_batch_count
        centered = values - batch_mean
        batch_m2 = tf.reduce_sum(tf.square(centered))
        batch_m3 = tf.reduce_sum(centered * tf.square(centered))
        batch_m4 = tf.reduce_sum(tf.square(tf.square(centered)))

        # The identity element rather than a branch: an empty batch reduces to
        # the infinity that leaves the running extreme alone.
        infinity = tf.constant([float("inf")], dtype=tf.float64)
        batch_min = tf.reduce_min(tf.concat([values, infinity], axis=0))
        batch_max = tf.reduce_max(tf.concat([values, -infinity], axis=0))

        self._pool(
            count=batch_count,
            mean=batch_mean,
            m2=batch_m2,
            m3=batch_m3,
            m4=batch_m4,
            minimum=batch_min,
            maximum=batch_max,
        )

    def _pool(
        self,
        count: tf.Tensor,
        mean: tf.Tensor,
        m2: tf.Tensor,
        m3: tf.Tensor,
        m4: tf.Tensor,
        minimum: tf.Tensor,
        maximum: tf.Tensor,
    ) -> None:
        """Fold a summarized group of values into the running accumulators.

        The pooled forms of the Chan-Golub-LeVeque update, carried up to the
        fourth moment. Both `update` and `merge` reduce to the same arithmetic.

        Args:
            count: How many values the group holds.
            mean: Their mean.
            m2: The sum of their squared deviations from that mean.
            m3: The sum of their cubed deviations.
            m4: The sum of their fourth-power deviations.
            minimum: The smallest of them, or +inf for an empty group.
            maximum: The largest of them, or -inf for an empty group.
        """
        total = self.n + count
        safe_total = tf.maximum(total, 1.0)
        delta = mean - self.mean
        held = self.n

        self.M4.assign(
            self.M4
            + m4
            + tf.square(tf.square(delta))
            * held
            * count
            * (held * held - held * count + count * count)
            / (safe_total * tf.square(safe_total))
            + 6.0
            * delta
            * delta
            * (held * held * m2 + count * count * self.M2)
            / tf.square(safe_total)
            + 4.0 * delta * (held * m3 - count * self.M3) / safe_total,
        )
        self.M3.assign(
            self.M3
            + m3
            + delta
            * tf.square(delta)
            * held
            * count
            * (held - count)
            / tf.square(safe_total)
            + 3.0 * delta * (held * m2 - count * self.M2) / safe_total,
        )
        self.M2.assign(self.M2 + m2 + delta * delta * held * count / safe_total)
        self.mean.assign(self.mean + delta * count / safe_total)
        self.minimum.assign(tf.minimum(self.minimum, minimum))
        self.maximum.assign(tf.maximum(self.maximum, maximum))
        self.n.assign(total)

    def merge(self, other: "WelfordAccumulator") -> None:
        """Pool another accumulator's values into this one.

        Args:
            other: The accumulator whose values are being folded in.
        """
        self._pool(
            count=tf.cast(other.n, tf.float64),
            mean=tf.cast(other.mean, tf.float64),
            m2=tf.cast(other.M2, tf.float64),
            m3=tf.cast(other.M3, tf.float64),
            m4=tf.cast(other.M4, tf.float64),
            minimum=tf.cast(other.minimum, tf.float64),
            maximum=tf.cast(other.maximum, tf.float64),
        )

    @property
    def variance(self) -> float:
        """Returns the variance of the accumulated values."""
        return self.M2 / (self.n - 1) if self.n > 1 else self.var

    @property
    def skewness(self) -> float:
        """How lopsided the values are: 0 when symmetric, positive to the right.

        Returns:
            The population (Fisher-Pearson) skewness, or 0.0 when there are too
            few values, or no spread, to have one.
        """
        if self.n < 3 or self.M2 <= 0:
            return tf.constant(0.0, dtype=tf.float64)
        return tf.sqrt(self.n) * self.M3 / tf.pow(self.M2, 1.5)

    @property
    def kurtosis(self) -> float:
        """How heavy the tails are, on the scale where a normal curve is 3.

        Returns:
            The population kurtosis, or 3.0 when there are too few values, or
            no spread, to have one.
        """
        if self.n < 4 or self.M2 <= 0:
            return tf.constant(3.0, dtype=tf.float64)
        return self.n * self.M4 / tf.square(self.M2)

    @property
    def smallest(self) -> float:
        """The smallest value seen, or 0.0 before any has been."""
        if self.n < 1:
            return tf.constant(0.0, dtype=tf.float64)
        return self.minimum.value()

    @property
    def largest(self) -> float:
        """The largest value seen, or 0.0 before any has been."""
        if self.n < 1:
            return tf.constant(0.0, dtype=tf.float64)
        return self.maximum.value()

    @property
    def count(self) -> int:
        """Returns the number of accumulated values."""
        return self.n


class CategoricalAccumulator:
    def __init__(self) -> None:
        """Initializes the accumulator for categorical values."""
        # Using a single accumulator since tf.string can hold both strings and bytes
        self.values = tf.Variable(
            [],
            dtype=tf.string,
            shape=tf.TensorShape(None),
            trainable=False,
        )
        self.int_values = tf.Variable(
            [],
            dtype=tf.int32,
            shape=tf.TensorShape(None),
            trainable=False,
        )

    @tf.function
    def update(self, new_values: tf.Tensor) -> None:
        """Updates the accumulator with new categorical values.

        Args:
            new_values: The new categorical values to add to the accumulator.
        """
        if new_values.dtype == tf.string:
            updated_values = tf.unique(tf.concat([self.values, new_values], axis=0))[0]
            self.values.assign(updated_values)
        elif new_values.dtype == tf.int32:
            updated_values = tf.unique(
                tf.concat([self.int_values, new_values], axis=0),
            )[0]
            self.int_values.assign(updated_values)
        else:
            raise ValueError(
                f"Unsupported data type for categorical features: {new_values.dtype}",
            )

    def get_unique_values(self) -> list:
        """Returns the unique categorical values accumulated so far."""
        all_values = tf.concat(
            [self.values, tf.strings.as_string(self.int_values)],
            axis=0,
        )
        return tf.unique(all_values)[0].numpy().tolist()


# The punctuation `TextVectorization` removes under its default
# `standardize="lower_and_strip_punctuation"`, copied from that layer so the
# vocabulary collected here is spelled the way the layer will look it up.
_KERAS_PUNCTUATION = r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']'


class TextAccumulator:
    def __init__(self) -> None:
        """Initializes the accumulator for text values, where each entry is a list of words separated by spaces.

        Attributes:
            words (tf.Variable): TensorFlow variable to store unique words as strings.
        """
        self.words = tf.Variable(
            [],
            dtype=tf.string,
            shape=tf.TensorShape(None),
            trainable=False,
        )
        logger.info("TextAccumulator initialized.")

    @tf.function
    def update(self, new_texts: tf.Tensor) -> None:
        """Updates the accumulator with new text values, extracting words and accumulating unique ones.

        Args:
            new_texts: A batch of text values (tf.Tensor of dtype tf.string),
            each entry containing words separated by spaces.

        Raises:
            ValueError: If the input tensor is not of dtype tf.string.
        """
        if new_texts.dtype != tf.string:
            raise ValueError(
                f"Unsupported data type for text features: {new_texts.dtype}",
            )

        # Tokenize exactly the way `TextVectorization` will, because this
        # vocabulary is handed to that layer verbatim. Its default
        # standardization lowercases and then strips punctuation before
        # splitting on whitespace; this accumulator lowercased but kept the
        # punctuation, so a column of ordinary prose produced a vocabulary of
        # "product," and "it!" while the layer looked up "product" and "it".
        # The width was right and the counts were right; the words simply were
        # not there, and a large share of every sentence fell into the
        # out-of-vocabulary slot.
        new_texts = tf.strings.lower(new_texts)
        new_texts = tf.strings.regex_replace(new_texts, _KERAS_PUNCTUATION, "")
        split_words = tf.strings.split(new_texts).flat_values

        # Concatenate new words with existing words and update unique words
        updated_words = tf.unique(tf.concat([self.words, split_words], axis=0))[0]
        self.words.assign(updated_words)

    def get_unique_words(self) -> list:
        """Returns the unique words accumulated so far as a list of strings.

        Returns:
            list of str: Unique words accumulated.
        """
        # `.numpy().tolist()` yields `bytes`; decode so the vocabulary is made of
        # real `str` values. Keeping `bytes` here breaks Keras serialization of the
        # TextVectorization layer and corrupts the vocabulary when the stats are
        # round-tripped through JSON.
        return [word.decode("utf-8") for word in self.words.value().numpy().tolist()]


class DateAccumulator:
    """Accumulator for computing statistics of date features including cyclical encoding."""

    def __init__(self):
        """Initializes the accumulators for date features."""
        # For year, month, and day of the week
        self.year_accumulator = WelfordAccumulator()
        self.month_sin_accumulator = WelfordAccumulator()
        self.month_cos_accumulator = WelfordAccumulator()
        self.day_of_week_sin_accumulator = WelfordAccumulator()
        self.day_of_week_cos_accumulator = WelfordAccumulator()

    def update(self, dates: tf.Tensor) -> None:
        """Updates the accumulators with new date values.

        Args:
            dates: Either a tensor of date strings (``YYYY-MM-DD`` or ``YYYY/MM/DD``),
                as read from a CSV column, or an already parsed numeric tensor of
                shape ``[batch_size, >=4]`` whose columns are
                ``[year, month, day_of_month, day_of_week]``.

        Raises:
            ValueError: If a parsed numeric tensor does not carry the four
                expected date components.
        """
        if dates.dtype == tf.string:
            parsed = DateParsingLayer()(tf.reshape(dates, [-1, 1]))
        else:
            parsed = tf.convert_to_tensor(dates)
            if parsed.shape.rank != 2 or parsed.shape[-1] < 4:
                raise ValueError(
                    "Parsed date tensors must have shape [batch_size, >=4] with "
                    "columns [year, month, day_of_month, day_of_week], got shape "
                    f"{parsed.shape}.",
                )

        parsed = tf.cast(parsed, tf.float32)
        year = parsed[:, 0]
        month = parsed[:, 1]
        day_of_week = parsed[:, 3]

        # Cyclical encoding
        pi = tf.constant(math.pi, dtype=tf.float32)
        month_sin = tf.math.sin(2 * pi * month / 12)
        month_cos = tf.math.cos(2 * pi * month / 12)
        day_of_week_sin = tf.math.sin(2 * pi * day_of_week / 7)
        day_of_week_cos = tf.math.cos(2 * pi * day_of_week / 7)

        self.year_accumulator.update(year)
        self.month_sin_accumulator.update(month_sin)
        self.month_cos_accumulator.update(month_cos)
        self.day_of_week_sin_accumulator.update(day_of_week_sin)
        self.day_of_week_cos_accumulator.update(day_of_week_cos)

    def mean(self) -> dict:
        """Returns the mean statistics for date features."""
        return {
            "year": self.year_accumulator.mean.numpy(),
            "month_sin": self.month_sin_accumulator.mean.numpy(),
            "month_cos": self.month_cos_accumulator.mean.numpy(),
            "day_of_week_sin": self.day_of_week_sin_accumulator.mean.numpy(),
            "day_of_week_cos": self.day_of_week_cos_accumulator.mean.numpy(),
        }

    def variance(self) -> dict:
        """Returns the variance statistics for date features."""
        return {
            "year": self.year_accumulator.variance.numpy(),
            "month_sin": self.month_sin_accumulator.variance.numpy(),
            "month_cos": self.month_cos_accumulator.variance.numpy(),
            "day_of_week_sin": self.day_of_week_sin_accumulator.variance.numpy(),
            "day_of_week_cos": self.day_of_week_cos_accumulator.variance.numpy(),
        }


def _warn_if_float32_flattens_the_column(
    feature: str,
    mean: float,
    variance: float,
) -> None:
    """Say so when a column's spread is too fine for float32 to hold.

    Everything downstream -- the CSV reader, the model's inputs, the
    normalization layer -- is float32, which carries about seven significant
    digits. A column whose values sit far from zero but vary only slightly
    loses that variation on the way in, before any statistic is computed and
    before any layer sees it. Unix timestamps in seconds are the usual case:
    values near 1.6e9 are 128 apart in float32, so a column spread over a
    minute arrives as two or three distinct numbers. Normalization then works
    perfectly on what is left, and the feature reaching the model is a
    constant.

    Nothing raises, because the numbers are the ones the file actually
    contains. This only makes the loss visible.

    Args:
        feature: The column's name, for the message.
        mean: The column's mean.
        variance: The column's variance.
    """
    mean = float(mean)
    variance = float(variance)
    if not math.isfinite(mean) or not math.isfinite(variance) or variance <= 0.0:
        return

    # The gap between neighbouring float32 values at this magnitude.
    resolution = float(np.spacing(np.float32(abs(mean))))
    spread = math.sqrt(variance)
    if resolution <= 0.0 or spread >= 16.0 * resolution:
        return

    levels = max(int(6.0 * spread / resolution), 1)
    logger.warning(
        f"Feature '{feature}' has values around {mean:.6g} varying by only "
        f"{spread:.6g}, which float32 cannot hold: neighbouring values at that "
        f"magnitude are {resolution:.6g} apart, so the column arrives as about "
        f"{levels} distinct value(s) and carries almost no information. "
        f"Subtract a reference point before training (for a Unix timestamp, "
        f"the start of your data), or declare the column as a DATE feature.",
    )


class DatasetStatistics:
    def __init__(
        self,
        path_data: str,
        features_specs: dict[str, FeatureType | str] = None,
        numeric_features: list[NumericalFeature] = None,
        categorical_features: list[CategoricalFeature] = None,
        text_features: list[CategoricalFeature] = None,
        date_features: list[str] = None,
        time_series_features: list[TimeSeriesFeature] = None,
        features_stats_path: Path = None,
        overwrite_stats: bool = False,
        batch_size: int = 50_000,
    ) -> None:
        """Initializes the statistics accumulators for numeric, categorical, text, and date features.

        Args:
            path_data: Path to the folder containing the CSV files.
            batch_size: The batch size to use when reading data from the dataset.
            features_stats_path: Path to the features statistics JSON file (defaults to None).
            overwrite_stats: Whether or not to overwrite existing statistics file (defaults to False).
            features_specs:
                A dictionary mapping feature names to feature specifications (defaults to None).
                Easier alternative to providing numerical and categorical lists.
            numeric_features: A list of numerical features to calculate statistics for (defaults to None).
            categorical_features: A list of categorical features to calculate statistics for (defaults to None).
            text_features: A list of text features to calculate statistics for (defaults to None).
            date_features: A list of date features to calculate statistics for (defaults to None).
            time_series_features: A list of time series features to calculate statistics for (defaults to None).
        """
        self.path_data = path_data
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.text_features = text_features or []
        self.date_features = date_features or []
        self.time_series_features = time_series_features or []
        self.features_specs = features_specs or {}

        # `features_specs` is documented as the easier alternative to listing the
        # features by type, but nothing derived those lists from it: an instance
        # built that way created no accumulators and `main()` returned {}, which
        # is what made `auto_configure` produce empty recommendations.
        if self.features_specs and not any(
            (
                self.numeric_features,
                self.categorical_features,
                self.text_features,
                self.date_features,
                self.time_series_features,
            ),
        ):
            converter = FeatureSpaceConverter()
            self.features_specs = converter._init_features_specs(self.features_specs)
            self.numeric_features = converter.numeric_features
            self.categorical_features = converter.categorical_features
            self.text_features = converter.text_features
            self.date_features = converter.date_features
            self.time_series_features = converter.time_series_features
        self.features_stats_path = features_stats_path or "features_stats.json"
        self.overwrite_stats = overwrite_stats
        self.batch_size = batch_size
        self.features_stats = {}

        # Initializing placeholders for statistics
        self.numeric_stats = {
            col: WelfordAccumulator() for col in self.numeric_features
        }
        self.categorical_stats = {
            col: CategoricalAccumulator() for col in self.categorical_features
        }
        self.text_stats = {col: TextAccumulator() for col in self.text_features}
        self.date_stats = {col: DateAccumulator() for col in self.date_features}
        self.time_series_stats = {}

    @staticmethod
    def _get_csv_file_pattern(path) -> str:
        """Get the csv file pattern that will handle directories and file paths.

        Nothing here depends on the instance, and `PreprocessingModel` needs the
        same resolution to adapt a text vectorizer on the data.

        Args:
            path (str): Path to the csv file (can be a directory or a file)

        Returns:
            str: File pattern that always has *.csv at the end

        """
        if path is None:
            raise ValueError(
                "`path_data` is required to compute statistics but was None. "
                "Pass a path to a CSV file or to a directory of CSV files.",
            )
        if not isinstance(path, str | os.PathLike):
            raise TypeError(
                "`path_data` must be a path to CSV data (a str or PathLike), "
                f"got {type(path).__name__}. In-memory frames are not read "
                "directly -- write the data to a CSV first and pass its path.",
            )

        file_path = Path(path)
        # A path that names a file means that file. It used to be replaced with
        # `<parent>/*.csv`, so `path_data="data/train.csv"` computed statistics
        # over every CSV sitting beside it -- `test.csv` included. That is the
        # usage every example in the documentation shows, and the leak was
        # silent: the numbers came back, just drawn from the wrong rows.
        if file_path.is_file():
            return str(file_path)
        if file_path.is_dir():
            return str(file_path / "*.csv")

        # Neither, so it is either a glob the caller wrote themselves
        # ("data/*.csv", "shard-*.csv") or a path that does not exist. A glob is
        # passed through; anything else is reported here rather than inside
        # `tf.data`, which answers a missing file with an empty dataset and
        # statistics quietly computed from nothing.
        if any(character in str(file_path) for character in "*?["):
            return str(file_path)
        if file_path.suffix:
            raise FileNotFoundError(
                f"`path_data` points at {str(file_path)!r}, which does not "
                "exist. Pass a CSV file, a directory of CSV files, or a glob.",
            )
        raise FileNotFoundError(
            f"`path_data` points at {str(file_path)!r}, which is not a file or "
            "a directory. Pass a CSV file, a directory of CSV files, or a glob.",
        )

    def _read_data_into_dataset(self) -> tf.data.Dataset:
        """Reading CSV files from the provided path into a tf.data.Dataset."""
        logger.info(f"Reading CSV data from the corresponding folder: {self.path_data}")
        _path_csvs_regex = self._get_csv_file_pattern(path=self.path_data)
        self.ds = tf.data.experimental.make_csv_dataset(
            file_pattern=_path_csvs_regex,
            num_epochs=1,
            shuffle=False,
            ignore_errors=True,
            batch_size=self.batch_size,
        )
        logger.info(f"DataSet Ready to be used (batched by: {self.batch_size}) ")
        return self.ds

    def _process_numeric_feature(self, feature: str, batch: tf.Tensor) -> None:
        """Process a single numeric feature from a batch.

        Args:
            feature: Feature name
            batch: Batch of data
        """
        self.numeric_stats[feature].update(batch[feature])

    def _process_categorical_feature(self, feature: str, batch: tf.Tensor) -> None:
        """Process a single categorical feature from a batch.

        Args:
            feature: Feature name
            batch: Batch of data
        """
        self.categorical_stats[feature].update(batch[feature])

    def _process_text_feature(self, feature: str, batch: tf.Tensor) -> None:
        """Process a single text feature from a batch.

        Args:
            feature: Feature name
            batch: Batch of data
        """
        self.text_stats[feature].update(batch[feature])

    def _process_date_feature(self, feature: str, batch: tf.Tensor) -> None:
        """Process a single date feature from a batch.

        Args:
            feature: Feature name
            batch: Batch of data
        """
        self.date_stats[feature].update(batch[feature])

    def _process_time_series_data(self) -> dict:
        """Process time series data, including sorting and grouping using TensorFlow dataset API.

        Returns:
            dict: Dictionary of processed time series features and their statistics
        """
        if not self.time_series_features and not any(
            isinstance(feature, TimeSeriesFeature)
            for feature in self.features_specs.values()
        ):
            return {}

        # Extract time series features from specs if not provided directly
        if not self.time_series_features and self.features_specs:
            self.time_series_features = [
                feature_name
                for feature_name, feature in self.features_specs.items()
                if isinstance(feature, TimeSeriesFeature)
                or (
                    hasattr(feature, "feature_type")
                    and feature.feature_type == FeatureType.TIME_SERIES
                )
            ]

        if not self.time_series_features:
            return {}

        # Read CSV files into TensorFlow dataset
        dataset = self._read_data_into_dataset()
        time_series_stats = {}

        # Process each time series feature
        for feature_name in self.time_series_features:
            feature = self.features_specs.get(feature_name)

            if not feature or not isinstance(feature, TimeSeriesFeature):
                continue

            # Check if the feature exists in the dataset
            has_feature = False
            for batch in dataset.take(1):
                has_feature = feature_name in batch
                break

            if not has_feature:
                logger.warning(
                    f"Feature '{feature_name}' not found in the dataset. Skipping statistics calculation.",
                )
                continue

            # A calendar feature's column holds dates, not numbers: everything
            # it produces is derived from the string by `CalendarFeatureLayer`,
            # and there is no mean or variance to take. Feeding it to the
            # accumulator anyway killed the statistics pass with "Cast string to
            # double is not supported" from inside a `tf.function`, so the
            # documented way of declaring calendar features could not be built
            # at all.
            if feature.dtype == tf.string:
                logger.debug(
                    f"'{feature_name}' is a calendar feature; its column holds "
                    f"dates, so there are no numeric statistics to collect.",
                )
                continue

            # Prepare for grouped processing if grouping is specified
            if feature.group_by and feature.group_by in list(
                dataset.element_spec.keys(),
            ):
                # Process data by groups
                group_data = {}

                # Extract data for each group
                for batch in dataset:
                    if feature_name in batch and feature.group_by in batch:
                        group_keys = batch[feature.group_by].numpy()
                        feature_values = batch[feature_name].numpy()
                        sort_keys = (
                            batch[feature.sort_by].numpy()
                            if feature.sort_by in batch
                            else None
                        )

                        # Organize data by group
                        for i in range(len(group_keys)):
                            group_key = group_keys[i]
                            # Convert bytes to string if necessary
                            if isinstance(group_key, bytes):
                                group_key = group_key.decode("utf-8")

                            if group_key not in group_data:
                                group_data[group_key] = []

                            if sort_keys is not None:
                                group_data[group_key].append(
                                    (sort_keys[i], feature_values[i]),
                                )
                            else:
                                group_data[group_key].append(
                                    (i, feature_values[i]),
                                )  # Use index as sort key

                # Create a separate accumulator for each group and process them
                group_accumulators = {}

                for group_key, pairs in group_data.items():
                    # Sort if sort_by is specified
                    if feature.sort_by:
                        pairs.sort(
                            key=lambda x: x[0],
                            reverse=not feature.sort_ascending,
                        )

                    # Extract sorted values
                    sorted_values = [pair[1] for pair in pairs]

                    if sorted_values:
                        # Create accumulator for this group
                        accumulator = WelfordAccumulator()
                        sorted_tensor = tf.constant(sorted_values, dtype=tf.float32)
                        accumulator.update(sorted_tensor)
                        group_accumulators[group_key] = accumulator

                # Combine statistics across groups. Each group is pooled in
                # whole. Replacing a group with `count` copies of its own mean
                # -- what this did before -- throws away everything the group
                # varied by, leaving only the variance *between* the group
                # means: two groups of spread 25 whose means happened to be
                # close reported a variance of 0.73 against a true 643.
                if group_accumulators:
                    combined_accumulator = WelfordAccumulator()
                    for accumulator in group_accumulators.values():
                        combined_accumulator.merge(accumulator)

                    # Calculate and store overall statistics
                    stats = {
                        "mean": float(combined_accumulator.mean.numpy()),
                        "var": float(combined_accumulator.variance.numpy()),
                        "min": float(combined_accumulator.smallest.numpy()),
                        "max": float(combined_accumulator.largest.numpy()),
                        "skewness": float(combined_accumulator.skewness.numpy()),
                        "kurtosis": float(combined_accumulator.kurtosis.numpy()),
                        "count": int(
                            sum(
                                acc.count.numpy() for acc in group_accumulators.values()
                            ),
                        ),
                        "dtype": feature.dtype.name
                        if hasattr(feature.dtype, "name")
                        else str(feature.dtype),
                        "sort_by": feature.sort_by,
                        "sort_ascending": feature.sort_ascending,
                        "group_by": feature.group_by,
                        "num_groups": len(group_accumulators),
                    }

                    time_series_stats[feature_name] = stats
            else:
                # No grouping - process the entire dataset
                accumulator = WelfordAccumulator()

                if feature.sort_by and feature.sort_by in list(
                    dataset.element_spec.keys(),
                ):
                    # Process in a streaming fashion to avoid memory issues
                    # Create buffer for sorting that can be processed in chunks
                    buffer_size = 10000  # Adjust based on memory availability
                    buffer = []

                    for batch in dataset:
                        if feature_name in batch and feature.sort_by in batch:
                            sort_keys = batch[feature.sort_by].numpy()
                            feature_values = batch[feature_name].numpy()

                            # Add batch data to buffer
                            for i in range(len(sort_keys)):
                                buffer.append((sort_keys[i], feature_values[i]))

                            # Process buffer when it gets full
                            if len(buffer) >= buffer_size:
                                # Sort buffer
                                buffer.sort(
                                    key=lambda x: x[0],
                                    reverse=not feature.sort_ascending,
                                )

                                # Extract values and update accumulator
                                sorted_values = [pair[1] for pair in buffer]
                                sorted_tensor = tf.constant(
                                    sorted_values,
                                    dtype=tf.float32,
                                )
                                accumulator.update(sorted_tensor)

                                # Clear buffer
                                buffer = []

                    # Process any remaining items in buffer
                    if buffer:
                        buffer.sort(
                            key=lambda x: x[0],
                            reverse=not feature.sort_ascending,
                        )
                        sorted_values = [pair[1] for pair in buffer]
                        sorted_tensor = tf.constant(sorted_values, dtype=tf.float32)
                        accumulator.update(sorted_tensor)
                else:
                    # If no sorting needed, just accumulate statistics directly
                    for batch in dataset:
                        if feature_name in batch:
                            accumulator.update(batch[feature_name])

                # Calculate statistics
                stats = {
                    "mean": float(accumulator.mean.numpy()),
                    "var": float(accumulator.variance.numpy()),
                    "min": float(accumulator.smallest.numpy()),
                    "max": float(accumulator.largest.numpy()),
                    "skewness": float(accumulator.skewness.numpy()),
                    "kurtosis": float(accumulator.kurtosis.numpy()),
                    "count": int(accumulator.count.numpy()),
                    "dtype": feature.dtype.name
                    if hasattr(feature.dtype, "name")
                    else str(feature.dtype),
                    "sort_by": feature.sort_by,
                    "sort_ascending": feature.sort_ascending,
                    "group_by": feature.group_by,
                }

                time_series_stats[feature_name] = stats

        return time_series_stats

    def _process_batch_parallel(self, batch: tf.Tensor) -> None:
        """Process a batch of data in parallel using ThreadPoolExecutor.

        Args:
            batch: Batch of data to process
        """
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []

            # Submit numeric feature processing tasks
            for feature in self.numeric_features:
                futures.append(
                    executor.submit(self._process_numeric_feature, feature, batch),
                )

            # Submit categorical feature processing tasks
            for feature in self.categorical_features:
                futures.append(
                    executor.submit(self._process_categorical_feature, feature, batch),
                )

            # Submit text feature processing tasks
            for feature in self.text_features:
                futures.append(
                    executor.submit(self._process_text_feature, feature, batch),
                )

            # Submit date feature processing tasks
            for feature in self.date_features:
                futures.append(
                    executor.submit(self._process_date_feature, feature, batch),
                )

            # Submit time series feature processing tasks
            futures.append(
                executor.submit(self._process_time_series_data),
            )

            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing feature: {str(e)}")
                    raise

    def _compute_feature_stats_parallel(
        self,
        feature_type: str,
        features: list[str],
    ) -> dict[str, Any]:
        """Compute statistics for a group of features in parallel.

        Args:
            feature_type: Type of features (numeric, categorical, text, or date)
            features: List of feature names

        Returns:
            Dictionary containing computed statistics
        """

        def compute_feature_stats(feature: str) -> tuple[str, dict]:
            """Compute statistics for a single feature.

            Args:
                feature: Name of the feature to compute statistics for

            Returns:
                tuple: A tuple containing:
                    - str: Feature name
                    - dict: Dictionary of computed statistics for the feature

            The computed statistics vary based on the feature_type:
                - numeric: mean, count, variance, and dtype
                - categorical: size of vocabulary, unique values, and dtype
                - text: vocabulary size, unique words, sequence length, and dtype
                - date: mean and variance for each date component
            """
            if feature_type == "numeric":
                accumulator = self.numeric_stats[feature]
                mean = accumulator.mean.numpy()
                variance = accumulator.variance.numpy()
                _warn_if_float32_flattens_the_column(feature, mean, variance)
                return feature, {
                    "mean": mean,
                    "count": accumulator.count.numpy(),
                    "var": variance,
                    # The advisor reads these four to tell one distribution
                    # from another and to size a rescaling factor. None of them
                    # were collected, so it read the neutral defaults and
                    # called every column normal.
                    "min": float(accumulator.smallest.numpy()),
                    "max": float(accumulator.largest.numpy()),
                    "skewness": float(accumulator.skewness.numpy()),
                    "kurtosis": float(accumulator.kurtosis.numpy()),
                    "dtype": self.features_specs[feature].dtype,
                }
            elif feature_type == "categorical":
                _dtype = self.features_specs[feature].dtype
                if _dtype == tf.int32:
                    unique_values = [
                        int(_byte)
                        for _byte in self.categorical_stats[feature].get_unique_values()
                    ]
                    unique_values.sort()
                else:
                    _unique_values = self.categorical_stats[feature].get_unique_values()
                    unique_values = [
                        (_byte).decode("utf-8") for _byte in _unique_values
                    ]
                return feature, {
                    "size": len(unique_values),
                    "vocab": unique_values,
                    "dtype": _dtype,
                }
            elif feature_type == "text":
                unique_words = self.text_stats[feature].get_unique_words()
                return feature, {
                    "size": len(unique_words),
                    "vocab": unique_words,
                    "sequence_length": 100,
                    "vocab_size": min(10000, len(unique_words)),
                    "dtype": tf.string,
                }
            elif feature_type == "date":
                _means_data: dict = self.date_stats[feature].mean()
                _vars_data: dict = self.date_stats[feature].variance()
                date_stats = {}
                for feat_name in _means_data:
                    date_stats[f"mean_{feat_name}"] = _means_data[feat_name]
                    date_stats[f"var_{feat_name}"] = _vars_data[feat_name]
                return feature, date_stats

            return feature, {}  # Default empty stats for unknown feature types

        stats = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks to the executor
            future_to_feature = {
                executor.submit(compute_feature_stats, feature): feature
                for feature in features
            }

            # Collect results as they complete
            for future in as_completed(future_to_feature):
                feature_name, feature_stats = future.result()
                stats[feature_name] = feature_stats

        return stats

    def _compute_final_statistics(self) -> dict[str, dict]:
        """Compute the final statistics for all features.

        Returns:
            Dictionary containing the computed statistics for all features
        """
        logger.info("Computing final statistics")
        stats = {}

        # Compute numeric statistics
        if self.numeric_features:
            stats["numeric_stats"] = self._compute_feature_stats_parallel(
                "numeric",
                self.numeric_features,
            )

        # Compute categorical statistics
        if self.categorical_features:
            stats["categorical_stats"] = self._compute_feature_stats_parallel(
                "categorical",
                self.categorical_features,
            )

        # Compute text statistics
        if self.text_features:
            stats["text"] = self._compute_feature_stats_parallel(
                "text",
                self.text_features,
            )

        # Compute date statistics
        if self.date_features:
            stats["date"] = self._compute_feature_stats_parallel(
                "date",
                self.date_features,
            )

        # Compute time series statistics
        time_series_stats = self._process_time_series_data()
        if time_series_stats:
            stats["time_series"] = time_series_stats

        # Store the computed statistics
        self.features_stats = stats
        return stats

    def calculate_dataset_statistics(self, dataset: tf.data.Dataset) -> dict[str, dict]:
        """Calculate the statistics of the dataset.

        Args:
            dataset: The dataset to calculate statistics for.

        Returns:
            Dictionary containing the computed statistics
        """
        logger.info("Calculating dataset statistics")
        for batch in dataset:
            self._process_batch_parallel(batch)

        self.features_stats = self._compute_final_statistics()
        return self.features_stats

    @staticmethod
    def _custom_serializer(obj) -> Any:
        """Custom JSON serializer for objects not serializable by default json code."""
        if isinstance(obj, tf.dtypes.DType):
            return obj.name  # Convert dtype to its string representation
        elif isinstance(obj, np.integer):
            return int(obj)  # Convert numpy int to Python int
        elif isinstance(obj, np.floating):
            return float(obj)  # Convert numpy float to Python float
        elif isinstance(obj, bytes):
            # `str(b"foo")` would serialize the *repr* ("b'foo'") and silently
            # corrupt vocabularies on reload, so decode explicitly.
            return obj.decode("utf-8")
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        logger.debug(f"Type {type(obj)} is not serializable")
        raise TypeError("Type not serializable")

    def _save_stats(self) -> None:
        """Saving feature stats locally."""
        logger.info(f"Saving feature stats locally to: {self.features_stats_path}")

        # Convert the string path to a Path object before calling open
        path_obj = Path(self.features_stats_path)
        with path_obj.open("w") as f:
            json.dump(self.features_stats, f, default=self._custom_serializer)
        logger.info("features_stats saved ")

    @staticmethod
    def _has_repr_encoded_vocabulary(features_stats: dict) -> bool:
        """Detect vocabularies saved as Python byte reprs by older releases.

        Args:
            features_stats: Statistics loaded from disk.

        Returns:
            True when any vocabulary entry looks like ``b'value'`` rather than
            the value itself.
        """
        for stats_group in features_stats.values():
            if not isinstance(stats_group, dict):
                continue
            for feature_stats in stats_group.values():
                if not isinstance(feature_stats, dict):
                    continue
                vocabulary = feature_stats.get("vocab")
                if not isinstance(vocabulary, list):
                    continue
                for entry in vocabulary:
                    if (
                        isinstance(entry, str)
                        and len(entry) >= 3
                        and entry.startswith(("b'", 'b"'))
                        and entry[-1] == entry[1]
                    ):
                        return True
        return False

    def _load_stats(self) -> dict:
        """Loads serialized features stats from a file, with custom handling for TensorFlow dtypes.

        Returns:
            A dictionary containing the loaded features statistics.
        """
        if self.overwrite_stats:
            logger.info("overwrite_stats is currently active ")
            return {}

        stats_path = Path(self.features_stats_path)
        if stats_path.is_file():
            logger.info(
                f"Found columns statistics, loading as features_stats: {self.features_stats_path}",
            )
            with stats_path.open() as f:
                self.features_stats = json.load(f)

            # Statistics written before the bytes-decoding fix hold repr strings
            # ("b'paris'") instead of the real categories. Such a file still
            # loads and still builds a model, but every category then misses the
            # vocabulary and encodes identically to an unseen value -- silent,
            # total signal loss. Recompute instead of trusting it.
            if self._has_repr_encoded_vocabulary(self.features_stats):
                logger.warning(
                    f"{self.features_stats_path} was written by an older KDP "
                    "release whose vocabularies were stored as byte reprs "
                    "(\"b'value'\"). Reusing it would silently map every "
                    "category to the out-of-vocabulary slot, so the statistics "
                    "are being recomputed from the data.",
                )
                self.features_stats = {}
                return self.features_stats

            # Convert dtype strings back to TensorFlow dtype objects
            for stats_type in (
                self.features_stats.values()
            ):  # 'numeric_stats' and 'categorical_stats'
                for _, feature_stats in stats_type.items():
                    if "dtype" in feature_stats:
                        feature_stats["dtype"] = tf.dtypes.as_dtype(
                            feature_stats["dtype"],
                        )
            logger.info("features_stats loaded ")
        else:
            logger.info("No serialized features stats were detected ...")
            self.features_stats = {}
        return self.features_stats

    def main(self) -> dict:
        """Calculates and returns final statistics for the dataset.

        Returns:
            A dictionary containing the calculated statistics for the dataset.
        """
        ds = self._read_data_into_dataset()
        stats = self.calculate_dataset_statistics(dataset=ds)
        self._save_stats()
        return stats

    def recommend_model_configuration(self) -> dict:
        """Analyze the computed dataset statistics and provide recommendations for optimal preprocessing.

        This method leverages the ModelAdvisor to analyze feature characteristics and suggest
        the best preprocessing strategies, layer configurations, and model parameters.

        Returns:
            dict: A dictionary containing feature-specific and global recommendations
                 along with a ready-to-use code snippet.
        """
        # Import the ModelAdvisor here to avoid circular imports
        from kdp.model_advisor import recommend_model_configuration

        # Ensure we have statistics to analyze
        if not hasattr(self, "features_stats") or not self.features_stats:
            logger.warning("No statistics available. Calculating statistics first.")
            self.main()

        # Generate recommendations based on the computed statistics
        recommendations = recommend_model_configuration(self.features_stats)

        logger.info(
            "Generated model configuration recommendations based on dataset statistics",
        )
        logger.info(
            f"Recommended configuration for {len(recommendations.get('features', {}))} features",
        )

        return recommendations
