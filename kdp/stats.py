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
    """Accumulator for computing the mean and variance of a sequence of numbers
    using the Welford algorithm (streaming data).
    """

    def __init__(self):
        """Initializes the accumulators for the Welford algorithm."""
        self.n = tf.Variable(
            0.0,
            dtype=tf.float32,
            trainable=False,
        )
        self.mean = tf.Variable(
            0.0,
            dtype=tf.float32,
            trainable=False,
        )
        self.M2 = tf.Variable(
            0.0,
            dtype=tf.float32,
            trainable=False,
        )
        self.var = tf.Variable(
            0.0,
            dtype=tf.float32,
            trainable=False,
        )

    @tf.function
    def update(self, values: tf.Tensor) -> None:
        """Updates the accumulators with new values using the Welford algorithm.

        Args:
            values: The new values to add to the accumulators.
        """
        values = tf.cast(values, tf.float32)
        n = self.n + tf.cast(tf.size(values), tf.float32)
        delta = values - self.mean
        self.mean.assign(self.mean + tf.reduce_sum(delta / n))
        self.M2.assign(self.M2 + tf.reduce_sum(delta * (values - self.mean)))
        self.n.assign(n)

    @property
    def variance(self) -> float:
        """Returns the variance of the accumulated values."""
        return self.M2 / (self.n - 1) if self.n > 1 else self.var

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

        # Split each string into words and flatten the list
        new_texts = tf.strings.regex_replace(new_texts, r"\s+", " ")
        split_words = tf.strings.split(new_texts).flat_values
        split_words = tf.strings.lower(split_words)

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

    def _get_csv_file_pattern(self, path) -> str:
        """Get the csv file pattern that will handle directories and file paths.

        Args:
            path (str): Path to the csv file (can be a directory or a file)

        Returns:
            str: File pattern that always has *.csv at the end

        """
        file_path = Path(path)
        # Check if the path is a directory
        if file_path.suffix:
            # Get the parent directory if the path is a file
            base_path = file_path.parent
            csv_pattern = base_path / "*.csv"
        else:
            csv_pattern = file_path / "*.csv"

        return str(csv_pattern)

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

                # Combine statistics across groups
                if group_accumulators:
                    # Create overall accumulator to combine statistics
                    combined_accumulator = WelfordAccumulator()

                    # Combine all group means weighted by count
                    all_values = []
                    for _, acc in group_accumulators.items():
                        mean_tensor = (
                            tf.ones(shape=(int(acc.count.numpy()),), dtype=tf.float32)
                            * acc.mean.numpy()
                        )
                        all_values.append(mean_tensor)

                    if all_values:
                        combined_tensor = tf.concat(all_values, axis=0)
                        combined_accumulator.update(combined_tensor)

                    # Calculate and store overall statistics
                    stats = {
                        "mean": float(combined_accumulator.mean.numpy()),
                        "var": float(combined_accumulator.variance.numpy()),
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
                return feature, {
                    "mean": self.numeric_stats[feature].mean.numpy(),
                    "count": self.numeric_stats[feature].count.numpy(),
                    "var": self.numeric_stats[feature].variance.numpy(),
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
