import pandas as pd
import datetime

import numpy as np
import tensorflow as tf


def _is_missing(value) -> bool:
    """Report whether a single value stands for "no data".

    Args:
        value: A scalar taken from a feature column.

    Returns:
        True for None, NaN and NaT.
    """
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_text(value) -> str:
    """Render a value as the string the preprocessing graph expects.

    `str()` on a pandas Timestamp yields "2023-03-01 00:00:00". `DateParsingLayer`
    splits on the date separator and calls `tf.strings.to_number` on each part,
    so the trailing time turned the day into "01 00:00:00" and the graph failed
    with "Cast string to float is not supported". Dates are rendered in the
    ISO form the layer parses, which is also how they appear in a CSV.

    Args:
        value: The value to render.

    Returns:
        str: The value as text.
    """
    if isinstance(value, pd.Timestamp | datetime.datetime | datetime.date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    return str(value)


class InferenceFormatter:
    """Base class for formatting data for inference in various contexts.

    This class provides common functionality for converting data to the format
    required by preprocessors during inference, regardless of feature types.

    Subclasses should implement specific formatting logic for different types
    of features (time series, text, etc.).
    """

    def __init__(self, preprocessor):
        """Initialize the InferenceFormatter.

        Args:
            preprocessor: The trained preprocessor model to prepare data for
        """
        self.preprocessor = preprocessor

    def prepare_inference_data(
        self,
        data: dict | pd.DataFrame,
        to_tensors: bool = False,
    ) -> dict | dict[str, tf.Tensor]:
        """Prepare data for inference based on preprocessor requirements.

        Args:
            data: The data to make predictions on
            to_tensors: Whether to convert the output to TensorFlow tensors

        Returns:
            Dict with properly formatted data for inference, either as Python types or as TensorFlow tensors
        """
        # Convert inputs to consistent format
        inference_data = self._convert_to_dict(data)

        # Convert to tensors if requested
        if to_tensors:
            return self._convert_to_tensors(inference_data)

        return inference_data

    def _convert_to_dict(self, data: dict | pd.DataFrame) -> dict:
        """Convert data to dictionary format required by the preprocessor.

        Args:
            data: Input data as DataFrame or Dict

        Returns:
            Dict with data in the correct format
        """
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to dict of lists
            data_dict = {}
            for column in data.columns:
                data_dict[column] = data[column].tolist()
            return data_dict
        elif isinstance(data, dict):
            # Ensure all values are lists/arrays
            for key, value in data.items():
                if not isinstance(value, list | np.ndarray):
                    data[key] = [value]  # Convert single values to lists
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _convert_to_tensors(self, data: dict) -> dict[str, tf.Tensor]:
        """Convert dictionary data to TensorFlow tensors.

        Args:
            data: Dictionary of data

        Returns:
            Dictionary with the same keys but values as TensorFlow tensors
        """
        tf_data = {}
        for key, value in data.items():
            values = list(value)
            # Decide the column's type from the values that are actually there.
            # The previous check treated any column containing a missing value
            # as numeric, so a categorical column with one gap was handed to
            # `tf.constant(..., tf.float32)` and raised "mixed types".
            present = [item for item in values if not _is_missing(item)]
            is_numeric = bool(present) and all(
                isinstance(item, int | float | np.number) and not isinstance(item, bool)
                for item in present
            )

            if is_numeric:
                tf_data[key] = tf.constant(
                    [
                        float("nan") if _is_missing(item) else float(item)
                        for item in values
                    ],
                    dtype=tf.float32,
                )
            else:
                # Missing categories become the empty string, which every
                # vocabulary layer maps to its out-of-vocabulary slot.
                tf_data[key] = tf.constant(
                    ["" if _is_missing(item) else _as_text(item) for item in values],
                )

        return tf_data
