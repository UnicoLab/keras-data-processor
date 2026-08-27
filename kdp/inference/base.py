import pandas as pd
import numpy as np
import tensorflow as tf


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
            # Infer the type from the values
            if (
                len(value) > 0
                and isinstance(value[0], int | float | np.number | type(None))
                or any(
                    isinstance(v, int | float | np.number) or pd.isna(v) for v in value
                )
            ):
                # Numerical features as float32
                tf_data[key] = tf.constant(value, dtype=tf.float32)
            else:
                # Everything else as string
                tf_data[key] = tf.constant(value)

        return tf_data
