import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
from loguru import logger
import pandas as pd


@tf.keras.utils.register_keras_serializable(package="kdp.layers")
class CalendarFeatureLayer(Layer):
    """Layer for generating calendar features from date or timestamp inputs.

    This layer extracts calendar features like day of week, month, is_weekend,
    etc. from date or timestamp inputs. These features can help models
    learn seasonal patterns related to the calendar.

    Args:
        features: List of calendar features to extract. Options:
            - 'year': Year as a float
            - 'month': Month of year (1-12)
            - 'day': Day of month (1-31)
            - 'day_of_week': Day of week (0-6, 0 is Monday)
            - 'day_of_year': Day of year (1-366)
            - 'week_of_year': Week of year (1-53)
            - 'is_weekend': Binary indicator for weekend
            - 'quarter': Quarter of year (1-4)
            - 'is_month_start': Binary indicator for first day of month
            - 'is_month_end': Binary indicator for last day of month
            - 'is_quarter_start': Binary indicator for first day of quarter
            - 'is_quarter_end': Binary indicator for last day of quarter
            - 'is_year_start': Binary indicator for first day of year
            - 'is_year_end': Binary indicator for last day of year
            - 'month_sin': Sinusoidal encoding of month
            - 'month_cos': Cosinusoidal encoding of month
            - 'day_sin': Sinusoidal encoding of day of month
            - 'day_cos': Cosinusoidal encoding of day of month
            - 'day_of_week_sin': Sinusoidal encoding of day of week
            - 'day_of_week_cos': Cosinusoidal encoding of day of week
        cyclic_encoding: Whether to use sin/cos encoding for cyclic features
        input_format: Format of the input date string. Default is '%Y-%m-%d'.
        normalize: Whether to normalize numeric features to [0, 1] range.
        onehot_categorical: Whether to one-hot encode categorical features.
    """

    def __init__(
        self,
        features=None,
        cyclic_encoding=True,
        input_format="%Y-%m-%d",
        normalize=True,
        onehot_categorical=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Default features if none provided
        if features is None:
            self.features = [
                "month",
                "day",
                "day_of_week",
                "is_weekend",
                "month_sin",
                "month_cos",
                "day_of_week_sin",
                "day_of_week_cos",
            ]
        else:
            self.features = features

        self.cyclic_encoding = cyclic_encoding
        self.input_format = input_format
        self.normalize = normalize
        self.onehot_categorical = onehot_categorical

        # Define cyclic features for sin/cos encoding
        self.cyclic_features = {
            "month": 12,
            "day": 31,
            "day_of_week": 7,
            "day_of_year": 366,
            "week_of_year": 53,
            "quarter": 4,
            "hour": 24,
            "minute": 60,
            "second": 60,
        }

        # Validate features
        all_valid_features = list(self.cyclic_features.keys()) + [
            "year",
            "is_weekend",
            "is_month_start",
            "is_month_end",
            "is_quarter_start",
            "is_quarter_end",
            "is_year_start",
            "is_year_end",
        ]
        for feature in self.features:
            base_feature = (
                feature.split("_")[0]
                if "_sin" in feature or "_cos" in feature
                else feature
            )
            if (
                base_feature not in all_valid_features
                and feature not in all_valid_features
            ):
                raise ValueError(f"Invalid feature: {feature}")

    def build(self, input_shape) -> None:
        super().build(input_shape)

    def call(self, inputs, training=None) -> tf.Tensor:
        """Extract calendar features from date inputs.

        Args:
            inputs: Input tensor of shape (batch_size, 1) or (batch_size,) with date strings
            training: Boolean tensor indicating whether the call is for training

        Returns:
            Tensor with extracted calendar features
        """

        # Process date inputs using pandas for more flexibility
        def extract_calendar_features(date_inputs):
            # Convert tensor to numpy
            if isinstance(date_inputs, tf.Tensor):
                date_inputs = date_inputs.numpy()

            # Convert bytes to strings if needed
            if date_inputs.dtype.type is np.bytes_:
                date_inputs = np.array([s.decode("utf-8") for s in date_inputs])
            elif date_inputs.dtype.type is np.object_:
                # Handle case where numpy treats the array as object type but contains bytes
                try:
                    # Check if first element is bytes and convert all elements
                    if isinstance(date_inputs[0], bytes):
                        date_inputs = np.array(
                            [
                                s.decode("utf-8") if isinstance(s, bytes) else s
                                for s in date_inputs
                            ],
                        )
                    # Also handle case where strings are repr'd as bytes
                    elif isinstance(date_inputs[0], str) and date_inputs[0].startswith(
                        "b'",
                    ):
                        date_inputs = np.array(
                            [
                                s[2:-1] if s.startswith("b'") and s.endswith("'") else s
                                for s in date_inputs
                            ],
                        )
                except (IndexError, TypeError):
                    pass  # Handle empty arrays or arrays with mixed types

            # If input is rank 2 with shape (batch_size, 1), reshape to 1D
            if len(date_inputs.shape) == 2 and date_inputs.shape[1] == 1:
                date_inputs = date_inputs.reshape(-1)

            # Convert to pandas datetime
            try:
                dates = pd.to_datetime(date_inputs, format=self.input_format)
            except (ValueError, TypeError) as e:
                logger.debug(f"First conversion attempt failed: {e}")
                try:
                    # Try without specific format if the initial conversion fails
                    dates = pd.to_datetime(date_inputs)
                except (ValueError, TypeError) as e2:
                    logger.debug(f"Second conversion attempt failed: {e2}")
                    # Last resort: try to clean the strings and convert
                    cleaned_inputs = []
                    for d in date_inputs:
                        if isinstance(d, bytes | str):
                            # Clean up string representation of bytes
                            if (
                                isinstance(d, str)
                                and d.startswith("b'")
                                and d.endswith("'")
                            ):
                                d = d[2:-1]
                            # Clean up bytes
                            elif isinstance(d, bytes):
                                d = d.decode("utf-8")
                        cleaned_inputs.append(d)
                    dates = pd.to_datetime(cleaned_inputs, errors="coerce")

            # Create a DataFrame to store features
            df = pd.DataFrame(index=range(len(dates)))

            # Extract requested features
            for feature in self.features:
                if feature == "year":
                    df[feature] = dates.year
                    if self.normalize:
                        # Normalize year to recent range (2000-2030 as default)
                        min_year = 2000
                        max_year = 2030
                        df[feature] = (df[feature] - min_year) / (max_year - min_year)

                elif feature == "month":
                    df[feature] = dates.month
                    if self.normalize:
                        df[feature] = (df[feature] - 1) / 11  # 1-12 -> 0-1

                elif feature == "day":
                    df[feature] = dates.day
                    if self.normalize:
                        df[feature] = (df[feature] - 1) / 30  # 1-31 -> 0-1

                elif feature == "day_of_week":
                    df[feature] = dates.dayofweek  # 0-6
                    if self.normalize:
                        df[feature] = df[feature] / 6  # 0-6 -> 0-1

                elif feature == "day_of_year":
                    df[feature] = dates.dayofyear
                    if self.normalize:
                        df[feature] = (df[feature] - 1) / 365  # 1-366 -> 0-1

                elif feature == "week_of_year":
                    df[feature] = dates.isocalendar().week
                    if self.normalize:
                        df[feature] = (df[feature] - 1) / 52  # 1-53 -> 0-1

                elif feature == "quarter":
                    df[feature] = dates.quarter
                    if self.normalize:
                        df[feature] = (df[feature] - 1) / 3  # 1-4 -> 0-1

                elif feature == "is_weekend":
                    df[feature] = (dates.dayofweek >= 5).astype(float)  # 5=Sat, 6=Sun

                elif feature == "is_month_start":
                    df[feature] = dates.is_month_start.astype(float)

                elif feature == "is_month_end":
                    df[feature] = dates.is_month_end.astype(float)

                elif feature == "is_quarter_start":
                    df[feature] = dates.is_quarter_start.astype(float)

                elif feature == "is_quarter_end":
                    df[feature] = dates.is_quarter_end.astype(float)

                elif feature == "is_year_start":
                    df[feature] = dates.is_year_start.astype(float)

                elif feature == "is_year_end":
                    df[feature] = dates.is_year_end.astype(float)

                elif "_sin" in feature or "_cos" in feature:
                    is_cos = "_cos" in feature
                    base_feature = feature.split("_")[0]

                    if base_feature in self.cyclic_features:
                        # Get cycle length
                        cycle_length = self.cyclic_features[base_feature]

                        # Get base feature values
                        if base_feature == "month":
                            values = dates.month
                        elif base_feature == "day":
                            values = dates.day
                        elif base_feature == "day_of_week":
                            values = dates.dayofweek + 1  # 1-7
                        elif base_feature == "day_of_year":
                            values = dates.dayofyear
                        elif base_feature == "week_of_year":
                            values = dates.isocalendar().week
                        elif base_feature == "quarter":
                            values = dates.quarter
                        elif base_feature == "hour":
                            values = dates.hour
                        elif base_feature == "minute":
                            values = dates.minute
                        elif base_feature == "second":
                            values = dates.second

                        # Apply sin/cos encoding
                        angle = 2 * np.pi * values / cycle_length
                        if is_cos:
                            df[feature] = np.cos(angle)
                        else:
                            df[feature] = np.sin(angle)

            # Convert to numpy array
            return df.values.astype(np.float32)

        # Apply the function
        result = tf.py_function(extract_calendar_features, [inputs], tf.float32)

        # Set the shape
        n_features = len(self.features)
        result.set_shape([inputs.shape[0], n_features])

        return result

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute the output shape of the layer."""
        batch_size = input_shape[0]
        n_features = len(self.features)

        return (batch_size, n_features)

    def get_config(self) -> dict:
        """Return the configuration of the layer."""
        config = {
            "features": self.features,
            "cyclic_encoding": self.cyclic_encoding,
            "input_format": self.input_format,
            "normalize": self.normalize,
            "onehot_categorical": self.onehot_categorical,
        }
        base_config = super().get_config()
        return {**base_config, **config}
