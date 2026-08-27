import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="kdp.layers")
class DateParsingLayer(tf.keras.layers.Layer):
    def __init__(self, date_format: str = "YYYY-MM-DD", **kwargs) -> None:
        """Initializing DateParsingLayer.

        Args:
            date_format (str): format of the string encoded date to parse.
                Supported formats: YYYY-MM-DD, YYYY/MM/DD
            kwargs (dict): other params to pass to the class.
        """
        super().__init__(**kwargs)
        self.date_format = date_format

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Base forward pass definition.

        Args:
            inputs (tf.Tensor): Tensor with input data.

        Returns:
            tf.Tensor: processed date tensor with all components
            [year, month, day_of_month, day_of_week].
        """

        def parse_date(date_str: str) -> tf.Tensor:
            # Handle missing/invalid dates
            is_valid = tf.strings.regex_full_match(
                date_str,
                r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,2}$",
            )
            tf.debugging.assert_equal(
                is_valid,
                True,
                message="Invalid date format. Expected YYYY-MM-DD or YYYY/MM/DD",
            )

            # First, standardize the separator to '-' in case of YYYY/MM/DD format
            date_str = tf.strings.regex_replace(date_str, "/", "-")

            parts = tf.strings.split(date_str, "-")
            year = tf.strings.to_number(parts[0], out_type=tf.int32)
            month = tf.strings.to_number(parts[1], out_type=tf.int32)
            day_of_month = tf.strings.to_number(parts[2], out_type=tf.int32)

            # Validate date components
            # Validate year is in reasonable range
            tf.debugging.assert_greater_equal(
                year,
                1000,
                message="Year must be >= 1000",
            )
            tf.debugging.assert_less_equal(
                year,
                2200,
                message="Year must be <= 2200",
            )

            # Validate month is between 1-12
            tf.debugging.assert_greater_equal(
                month,
                1,
                message="Month must be >= 1",
            )
            tf.debugging.assert_less_equal(
                month,
                12,
                message="Month must be <= 12",
            )

            # Validate day is between 1-31
            tf.debugging.assert_greater_equal(
                day_of_month,
                1,
                message="Day must be >= 1",
            )
            tf.debugging.assert_less_equal(
                day_of_month,
                31,
                message="Day must be <= 31",
            )

            # Calculate day of week using Zeller's congruence
            y = tf.where(month < 3, year - 1, year)
            m = tf.where(month < 3, month + 12, month)
            k = y % 100
            j = y // 100
            h = (
                day_of_month + ((13 * (m + 1)) // 5) + k + (k // 4) + (j // 4) - (2 * j)
            ) % 7
            day_of_week = tf.where(
                h == 0, 6, h - 1
            )  # Adjust to 0-6 range where 0 is Sunday

            return tf.stack([year, month, day_of_month, day_of_week])

        parsed_dates = tf.map_fn(
            parse_date, tf.squeeze(inputs), fn_output_signature=tf.int32
        )
        return parsed_dates

    def compute_output_shape(self, input_shape: int) -> int:
        """Getting output shape."""
        return tf.TensorShape([input_shape[0], 4])  # Changed to 4 components

    def get_config(self) -> dict:
        """Saving configuration."""
        config = super().get_config()
        config.update({"date_format": self.date_format})
        return config

    @classmethod
    def from_config(cls, config: dict) -> "DateParsingLayer":
        """Restoring configuration."""
        return cls(**config)
