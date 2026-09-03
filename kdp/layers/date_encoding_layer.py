import math
import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="kdp.layers")
class DateEncodingLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        """Initializing DateEncodingLayer."""
        super().__init__(**kwargs)

    @tf.function
    def normalize_year(self, year: tf.Tensor) -> tf.Tensor:
        """Normalize the year to a fractional year value (0-1)."""
        # Example: year could be something like 2023.5 representing mid-2023
        return year % 1.0

    @tf.function
    def cyclic_encoding(
        self,
        value: tf.Tensor,
        period: float,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Encode a value as a cyclical feature using sine and cosine transformations.

        Args:
            value: A tensor of floats representing the value to be encoded.
            period: The period of the cycle (e.g., 12 for months, 7 for days).

        Returns:
            A tuple (sin_encoded, cos_encoded) representing the cyclical features.
        """
        _pi = tf.constant(math.pi)
        normalized_value = value / period
        sin_component = tf.math.sin(2 * _pi * normalized_value)
        cos_component = tf.math.cos(2 * _pi * normalized_value)
        return sin_component, cos_component

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Splits the date into 4 components: year, month, day and day of the week and
        encodes it into sin and cos cyclical projections.

        Args:
            inputs (tf.Tensor): input data [year, month, day_of_month, day_of_week].

        Returns:
            tf.Tensor: cyclically encoded data (sin and cos components).
        """
        # Reshape input if necessary
        input_shape = tf.shape(inputs)
        if len(input_shape) == 3:
            inputs = tf.squeeze(inputs, axis=-1)

        # Extract features
        year = inputs[:, 0]
        month = inputs[:, 1]
        day_of_month = inputs[:, 2]  # New: day of month
        day_of_week = inputs[:, 3]  # Now at index 3

        # Convert to float
        year_float = tf.cast(year, tf.float32)
        month_float = tf.cast(month, tf.float32)
        day_of_month_float = tf.cast(day_of_month, tf.float32)
        day_of_week_float = tf.cast(day_of_week, tf.float32)

        # Ensure inputs are in the correct range
        year_float = self.normalize_year(year_float)

        # Encode each feature in cyclinc projections
        year_sin, year_cos = self.cyclic_encoding(year_float, period=1.0)
        month_sin, month_cos = self.cyclic_encoding(month_float, period=12.0)
        day_of_month_sin, day_of_month_cos = self.cyclic_encoding(
            day_of_month_float,
            period=31.0,
        )
        day_of_week_sin, day_of_week_cos = self.cyclic_encoding(
            day_of_week_float,
            period=7.0,
        )

        encoded = tf.stack(
            [
                year_sin,
                year_cos,
                month_sin,
                month_cos,
                day_of_month_sin,
                day_of_month_cos,
                day_of_week_sin,
                day_of_week_cos,
            ],
            axis=-1,
        )

        # Anything past the four date components belongs to a layer that ran
        # before this one and is carried through. `SeasonLayer` needs the month
        # as a number, which only exists before this encoding, so it runs first
        # and appends its one-hot here -- dropping the extra columns would throw
        # its answer away.
        return tf.concat([encoded, tf.cast(inputs[:, 4:], tf.float32)], axis=-1)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute the output shape after cyclic encoding.

        Args:
            input_shape: Shape of the input tensor, `[batch, 4]` or wider when an
                earlier layer appended columns.

        Returns:
            tf.TensorShape: `[batch, 8]` for the eight cyclic components, plus
            any columns beyond the four date ones, which are carried through.
        """
        width = input_shape[-1]
        carried = 0 if width is None else max(int(width) - 4, 0)
        return tf.TensorShape([input_shape[0], 8 + carried])

    def get_config(self) -> dict:
        """Returns the configuration of the layer as a dictionary."""
        return super().get_config()

    @classmethod
    def from_config(cls, config: dict) -> "DateEncodingLayer":
        """Reloading current configuration."""
        return cls(**config)
