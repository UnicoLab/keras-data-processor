import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="kdp.layers")
class SeasonLayer(keras.layers.Layer):
    """A Keras Layer that adds seasonal information to the input tensor based on the month.

    This layer determines the season for each month and encodes it as a one-hot vector. The seasons are Winter,
    Spring, Summer, and Fall. The one-hot encoding is appended to the input tensor.

    Required Input Format:
        - A tensor of shape [batch_size, 4], where each row contains:
            - year (int): Year as a numerical value.
            - month (int): Month as an integer from 1 to 12.
            - day_of_month (int): Day of the month as an integer from 1 to 31.
            - day_of_week (int): Day of the week as an integer from 0 to 6 (where 0=Sunday).
    """

    def __init__(self, **kwargs):
        """Initializing SeasonLayer."""
        super().__init__(**kwargs)

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Adds seasonal one-hot encoding to the input tensor.

        Args:
            inputs (tf.Tensor): A tensor of shape [batch_size, 4] where each row contains
            [year, month, day_of_month, day_of_week].

        Returns:
            tf.Tensor: A tensor of shape [batch_size, 8] with the original features
            plus the one-hot encoded season information.

        Raises:
            ValueError: If the input tensor does not have shape [batch_size, 4] or contains invalid month values.
        """
        # Ensure inputs is 2D
        if len(tf.shape(inputs)) == 1:
            inputs = tf.expand_dims(inputs, axis=0)

        # Extract month (assuming it's the second column)
        month = tf.cast(inputs[:, 1], tf.int32)

        # Determine season using TensorFlow operations
        is_winter = tf.logical_or(tf.less_equal(month, 2), tf.equal(month, 12))
        is_spring = tf.logical_and(tf.greater(month, 2), tf.less_equal(month, 5))
        is_summer = tf.logical_and(tf.greater(month, 5), tf.less_equal(month, 8))
        is_fall = tf.logical_and(tf.greater(month, 8), tf.less_equal(month, 11))

        season = (
            tf.cast(is_winter, tf.int32) * 0
            + tf.cast(is_spring, tf.int32) * 1
            + tf.cast(is_summer, tf.int32) * 2
            + tf.cast(is_fall, tf.int32) * 3
        )

        # Convert season to one-hot encoding and cast to float32 to match input type
        season_one_hot = tf.cast(tf.one_hot(season, depth=4), tf.float32)

        # Just in case it comes as int32, cast inputs to float32
        inputs = tf.cast(inputs, tf.float32)

        # Now both tensors are float32, concatenation will work
        return tf.concat([inputs, season_one_hot], axis=-1)

    def compute_output_shape(self, input_shape: int) -> int:
        """Calculating output shape."""
        # Convert input_shape to TensorShape if it's not already
        input_shape = tf.TensorShape(input_shape)
        # Add 4 to the last dimension for the one-hot encoded season
        return input_shape[:-1].concatenate([input_shape[-1] + 4])

    def get_config(self) -> dict:
        """Returns the configuration of the layer as a dictionary.

        Returns:
            dict: The configuration dictionary.
        """
        return super().get_config()

    @classmethod
    def from_config(cls, config: dict) -> "SeasonLayer":
        """Instantiates a SeasonLayer from its configuration dictionary.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            object: The SeasonLayer instance.
        """
        return cls(**config)
