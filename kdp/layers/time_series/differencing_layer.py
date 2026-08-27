import keras
import tensorflow as tf
from keras.layers import Layer


@keras.saving.register_keras_serializable(package="kdp.layers")
class DifferencingLayer(Layer):
    """Layer for computing differences of time series data.

    This layer computes differences of various orders (first-order, second-order, etc.).
    It's useful for making time series stationary.

    Args:
        order: The order of differencing to apply (default=1)
        drop_na: Whether to drop rows with NA values after differencing (default=True)
        fill_value: Value to use for padding when drop_na=False (default=0.0)
        keep_original: Whether to include the original values in the output (default=False)
    """

    def __init__(
        self,
        order=1,
        drop_na=True,
        fill_value=0.0,
        keep_original=False,
        **kwargs,
    ):
        """Initialize the DifferencingLayer.

        See the class docstring for the accepted arguments and what
        each one controls.
        """
        super().__init__(**kwargs)
        self.order = order
        self.drop_na = drop_na
        self.fill_value = fill_value
        self.keep_original = keep_original

        # Validate order
        if self.order <= 0:
            raise ValueError(f"Order must be positive. Got {order}")

    def build(self, input_shape) -> None:
        """Build the layer's weights for a given input shape.

        Args:
            input_shape: Shape of the input tensor.
        """
        super().build(input_shape)

    def call(self, inputs) -> tf.Tensor:
        """Apply the differencing operation.

        Args:
            inputs: Input tensor of shape (time_steps,) or (time_steps, features).

        Returns:
            Tensor with original and/or differenced values depending on configuration.
        """
        inputs = tf.convert_to_tensor(inputs)
        # Use the static rank: `tf.rank` returns a tensor, which cannot drive a
        # Python `if` once the layer is traced into a graph.
        input_is_1d = inputs.shape.rank == 1
        if input_is_1d:
            inputs = tf.reshape(inputs, (-1, 1))

        # Compute differences of the specified order.
        diff = inputs
        for _ in range(self.order):
            diff_values = diff[1:] - diff[:-1]
            if self.drop_na:
                diff = diff_values
            else:
                padding = tf.fill(
                    [1, tf.shape(diff_values)[1]],
                    tf.cast(self.fill_value, diff_values.dtype),
                )
                diff = tf.concat([padding, diff_values], axis=0)

        if self.keep_original:
            # When the leading rows are dropped, the originals must be trimmed
            # by the same amount so the two halves line up row for row.
            original = inputs[self.order :] if self.drop_na else inputs
            length = tf.minimum(tf.shape(original)[0], tf.shape(diff)[0])
            result = tf.concat([original[:length], diff[:length]], axis=-1)
        else:
            result = diff

        # A 1-D series that maps to a single column is returned as 1-D, matching
        # the rank the caller passed in.
        if input_is_1d and not self.keep_original and self.order == 1 and self.drop_na:
            result = tf.reshape(result, [-1])

        return result

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute the output shape for a given input shape.

        Args:
            input_shape: Shape of the input, ``(time_steps,)`` or
                ``(time_steps, features)``.

        Returns:
            The corresponding output shape.
        """
        input_shape = tuple(input_shape)
        input_is_1d = len(input_shape) == 1
        time_steps = input_shape[0]
        n_features = 1 if input_is_1d else input_shape[-1]

        # Each differencing pass consumes `order` leading rows unless they are
        # padded back with fill_value.
        if self.drop_na and time_steps is not None:
            time_steps = max(time_steps - self.order, 0)

        n_output_features = n_features * (2 if self.keep_original else 1)

        if input_is_1d and not self.keep_original and self.order == 1 and self.drop_na:
            return (time_steps,)
        return (time_steps, n_output_features)

    def get_config(self) -> dict:
        """Return the configuration needed to re-create this layer.

        Returns:
            The layer configuration.
        """
        config = {
            "order": self.order,
            "drop_na": self.drop_na,
            "fill_value": self.fill_value,
            "keep_original": self.keep_original,
        }
        base_config = super().get_config()
        return {**base_config, **config}
