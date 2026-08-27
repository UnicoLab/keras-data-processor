import keras
import tensorflow as tf
from keras.layers import Layer


@keras.saving.register_keras_serializable(package="kdp.layers")
class LagFeatureLayer(Layer):
    """Layer for creating lag features from time series data.

    This layer creates lagged versions of the input feature, useful for
    capturing dependencies on past values in time series data.

    Args:
        lag_indices: List of integers indicating the lag steps to create.
        drop_na: Boolean indicating whether to drop rows with insufficient history.
        fill_value: Value to use for padding when drop_na=False.
        keep_original: Whether to include the original values in the output.
    """

    def __init__(
        self,
        lag_indices,
        drop_na=True,
        fill_value=0.0,
        keep_original=False,
        **kwargs,
    ):
        """Initialize the LagFeatureLayer.

        See the class docstring for the accepted arguments and what
        each one controls.
        """
        super().__init__(**kwargs)
        self.lag_indices = lag_indices
        self.drop_na = drop_na
        self.fill_value = fill_value
        self.keep_original = keep_original

    def build(self, input_shape) -> None:
        """Build the layer's weights for a given input shape.

        Args:
            input_shape: Shape of the input tensor.
        """
        super().build(input_shape)

    def call(self, inputs) -> tf.Tensor:
        """Apply the lag feature transformation.

        Args:
            inputs: Input tensor of shape (batch_size, ...) or (batch_size, time_steps)

        Returns:
            Tensor with original and/or lagged features depending on configuration
        """
        inputs = tf.convert_to_tensor(inputs)
        # Static rank: `tf.rank` returns a tensor, which cannot drive a Python
        # `if` once the layer is traced into a graph.
        input_is_1d = inputs.shape.rank == 1
        if input_is_1d:
            # Reshape to 2D for consistent processing
            inputs = tf.reshape(inputs, (-1, 1))

        # Initialize list to store results
        result_tensors = []

        # Keep the original values if specified
        if self.keep_original:
            result_tensors.append(inputs)

        # Create lag features for each lag index
        for lag in self.lag_indices:
            # Create a shifted version of the input tensor
            padded_inputs = tf.pad(
                inputs,
                [[lag, 0], [0, 0]],
                constant_values=self.fill_value,
            )
            lagged = padded_inputs[:-lag]

            # Add to the result tensors
            result_tensors.append(lagged)

        # Combine all tensors along last axis
        result = tf.concat(result_tensors, axis=-1)

        # Drop rows with insufficient history if required
        if self.drop_na:
            max_lag = max(self.lag_indices)
            result = result[max_lag:]

        # A 1-D series producing a single column is returned as 1-D, matching
        # the rank the caller passed in.
        if input_is_1d and not self.keep_original and len(self.lag_indices) == 1:
            result = tf.reshape(result, (-1,))

        return result

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute the output shape for a given input shape.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            The corresponding output shape.
        """
        input_shape = tuple(input_shape)
        input_is_1d = len(input_shape) == 1
        time_steps = input_shape[0]
        n_features = 1 if input_is_1d else input_shape[-1]

        # Dropping the warm-up rows costs the largest lag; a symbolic batch
        # dimension stays None.
        if self.drop_na and time_steps is not None:
            time_steps = max(time_steps - max(self.lag_indices), 0)

        n_columns = n_features * (
            len(self.lag_indices) + (1 if self.keep_original else 0)
        )

        if input_is_1d and not self.keep_original and len(self.lag_indices) == 1:
            return (time_steps,)
        return (time_steps, n_columns)

    def get_config(self) -> dict:
        """Return the configuration needed to re-create this layer.

        Returns:
            The layer configuration.
        """
        config = {
            "lag_indices": self.lag_indices,
            "drop_na": self.drop_na,
            "fill_value": self.fill_value,
            "keep_original": self.keep_original,
        }
        base_config = super().get_config()
        return {**base_config, **config}
