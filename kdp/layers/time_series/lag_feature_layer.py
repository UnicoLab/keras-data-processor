import tensorflow as tf
from tensorflow.keras.layers import Layer


@tf.keras.utils.register_keras_serializable(package="kdp.layers")
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
        # Get the input shape and determine if reshaping is needed
        original_rank = tf.rank(inputs)
        if original_rank == 1:
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

        # Reshape back to original rank if needed
        if original_rank == 1 and not self.keep_original and len(self.lag_indices) == 1:
            result = tf.reshape(result, (-1,))

        return result

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute the output shape for a given input shape.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            The corresponding output shape.
        """
        output_shape = list(input_shape)
        feature_dim = 0

        if self.keep_original:
            feature_dim += 1

        feature_dim += len(self.lag_indices)

        if len(output_shape) == 1:
            if feature_dim == 1 and not self.keep_original:
                # Just return the same shape if we only have one feature and not keeping original
                return tuple(output_shape)
            else:
                # Add feature dimension
                output_shape.append(feature_dim)
        else:
            # Update the last dimension for feature count
            output_shape[-1] = feature_dim

        # For symbolic shape (where batch dim is None), we can't modify the batch size
        # None batch dimension means variable batch size at runtime
        # So we just return the shape with the updated feature dimension

        return tuple(output_shape)

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
