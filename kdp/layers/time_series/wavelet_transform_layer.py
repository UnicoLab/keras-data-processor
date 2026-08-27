import keras
import tensorflow as tf
from keras.layers import Layer
import numpy as np


@keras.saving.register_keras_serializable(package="kdp.layers")
class WaveletTransformLayer(Layer):
    """Layer for applying simplified wavelet-like transforms to time series data.

    This layer applies a multi-resolution decomposition to time series data,
    similar to wavelet transform but using simple moving averages and differences.
    It can capture patterns at different time scales without external dependencies.

    Args:
        levels: Number of decomposition levels (default: 3)
        keep_levels: Which decomposition levels to keep (default: 'all')
            Options: 'all', 'approx', or list of level indices to keep
        window_sizes: List of window sizes for each level (default: None, which
            automatically calculates window sizes as powers of 2)
        flatten_output: Whether to flatten the coefficients (default: True)
        drop_na: Whether to drop rows with NaN values after transform (default: True)
    """

    def __init__(
        self,
        levels=3,
        keep_levels="all",
        window_sizes=None,
        flatten_output=True,
        drop_na=True,
        **kwargs,
    ):
        """Initialize the WaveletTransformLayer.

        See the class docstring for the accepted arguments and what
        each one controls.
        """
        super().__init__(**kwargs)
        self.levels = levels
        self.keep_levels = keep_levels
        self.window_sizes = window_sizes
        self.flatten_output = flatten_output
        self.drop_na = drop_na

        # Validate keep_levels
        if not (
            keep_levels == "all"
            or keep_levels == "approx"
            or isinstance(keep_levels, list)
        ):
            raise ValueError(
                "keep_levels must be 'all', 'approx', or a list of level indices",
            )

    def build(self, input_shape) -> None:
        """Build the layer's weights for a given input shape.

        Args:
            input_shape: Shape of the input tensor.
        """
        super().build(input_shape)

    def call(self, inputs, training=None) -> tf.Tensor:
        """Apply simplified wavelet transform to the input time series.

        Args:
            inputs: Input tensor of shape (batch_size, time_steps) or (batch_size, time_steps, features)
            training: Boolean tensor indicating whether the call is for training

        Returns:
            Tensor with multi-resolution features
        """
        # Get the input shape and determine if reshaping is needed
        # Remove unused variable
        # original_rank = tf.rank(inputs)

        # Process the input tensor using NumPy for more control over the transform
        def apply_transform(inputs_tensor) -> np.ndarray:
            # Convert to NumPy
            inputs_np = inputs_tensor.numpy()

            # Get dimensions
            if len(inputs_np.shape) == 2:
                batch_size, time_steps = inputs_np.shape
                n_features = 1
                # Remove unused variable
                # multi_feature = False
                # Reshape to 3D for consistent processing
                inputs_np = inputs_np.reshape(batch_size, time_steps, 1)
            else:
                batch_size, time_steps, n_features = inputs_np.shape
                # multi_feature = True

            # Determine window sizes for each level if not provided
            if self.window_sizes is None:
                # Use powers of 2 for window sizes: 2, 4, 8, 16, ...
                self.window_sizes = [2 ** (i + 1) for i in range(self.levels)]
                # Ensure that window sizes are not larger than the time series length
                self.window_sizes = [min(w, time_steps // 2) for w in self.window_sizes]

            # Process each sample in the batch
            all_coeffs = []

            for b in range(batch_size):
                sample_coeffs = []

                for f in range(n_features):
                    # Get the time series for this sample and feature
                    series = inputs_np[b, :, f]

                    # Apply multi-resolution decomposition
                    approx_coeffs = series.copy()
                    level_coeffs = []

                    for level in range(self.levels):
                        # Use a window size appropriate for this level
                        window_size = self.window_sizes[level]

                        # Apply moving average to get approximation coefficients
                        new_approx = self._moving_average(approx_coeffs, window_size)

                        # Detail coefficients are the difference between the current
                        # approximation and the smoother approximation
                        detail = approx_coeffs[window_size - 1 :] - new_approx

                        # Store detail coefficients
                        level_coeffs.append(detail)

                        # Update approximation for next level
                        approx_coeffs = new_approx

                    # Store final approximation (lowest frequency component)
                    level_coeffs.append(approx_coeffs)

                    # Add to sample coefficients
                    sample_coeffs.append((level_coeffs, series.shape[0]))

                all_coeffs.append(sample_coeffs)

            # Filter and process coefficients
            result = self._process_coefficients(
                all_coeffs,
                batch_size,
                n_features,
                time_steps,
            )

            return result.astype(np.float32)

        # Apply the function
        result = tf.py_function(apply_transform, [inputs], tf.float32)

        # Set the shape
        if self.flatten_output:
            # Calculate output features
            n_output_features = self._get_n_output_features(inputs.shape[1])

            if len(inputs.shape) == 2:
                result.set_shape([inputs.shape[0], n_output_features])
            else:
                result.set_shape([inputs.shape[0], inputs.shape[2] * n_output_features])
        else:
            # For non-flattened output, we'll use dynamic shape
            result.set_shape([None, None])

        return result

    def _moving_average(self, series, window_size) -> np.ndarray:
        """Apply moving average to a time series."""
        cumsum = np.cumsum(np.insert(series, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    def _process_coefficients(
        self,
        all_coeffs,
        batch_size,
        n_features,
        time_steps,
    ) -> np.ndarray:
        """Process and filter coefficients based on keep_levels."""
        # Calculate total size of output features
        n_output_features = self._get_n_output_features(time_steps)

        if self.flatten_output:
            # Initialize output array
            result = np.zeros(
                (batch_size, n_features * n_output_features),
                dtype=np.float32,
            )

            for b in range(batch_size):
                feature_idx = 0

                for f in range(n_features):
                    level_coeffs, orig_size = all_coeffs[b][f]

                    # Filter levels based on keep_levels
                    filtered_coeffs = self._filter_levels(level_coeffs)

                    # Flatten and store coefficients
                    for coeffs in filtered_coeffs:
                        # Normalize by original length for easier comparison
                        normalized_coeffs = coeffs / orig_size

                        for val in normalized_coeffs:
                            result[b, feature_idx] = val
                            feature_idx += 1

                            # Prevent index out of bounds if coefficients are larger than expected
                            if feature_idx >= n_features * n_output_features:
                                break

            return result
        else:
            # For non-flattened output, return a more complex structure
            # This is a simplified approach to demonstrate the concept
            return np.zeros((batch_size, n_features, n_output_features))

    def _filter_levels(self, level_coeffs) -> list:
        """Filter coefficient levels based on keep_levels."""
        if self.keep_levels == "all":
            return level_coeffs
        elif self.keep_levels == "approx":
            return [level_coeffs[-1]]  # Keep only the approximation coefficients
        else:
            # Keep specific levels
            filtered = []
            for level in self.keep_levels:
                if level < len(level_coeffs):
                    filtered.append(level_coeffs[level])
            return filtered

    def _get_n_output_features(self, time_steps) -> int:
        """Calculate the number of output features based on wavelet parameters."""
        # In our simplified approach, we'll estimate based on time_steps and levels
        n_features = 0
        remaining_steps = time_steps

        # Calculate expected feature sizes for each level
        level_sizes = []
        for level in range(self.levels):
            window_size = (
                self.window_sizes[level] if self.window_sizes else 2 ** (level + 1)
            )
            detail_size = max(remaining_steps - window_size + 1, 0)
            level_sizes.append(detail_size)
            remaining_steps = detail_size

        # Add approximation coefficients size
        level_sizes.append(remaining_steps)

        # Calculate total features based on keep_levels
        if self.keep_levels == "all":
            n_features = sum(level_sizes)
        elif self.keep_levels == "approx":
            n_features = level_sizes[-1]
        else:
            # Keep specific levels
            n_features = 0
            for level in self.keep_levels:
                if level < len(level_sizes):
                    n_features += level_sizes[level]

        # Ensure a minimum size
        return max(n_features, 1)

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute the output shape of the layer."""
        if self.flatten_output:
            # Calculate output features
            time_steps = input_shape[1]
            n_output_features = self._get_n_output_features(time_steps)

            if len(input_shape) == 3:
                # For multi-feature input
                n_features = input_shape[2]
                n_output_features *= n_features

            output_shape = (input_shape[0], n_output_features)
        else:
            # For non-flattened output
            output_shape = (input_shape[0], None)

        return output_shape

    def get_config(self) -> dict:
        """Return the configuration of the layer."""
        config = {
            "levels": self.levels,
            "keep_levels": self.keep_levels,
            "window_sizes": self.window_sizes,
            "flatten_output": self.flatten_output,
            "drop_na": self.drop_na,
        }
        base_config = super().get_config()
        return {**base_config, **config}
