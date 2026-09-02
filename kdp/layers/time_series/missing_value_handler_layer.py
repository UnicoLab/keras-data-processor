import keras
import tensorflow as tf
from keras.layers import Layer
import numpy as np


@keras.saving.register_keras_serializable(package="kdp.layers")
class MissingValueHandlerLayer(Layer):
    """Layer for advanced handling of missing values in time series data.

    This layer implements various strategies for handling missing values in time series data,
    including forward fill, backward fill, interpolation, and statistical imputation methods.

    Args:
        mask_value: Value used to indicate missing values (default: 0.0)
        strategy: Strategy for handling missing values
            - 'forward_fill': Fill missing values with the last valid value
            - 'backward_fill': Fill missing values with the next valid value
            - 'linear_interpolation': Linear interpolation between valid values
            - 'mean': Fill missing values with the mean of the series
            - 'median': Fill missing values with the median of the series
            - 'rolling_mean': Fill missing values with rolling mean
            - 'seasonal': Fill missing values based on seasonal patterns
        window_size: Window size for rolling strategies (default: 5)
        seasonal_period: Period for seasonal imputation (default: 7)
        add_indicators: Whether to add binary indicators for missing values (default: True)
        extrapolate: Whether to extrapolate for missing values at the beginning/end (default: True)
    """

    def __init__(
        self,
        mask_value=0.0,
        strategy="forward_fill",
        window_size=5,
        seasonal_period=7,
        add_indicators=True,
        extrapolate=True,
        **kwargs,
    ):
        """Initialize the MissingValueHandlerLayer.

        See the class docstring for the accepted arguments and what
        each one controls.
        """
        super().__init__(**kwargs)
        self.mask_value = mask_value
        self.strategy = strategy
        self.window_size = window_size
        self.seasonal_period = seasonal_period
        self.add_indicators = add_indicators
        self.extrapolate = extrapolate

        # Validate parameters
        valid_strategies = [
            "forward_fill",
            "backward_fill",
            "linear_interpolation",
            "mean",
            "median",
            "rolling_mean",
            "seasonal",
        ]
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Strategy must be one of {valid_strategies}, got {strategy}",
            )

    def call(self, inputs, training=None) -> tf.Tensor:
        """Apply missing value handling strategy.

        Args:
            inputs: Input tensor of shape (batch_size, time_steps) or (batch_size, time_steps, features)
            training: Boolean tensor indicating whether the call is for training (not used)

        Returns:
            Tensor with imputed values and optionally missing value indicators
        """
        # For simplicity and reliability, we'll use a numpy-based implementation
        # wrapped with tf.py_function

        inputs_tensor = inputs

        # Use py_function to apply numpy-based imputation
        if len(inputs_tensor.shape) == 2:
            # 2D input (batch_size, time_steps)
            result = tf.py_function(self._numpy_impute_2d, [inputs_tensor], tf.float32)

            # Ensure shape is preserved
            if self.add_indicators:
                result.set_shape([inputs_tensor.shape[0], inputs_tensor.shape[1], 2])
            else:
                result.set_shape(inputs_tensor.shape)
        else:
            # 3D input (batch_size, time_steps, features)
            result = tf.py_function(self._numpy_impute_3d, [inputs_tensor], tf.float32)

            # Ensure shape is preserved
            if self.add_indicators:
                feature_dim = (
                    inputs_tensor.shape[2] * 2
                )  # Original features + indicators
                result.set_shape(
                    [inputs_tensor.shape[0], inputs_tensor.shape[1], feature_dim],
                )
            else:
                result.set_shape(inputs_tensor.shape)

        return result

    def _missing_mask(self, inputs: np.ndarray) -> np.ndarray:
        """Mark the entries this layer should treat as missing.

        Missing values were found with `inputs == self.mask_value`, and NaN is
        equal to nothing -- not even itself -- so a series carrying the marker
        pandas and numpy actually use passed through untouched, and the NaNs
        then poisoned every downstream statistic. Setting `mask_value` to NaN
        selects NaN detection here.

        Args:
            inputs: The batch to inspect.

        Returns:
            A boolean array, True wherever a value counts as missing.
        """
        if isinstance(self.mask_value, float) and np.isnan(self.mask_value):
            return np.isnan(inputs)
        return inputs == self.mask_value

    def _numpy_impute_2d(self, inputs_tensor) -> np.ndarray:
        """Numpy-based implementation of imputation for 2D tensors."""
        # Convert to numpy
        inputs = inputs_tensor.numpy()

        # Create missing mask
        missing_mask = self._missing_mask(inputs)

        # Make a copy to avoid modifying the input
        imputed = inputs.copy()

        # Apply imputation strategy
        if self.strategy == "forward_fill":
            self._numpy_forward_fill(imputed, missing_mask)
        elif self.strategy == "backward_fill":
            self._numpy_backward_fill(imputed, missing_mask)
        elif self.strategy == "linear_interpolation":
            self._numpy_linear_interpolation(imputed, missing_mask)
        elif self.strategy == "mean":
            self._numpy_mean_imputation(imputed, missing_mask)
        elif self.strategy == "median":
            self._numpy_median_imputation(imputed, missing_mask)
        elif self.strategy == "rolling_mean":
            self._numpy_rolling_mean_imputation(imputed, missing_mask)
        elif self.strategy == "seasonal":
            self._numpy_seasonal_imputation(imputed, missing_mask)

        # Add indicators if requested
        if self.add_indicators:
            indicators = missing_mask.astype(np.float32)
            return np.stack([imputed, indicators], axis=-1)
        else:
            return imputed

    def _numpy_impute_3d(self, inputs_tensor) -> np.ndarray:
        """Numpy-based implementation of imputation for 3D tensors."""
        # Convert to numpy
        inputs = inputs_tensor.numpy()

        # Create missing mask
        missing_mask = self._missing_mask(inputs)

        # Get dimensions
        batch_size, time_steps, n_features = inputs.shape

        # Make a copy to avoid modifying the input
        imputed = inputs.copy()

        # Apply imputation to each feature separately
        for f in range(n_features):
            feature_data = inputs[:, :, f]
            feature_mask = missing_mask[:, :, f]

            # Make a copy for each feature
            feature_imputed = feature_data.copy()

            # Apply imputation strategy
            if self.strategy == "forward_fill":
                self._numpy_forward_fill(feature_imputed, feature_mask)
            elif self.strategy == "backward_fill":
                self._numpy_backward_fill(feature_imputed, feature_mask)
            elif self.strategy == "linear_interpolation":
                self._numpy_linear_interpolation(feature_imputed, feature_mask)
            elif self.strategy == "mean":
                self._numpy_mean_imputation(feature_imputed, feature_mask)
            elif self.strategy == "median":
                self._numpy_median_imputation(feature_imputed, feature_mask)
            elif self.strategy == "rolling_mean":
                self._numpy_rolling_mean_imputation(feature_imputed, feature_mask)
            elif self.strategy == "seasonal":
                self._numpy_seasonal_imputation(feature_imputed, feature_mask)

            # Update the imputed array
            imputed[:, :, f] = feature_imputed

        # Add indicators if requested
        if self.add_indicators:
            indicators = missing_mask.astype(np.float32)
            return np.concatenate([imputed, indicators], axis=-1)
        else:
            return imputed

    def _numpy_forward_fill(self, data, mask) -> None:
        """Forward fill missing values in-place."""
        # For each batch
        for b in range(data.shape[0]):
            # Get the series and its mask
            series = data[b]
            series_mask = mask[b]

            # Skip if no missing values
            if not np.any(series_mask):
                continue

            # Initialize last valid value
            last_valid = None

            # Process each time step
            for t in range(len(series)):
                if series_mask[t]:
                    # Missing value
                    if last_valid is not None:
                        # Fill with last valid value
                        series[t] = last_valid
                else:
                    # Valid value, update last_valid
                    last_valid = series[t]

    def _numpy_backward_fill(self, data, mask) -> None:
        """Backward fill missing values in-place."""
        # For each batch
        for b in range(data.shape[0]):
            # Get the series and its mask
            series = data[b]
            series_mask = mask[b]

            # Skip if no missing values
            if not np.any(series_mask):
                continue

            # Initialize next valid value
            next_valid = None

            # Process each time step in reverse
            for t in range(len(series) - 1, -1, -1):
                if series_mask[t]:
                    # Missing value
                    if next_valid is not None:
                        # Fill with next valid value
                        series[t] = next_valid
                else:
                    # Valid value, update next_valid
                    next_valid = series[t]

    def _numpy_linear_interpolation(self, data, mask) -> None:
        """Linear interpolation between valid values in-place."""
        # For each batch
        for b in range(data.shape[0]):
            # Get the series and its mask
            series = data[b]
            series_mask = mask[b]

            # Skip if no missing values
            if not np.any(series_mask):
                continue

            # First, apply forward and backward fill
            series_copy = series.copy()

            # Forward fill
            last_valid = None
            for t in range(len(series)):
                if series_mask[t]:
                    if last_valid is not None:
                        series_copy[t] = last_valid
                else:
                    last_valid = series[t]

            # Backward fill
            next_valid = None
            for t in range(len(series) - 1, -1, -1):
                if series_mask[t]:
                    if next_valid is not None:
                        # For missing values, compute weighted average
                        if last_valid is not None:
                            # Find preceding and following valid indices
                            left_idx = (
                                np.nonzero(~series_mask[:t])[0][-1]
                                if np.any(~series_mask[:t])
                                else None
                            )
                            right_idx = (
                                np.nonzero(~series_mask[t + 1 :])[0][0] + t + 1
                                if np.any(~series_mask[t + 1 :])
                                else None
                            )

                            if left_idx is not None and right_idx is not None:
                                # Linear interpolation
                                left_val = series[left_idx]
                                right_val = series[right_idx]
                                dist = right_idx - left_idx
                                pos = t - left_idx
                                series[t] = (
                                    left_val + (right_val - left_val) * pos / dist
                                )
                            elif left_idx is not None:
                                # Only left value available, use forward fill
                                series[t] = series[left_idx]
                            elif right_idx is not None:
                                # Only right value available, use backward fill
                                series[t] = series[right_idx]
                        else:
                            # No left value, use backward fill
                            series[t] = next_valid
                else:
                    next_valid = series[t]

    def _numpy_mean_imputation(self, data, mask) -> None:
        """Mean imputation in-place."""
        # For each batch
        for b in range(data.shape[0]):
            # Get the series and its mask
            series = data[b]
            series_mask = mask[b]

            # Skip if no missing values
            if not np.any(series_mask):
                continue

            # Calculate mean of valid values
            valid_values = series[~series_mask]
            if len(valid_values) > 0:
                mean_value = np.mean(valid_values)

                # Fill missing values with mean
                series[series_mask] = mean_value

    def _numpy_median_imputation(self, data, mask) -> None:
        """Median imputation in-place."""
        # For each batch
        for b in range(data.shape[0]):
            # Get the series and its mask
            series = data[b]
            series_mask = mask[b]

            # Skip if no missing values
            if not np.any(series_mask):
                continue

            # Calculate median of valid values
            valid_values = series[~series_mask]
            if len(valid_values) > 0:
                median_value = np.median(valid_values)

                # Fill missing values with median
                series[series_mask] = median_value

    def _numpy_rolling_mean_imputation(self, data, mask) -> None:
        """Rolling mean imputation in-place."""
        # For each batch
        for b in range(data.shape[0]):
            # Get the series and its mask
            series = data[b]
            series_mask = mask[b]

            # Skip if no missing values
            if not np.any(series_mask):
                continue

            # Get window size
            window = self.window_size
            half_window = window // 2

            # Process each missing value
            for t in np.where(series_mask)[0]:
                # Define window boundaries
                start = max(0, t - half_window)
                end = min(len(series), t + half_window + 1)

                # Get valid values in window
                window_values = []
                for i in range(start, end):
                    if i != t and not series_mask[i]:
                        window_values.append(series[i])

                # Calculate window mean
                if len(window_values) > 0:
                    window_mean = np.mean(window_values)
                    series[t] = window_mean
                else:
                    # No valid values in window, use global mean
                    valid_values = series[~series_mask]
                    if len(valid_values) > 0:
                        series[t] = np.mean(valid_values)

    def _numpy_seasonal_imputation(self, data, mask) -> None:
        """Seasonal imputation in-place."""
        # For each batch
        for b in range(data.shape[0]):
            # Get the series and its mask
            series = data[b]
            series_mask = mask[b]

            # Skip if no missing values
            if not np.any(series_mask):
                continue

            # Get seasonal period
            period = self.seasonal_period

            # Process each missing value
            for t in np.where(series_mask)[0]:
                # Find values at the same phase in the cycle
                phase = t % period
                phase_indices = np.arange(phase, len(series), period)

                # Get valid values at this phase
                phase_values = []
                for idx in phase_indices:
                    if idx != t and not series_mask[idx]:
                        phase_values.append(series[idx])

                # Calculate phase mean
                if len(phase_values) > 0:
                    phase_mean = np.mean(phase_values)
                    series[t] = phase_mean
                else:
                    # No valid values at this phase, fall back to rolling mean
                    self._numpy_rolling_mean_imputation(
                        data[b : b + 1],
                        mask[b : b + 1],
                    )

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute output shape of the layer."""
        if len(input_shape) == 2:
            # (batch_size, time_steps)
            if self.add_indicators:
                return (input_shape[0], input_shape[1], 2)
            else:
                return input_shape
        else:
            # (batch_size, time_steps, features)
            if self.add_indicators:
                # Add indicators for each feature
                return (input_shape[0], input_shape[1], input_shape[2] * 2)
            else:
                return input_shape

    def get_config(self) -> dict:
        """Return layer configuration."""
        config = {
            "mask_value": self.mask_value,
            "strategy": self.strategy,
            "window_size": self.window_size,
            "seasonal_period": self.seasonal_period,
            "add_indicators": self.add_indicators,
            "extrapolate": self.extrapolate,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
