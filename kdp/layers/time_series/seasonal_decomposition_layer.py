import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np


@tf.keras.utils.register_keras_serializable(package="kdp.layers")
class SeasonalDecompositionLayer(Layer):
    """Layer for decomposing time series data into trend, seasonal, and residual components.

    This layer implements a simplified version of classical time series decomposition,
    breaking a time series into:
    - Trend component (long-term progression)
    - Seasonal component (repeating patterns at fixed intervals)
    - Residual component (remaining variation)

    Args:
        period: Length of the seasonal cycle. Must be provided.
        method: Decomposition method, either 'additive' or 'multiplicative'.
        trend_window: Size of the window for moving average trend extraction.
            If None, defaults to period.
        extrapolate_trend: Strategy for handling trend calculation at boundaries:
            'nearest' - use nearest valid values
            'linear' - use linear extrapolation
        keep_original: Whether to include the original values in the output.
        drop_na: Whether to drop rows with insufficient history.
    """

    def __init__(
        self,
        period,
        method="additive",
        trend_window=None,
        extrapolate_trend="nearest",
        keep_original=False,
        drop_na=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.period = period
        self.method = method
        self.trend_window = trend_window if trend_window is not None else period
        self.extrapolate_trend = extrapolate_trend
        self.keep_original = keep_original
        self.drop_na = drop_na

        # Validate parameters
        if self.method not in ["additive", "multiplicative"]:
            raise ValueError(
                f"Method must be 'additive' or 'multiplicative', got {method}",
            )
        if self.extrapolate_trend not in ["nearest", "linear"]:
            raise ValueError(
                f"Extrapolate_trend must be 'nearest' or 'linear', got {extrapolate_trend}",
            )

    def call(self, inputs) -> tf.Tensor:
        """Apply seasonal decomposition to the input time series.

        Args:
            inputs: Input tensor of shape (batch_size, time_steps) or (batch_size, time_steps, features)

        Returns:
            Tensor with decomposed components
        """
        # Handle different input shapes
        input_rank = len(inputs.shape)
        if input_rank == 2:
            # (batch_size, time_steps)
            return self._decompose_2d(inputs)
        else:
            # (batch_size, time_steps, features)
            # Process each feature separately
            outputs = []
            for i in range(inputs.shape[2]):
                feature = inputs[:, :, i]
                decomposed = self._decompose_2d(feature)
                outputs.append(decomposed)

            # Concatenate results along the feature dimension
            return tf.concat(outputs, axis=2)

    def _decompose_2d(self, inputs):
        """Decompose a single 2D time series."""
        # Extract dimensions - remove unused variables
        # batch_size = tf.shape(inputs)[0]
        # time_steps = tf.shape(inputs)[1]

        # Calculate trend component using moving average
        trend = self._calculate_trend(inputs)

        # Calculate seasonal component
        if self.method == "additive":
            detrended = inputs - trend
        else:  # multiplicative
            # Avoid division by zero
            eps = 1e-10
            safe_trend = tf.maximum(trend, eps)
            detrended = inputs / safe_trend
            # Replace NaNs and Infs
            detrended = tf.where(
                tf.math.is_finite(detrended),
                detrended,
                tf.zeros_like(detrended),
            )

        # Calculate seasonal component
        seasonal = self._calculate_seasonal(detrended)

        # Calculate residual component
        if self.method == "additive":
            residual = inputs - trend - seasonal
        else:  # multiplicative
            # Avoid division by zero
            eps = 1e-10
            safe_trend = tf.maximum(trend, eps)
            safe_seasonal = tf.maximum(seasonal, eps)

            residual = inputs / (safe_trend * safe_seasonal)
            # Replace NaNs and Infs
            residual = tf.where(
                tf.math.is_finite(residual),
                residual,
                tf.zeros_like(residual),
            )

        # Stack components
        components = [trend, seasonal, residual]
        if self.keep_original:
            components.insert(0, inputs)

        # Stack along the last dimension
        result = tf.stack(components, axis=2)

        # Drop rows if needed
        if self.drop_na:
            drop_size = (self.trend_window - 1) // 2
            if drop_size > 0:
                result = result[drop_size:, :, :]

        return result

    def _calculate_trend(self, inputs):
        """Calculate trend component using centered moving average."""

        # Use numpy-style operations with tf.py_function for simplicity
        def moving_average(batch_tensor):
            # Convert to numpy for easier manipulation
            batch_np = batch_tensor.numpy()
            result = np.zeros_like(batch_np)

            # Apply moving average for each batch item
            window_size = self.trend_window
            half_window = window_size // 2

            for b in range(batch_np.shape[0]):
                x = batch_np[b]
                # Initialize trend with zeros
                trend = np.zeros_like(x)

                # Calculate moving average
                for i in range(len(x)):
                    # Define window boundaries
                    start_idx = max(0, i - half_window)
                    end_idx = min(len(x), i + half_window + 1)
                    # Calculate average of values in window
                    if end_idx > start_idx:
                        trend[i] = np.mean(x[start_idx:end_idx])
                    else:
                        trend[i] = x[i]  # Fallback if window is empty

                result[b] = trend

            return result.astype(np.float32)

        # Apply moving average using tf.py_function
        trend = tf.py_function(moving_average, [inputs], tf.float32)

        # Ensure shape is preserved
        trend.set_shape(inputs.shape)
        return trend

    def _calculate_seasonal(self, detrended):
        """Calculate seasonal component by averaging values at the same phase."""

        # Use numpy-style operations with tf.py_function for simplicity
        def extract_seasonal(batch_tensor):
            # Convert to numpy for easier manipulation
            batch_np = batch_tensor.numpy()
            result = np.zeros_like(batch_np)

            # Apply seasonal extraction for each batch item
            period = self.period

            for b in range(batch_np.shape[0]):
                x = batch_np[b]
                # Initialize seasonal component
                seasonal = np.zeros_like(x)

                # Calculate average for each phase in the period
                for phase in range(period):
                    # Get indices for this phase
                    indices = np.arange(phase, len(x), period)
                    if len(indices) > 0:
                        # Calculate mean for this phase
                        phase_values = x[indices]
                        phase_mean = np.nanmean(phase_values)  # Handle NaN values

                        # Assign the mean to all positions with this phase
                        for idx in indices:
                            seasonal[idx] = phase_mean

                # For multiplicative model, normalize the seasonal component
                if self.method == "multiplicative":
                    # Calculate mean of seasonal component
                    seasonal_mean = np.nanmean(seasonal)
                    # Avoid division by zero
                    seasonal = (
                        seasonal / seasonal_mean
                        if abs(seasonal_mean) > 1e-10
                        else np.ones_like(seasonal)
                    )

                result[b] = seasonal

            return result.astype(np.float32)

        # Apply seasonal extraction using tf.py_function
        seasonal = tf.py_function(extract_seasonal, [detrended], tf.float32)

        # Ensure shape is preserved
        seasonal.set_shape(detrended.shape)
        return seasonal

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute output shape of the layer."""
        if len(input_shape) == 2:
            # (batch_size, time_steps) -> (batch_size, time_steps, n_components)
            batch_size, time_steps = input_shape
            n_components = 4 if self.keep_original else 3

            # Adjust batch size if dropping rows
            if self.drop_na and batch_size is not None:
                drop_rows = (self.trend_window - 1) // 2
                batch_size = max(1, batch_size - drop_rows)

            return (batch_size, time_steps, n_components)
        else:
            # (batch_size, time_steps, features) -> (batch_size, time_steps, features * n_components)
            batch_size, time_steps, features = input_shape
            n_components = 4 if self.keep_original else 3

            # Adjust batch size if dropping rows
            if self.drop_na and batch_size is not None:
                drop_rows = (self.trend_window - 1) // 2
                batch_size = max(1, batch_size - drop_rows)

            return (batch_size, time_steps, features * n_components)

    def get_config(self) -> dict:
        """Return layer configuration."""
        config = {
            "period": self.period,
            "method": self.method,
            "trend_window": self.trend_window,
            "extrapolate_trend": self.extrapolate_trend,
            "keep_original": self.keep_original,
            "drop_na": self.drop_na,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
