import keras
import tensorflow as tf
from keras.layers import Layer
import numpy as np


@keras.saving.register_keras_serializable(package="kdp.layers")
class AutoLagSelectionLayer(Layer):
    """Layer for automatically selecting optimal lag features based on autocorrelation analysis.

    This layer analyzes the autocorrelation of time series data to identify important
    lag values, then creates lag features for those values. This is more efficient
    than creating lag features for all possible lags.

    Args:
        max_lag: Maximum lag to consider
        n_lags: Number of lag features to create (default: 5)
        threshold: Autocorrelation significance threshold (default: 0.2)
        method: Method for selecting lags
            - 'top_k': Select the top k lags with highest autocorrelation
            - 'threshold': Select all lags with autocorrelation above threshold
        drop_na: Whether to drop rows with insufficient history
        fill_value: Value to use for padding when drop_na=False
        keep_original: Whether to include the original values in the output
    """

    def __init__(
        self,
        max_lag=30,
        n_lags=5,
        threshold=0.2,
        method="top_k",
        drop_na=True,
        fill_value=0.0,
        keep_original=True,
        **kwargs,
    ):
        """Initialize the AutoLagSelectionLayer.

        See the class docstring for the accepted arguments and what
        each one controls.
        """
        super().__init__(**kwargs)
        self.max_lag = max_lag
        self.n_lags = n_lags
        self.threshold = threshold
        self.method = method
        self.drop_na = drop_na
        self.fill_value = fill_value
        self.keep_original = keep_original

        # Validate parameters
        if self.method not in ["top_k", "threshold"]:
            raise ValueError(f"Method must be 'top_k' or 'threshold', got {method}")

        # Initialize selected lags
        self.selected_lags = None

    def build(self, input_shape) -> None:
        """Build the layer's weights for a given input shape.

        Args:
            input_shape: Shape of the input tensor.
        """
        super().build(input_shape)

    def call(self, inputs, training=None) -> tf.Tensor:
        """Apply automatic lag selection.

        Args:
            inputs: Input tensor of shape (batch_size, time_steps) or (batch_size, time_steps, features)
            training: Boolean tensor indicating whether the call is for training (not used)

        Returns:
            Tensor with selected lag features
        """
        # Get the input shape and determine if reshaping is needed
        original_rank = tf.rank(inputs)

        # Handle different input shapes
        if original_rank == 2:
            # Shape: (batch_size, time_steps)
            series = inputs
            multi_feature = False
        else:
            # Shape: (batch_size, time_steps, features)
            # For now, just use the first feature for autocorrelation analysis
            # This could be extended to analyze each feature separately
            series = inputs[:, :, 0]
            multi_feature = True

        # During training, compute the autocorrelation and select lags
        # During inference, use the precomputed lags
        if training is None or training:
            # Compute autocorrelation for lag selection
            acf = self._compute_autocorrelation(series)

            # Select lags based on autocorrelation
            self.selected_lags = self._select_lags(acf)

        # Use K.in_train_phase to conditionally execute code based on training mode
        # This ensures compatibility with TF saved model/graph execution
        if self.selected_lags is None:
            # Default to sequential lags if none are selected yet
            default_lags = tf.range(1, self.n_lags + 1)
            self.selected_lags = default_lags

        # Create lag features
        # Handle lag feature creation as a NumPy operation for more control
        def create_lag_features(inputs_tensor, selected_lags_tensor) -> np.ndarray:
            # Convert to NumPy
            inputs_np = inputs_tensor.numpy()
            selected_lags_np = selected_lags_tensor.numpy()

            # Get dimensions
            if len(inputs_np.shape) == 2:
                batch_size, time_steps = inputs_np.shape
                n_features = 1
                single_feature = True
            else:
                batch_size, time_steps, n_features = inputs_np.shape
                single_feature = False

            # Number of output features
            n_output_features = n_features * (
                1 if self.keep_original else 0
            ) + n_features * len(selected_lags_np)

            # Create output array
            if self.drop_na:
                max_lag = np.max(selected_lags_np)
                # Ensure we don't create a negative dimension
                output_batch_size = max(1, batch_size - max_lag)
                result = np.zeros(
                    (output_batch_size, time_steps, n_output_features),
                    dtype=inputs_np.dtype,
                )
            else:
                result = np.zeros(
                    (batch_size, time_steps, n_output_features),
                    dtype=inputs_np.dtype,
                )

            # Feature index counter
            feature_idx = 0

            # Add original features if requested
            if self.keep_original:
                if single_feature:
                    # Add feature dimension for 2D input
                    if self.drop_na:
                        max_lag = np.max(selected_lags_np)
                        if batch_size > max_lag:
                            result[:, :, feature_idx] = inputs_np[max_lag:]
                    else:
                        result[:, :, feature_idx] = inputs_np
                    feature_idx += 1
                else:
                    # Add all original features for 3D input
                    if self.drop_na:
                        max_lag = np.max(selected_lags_np)
                        if batch_size > max_lag:
                            result[:, :, :n_features] = inputs_np[max_lag:]
                    else:
                        result[:, :, :n_features] = inputs_np
                    feature_idx += n_features

            # Add lag features
            for lag in selected_lags_np:
                if single_feature:
                    # For 2D input (single feature)
                    if self.drop_na:
                        max_lag = np.max(selected_lags_np)
                        if batch_size > max_lag:
                            # Shift the input series and place in output
                            for i in range(min(batch_size - max_lag, result.shape[0])):
                                # Use data from (i + max_lag - lag) to create lag feature at position i
                                orig_idx = i + max_lag
                                if orig_idx - lag >= 0:
                                    result[i, lag:, feature_idx] = inputs_np[
                                        orig_idx - lag,
                                        :-lag,
                                    ]
                                else:
                                    # Handle case where lag goes beyond input bounds
                                    result[i, lag:, feature_idx] = self.fill_value
                    else:
                        # Without drop_na, we pad the beginning with fill_value
                        for i in range(batch_size):
                            if lag > 0:
                                # First `lag` positions are padded
                                result[i, :lag, feature_idx] = self.fill_value
                                # For test_call_2d, we need to ensure the shifted values match exactly
                                # what's expected in the test
                                result[i, lag:, feature_idx] = inputs_np[i, :-lag]
                    feature_idx += 1
                else:
                    # For 3D input (multiple features)
                    if self.drop_na:
                        max_lag = np.max(selected_lags_np)
                        if batch_size > max_lag:
                            # Shift each feature and place in output
                            for f in range(n_features):
                                for i in range(
                                    min(batch_size - max_lag, result.shape[0]),
                                ):
                                    # Use data from (i + max_lag - lag) to create lag feature at position i
                                    orig_idx = i + max_lag
                                    if orig_idx - lag >= 0:
                                        result[i, lag:, feature_idx + f] = inputs_np[
                                            orig_idx - lag,
                                            :-lag,
                                            f,
                                        ]
                                    else:
                                        # Handle case where lag goes beyond input bounds
                                        result[
                                            i,
                                            lag:,
                                            feature_idx + f,
                                        ] = self.fill_value
                    else:
                        # Without drop_na, we pad the beginning with fill_value
                        for f in range(n_features):
                            for i in range(batch_size):
                                if lag > 0:
                                    # First `lag` positions are padded
                                    result[i, :lag, feature_idx + f] = self.fill_value
                                    # Rest are shifted values
                                    result[i, lag:, feature_idx + f] = inputs_np[
                                        i,
                                        :-lag,
                                        f,
                                    ]
                    feature_idx += n_features

            return result.astype(np.float32)

        # Apply the function
        if self.selected_lags is not None:
            result = tf.py_function(
                create_lag_features,
                [inputs, self.selected_lags],
                tf.float32,
            )

            # Set the shape
            if multi_feature:
                n_features = inputs.shape[2]
                n_output_features = (
                    n_features * (1 if self.keep_original else 0)
                    + n_features * self.n_lags
                )
            else:
                n_output_features = (1 if self.keep_original else 0) + self.n_lags

            if self.drop_na:
                # `create_lag_features` allocates `max(1, rows - max_lag)` rows,
                # so the declared shape has to use the same clamp. Declaring the
                # unclamped difference made `set_shape` reject a negative
                # dimension whenever the input had fewer rows than the largest
                # selected lag -- which is every single-row batch, and so the
                # default configuration always raised.
                max_lag = int(tf.reduce_max(self.selected_lags))
                rows = inputs.shape[0]
                declared_rows = None if rows is None else max(1, rows - max_lag)
                result.set_shape([declared_rows, inputs.shape[1], n_output_features])
            else:
                result.set_shape([inputs.shape[0], inputs.shape[1], n_output_features])

            return result
        else:
            # Fallback case (shouldn't happen in normal execution)
            return inputs

    def _compute_autocorrelation(self, series) -> np.ndarray:
        """Compute autocorrelation for lags 1 to max_lag using numpy for more accuracy."""

        # Convert to numpy for more control over computation
        def compute_acf(batch_tensor) -> np.ndarray:
            # Convert to numpy array
            batch_np = batch_tensor.numpy()
            result = np.zeros((batch_np.shape[0], self.max_lag + 1), dtype=np.float32)

            # For each series in the batch
            for b in range(batch_np.shape[0]):
                x = batch_np[b]

                # Mean and standard deviation
                mean_x = np.mean(x)
                std_x = np.std(x)

                # Normalize series
                x_norm = (x - mean_x) / (std_x + 1e-10)

                # Lag 0 autocorrelation is 1
                result[b, 0] = 1.0

                # Compute autocorrelation for each lag
                for lag in range(1, self.max_lag + 1):
                    # For lag correlations, ensure we're comparing elements
                    # at the same positions
                    if len(x_norm[lag:]) > 0:
                        corr = np.corrcoef(x_norm[lag:], x_norm[:-lag])[0, 1]
                        result[b, lag] = corr

            return result

        # Apply the computation
        acf = tf.py_function(compute_acf, [series], tf.float32)

        # Set the shape
        acf.set_shape([series.shape[0], self.max_lag + 1])

        return acf

    def _select_lags(self, acf) -> tf.Tensor:
        """Select lags based on autocorrelation values."""
        # Use batch mean autocorrelation for lag selection
        mean_acf = tf.reduce_mean(acf, axis=0)

        if self.method == "top_k":
            # Select top k lags (excluding lag 0)
            _, indices = tf.math.top_k(tf.abs(mean_acf[1:]), k=self.n_lags)
            # Add 1 to indices since we excluded lag 0
            selected_lags = indices + 1
        else:  # threshold
            # Select lags with autocorrelation above threshold (excluding lag 0)
            above_threshold = tf.where(tf.abs(mean_acf[1:]) > self.threshold)
            # Add 1 to indices since we excluded lag 0
            selected_lags = above_threshold + 1

            # If too few lags are above threshold, fall back to top_k
            if tf.shape(selected_lags)[0] < 1:
                _, indices = tf.math.top_k(tf.abs(mean_acf[1:]), k=self.n_lags)
                selected_lags = indices + 1

        # Sort lags in ascending order for interpretability
        return tf.sort(selected_lags)

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute the output shape."""
        output_shape = list(input_shape)

        # Calculate the number of output features
        if len(output_shape) == 2:
            # For 2D input, add feature dimension
            feature_dim = 0
            if self.keep_original:
                feature_dim += 1
            feature_dim += self.n_lags
            output_shape.append(feature_dim)
        else:
            # For 3D input, update the feature dimension
            feature_dim = output_shape[-1]
            feature_dim = (
                feature_dim + feature_dim * self.n_lags
                if self.keep_original
                else feature_dim * self.n_lags
            )
            output_shape[-1] = feature_dim

        # Update batch dimension if dropping rows
        if self.drop_na:
            # Adjust batch dimension based on the maximum lag
            if hasattr(self, "selected_lags") and self.selected_lags is not None:
                if isinstance(self.selected_lags, tf.Tensor):
                    max_lag = tf.reduce_max(self.selected_lags).numpy()
                else:
                    max_lag = max(self.selected_lags)
            else:
                # If selected_lags not known, fall back to max_lag
                max_lag = self.max_lag

            if output_shape[0] is not None:
                output_shape[0] = max(
                    1,
                    output_shape[0] - max_lag,
                )  # Ensure batch size is at least 1
        return tuple(output_shape)

    def get_config(self) -> dict:
        """Return the configuration."""
        config = {
            "max_lag": self.max_lag,
            "n_lags": self.n_lags,
            "threshold": self.threshold,
            "method": self.method,
            "drop_na": self.drop_na,
            "fill_value": self.fill_value,
            "keep_original": self.keep_original,
        }
        base_config = super().get_config()
        return {**base_config, **config}
