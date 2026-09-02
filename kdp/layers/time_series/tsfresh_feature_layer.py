import keras
import tensorflow as tf
from keras.layers import Layer
import numpy as np


@keras.saving.register_keras_serializable(package="kdp.layers")
class TSFreshFeatureLayer(Layer):
    """Layer for extracting statistical features from time series data.

    This layer extracts statistical features inspired by the tsfresh library,
    such as mean, std, min, max, quantiles, energy, and more complex features
    like number of peaks, autocorrelation, and trend coefficients.

    Args:
        features: List of statistical features to extract. Options:
            - 'mean': Mean of the time series
            - 'std': Standard deviation of the time series
            - 'min': Minimum value of the time series
            - 'max': Maximum value of the time series
            - 'median': Median value of the time series
            - 'sum': Sum of values in the time series
            - 'energy': Sum of squares of values
            - 'iqr': Interquartile range (75% - 25% quantile)
            - 'kurtosis': Kurtosis (4th moment - peakedness of distribution)
            - 'skewness': Skewness (3rd moment - asymmetry of distribution)
            - 'abs_energy': Sum of absolute values
            - 'abs_mean': Mean of absolute values
            - 'count_above_mean': Number of values above mean
            - 'count_below_mean': Number of values below mean
            - 'first_location_of_max': Position of the first maximum as a
              fraction of the series length
            - 'first_location_of_min': Position of the first minimum as a
              fraction of the series length
            - 'quantile_05': 5% quantile
            - 'quantile_25': 25% quantile
            - 'quantile_50': 50% quantile (median)
            - 'quantile_75': 75% quantile
            - 'quantile_95': 95% quantile
            - 'linear_trend_coef': Linear trend coefficients (slope, intercept)
            - 'peak_count': Peaks (local maxima) per sample, i.e. the count
              divided by the series length
            - 'valley_count': Valleys (local minima) per sample, likewise
              divided by the series length
            - 'fft_coef_n': First n FFT coefficients
            - 'autocorrelation_lag_n': Autocorrelation at lag n
        window_size: Size of rolling window for feature extraction (default: None,
            which means to compute features over the entire series)
        stride: Step size for sliding window (default: 1)
        drop_na: Whether to drop rows with NaN values (default: True)
        normalize: Whether to normalize features (default: False)
    """

    def __init__(
        self,
        features=None,
        window_size=None,
        stride=1,
        drop_na=True,
        normalize=False,
        **kwargs,
    ):
        """Initialize the TSFreshFeatureLayer.

        See the class docstring for the accepted arguments and what
        each one controls.
        """
        super().__init__(**kwargs)

        # Default features if none provided
        if features is None:
            self.features = [
                "mean",
                "std",
                "min",
                "max",
                "median",
                "iqr",
                "count_above_mean",
                "count_below_mean",
            ]
        else:
            self.features = features

        self.window_size = window_size
        self.stride = stride
        self.drop_na = drop_na
        self.normalize = normalize

        # Validate features
        valid_features = [
            "mean",
            "std",
            "min",
            "max",
            "median",
            "sum",
            "energy",
            "iqr",
            "kurtosis",
            "skewness",
            "abs_energy",
            "abs_mean",
            "count_above_mean",
            "count_below_mean",
            "first_location_of_max",
            "first_location_of_min",
            "quantile_05",
            "quantile_25",
            "quantile_50",
            "quantile_75",
            "quantile_95",
            "linear_trend_coef",
            "peak_count",
            "valley_count",
        ]

        # Validate each feature
        for feature in self.features:
            base_feature = feature

            # Handle parameterized features like fft_coef_n or autocorrelation_lag_n
            if "_" in feature and feature.split("_")[0] in ["fft", "autocorrelation"]:
                base_feature = "_".join(feature.split("_")[:-1])

            if base_feature not in valid_features and not (
                base_feature.startswith("fft_coef")
                or base_feature.startswith("autocorrelation_lag")
            ):
                raise ValueError(f"Invalid feature: {feature}")

    def build(self, input_shape) -> None:
        """Build the layer's weights for a given input shape.

        Args:
            input_shape: Shape of the input tensor.
        """
        super().build(input_shape)

    def call(self, inputs, training=None) -> tf.Tensor:
        """Extract statistical features from time series data.

        Args:
            inputs: Input tensor of shape (batch_size, time_steps) or (batch_size, time_steps, features)
            training: Boolean tensor indicating whether the call is for training

        Returns:
            Tensor with extracted statistical features
        """

        # Process the input tensor using NumPy for more control over feature extraction
        def extract_tsfresh_features(inputs_tensor) -> np.ndarray:
            # Convert to NumPy
            inputs_np = inputs_tensor.numpy()

            # Get dimensions
            if len(inputs_np.shape) == 2:
                batch_size, time_steps = inputs_np.shape
                n_features = 1
                # Reshape to 3D for consistent processing
                inputs_np = inputs_np.reshape(batch_size, time_steps, 1)
            else:
                batch_size, time_steps, n_features = inputs_np.shape

            # Determine if we're using windows
            if self.window_size is None:
                window_size = time_steps
                n_windows = 1
                stride = 1
            else:
                window_size = min(self.window_size, time_steps)
                stride = self.stride
                n_windows = (time_steps - window_size) // stride + 1

            # Calculate number of output features
            n_output_per_feature = self._get_n_output_features()
            n_output_features = n_features * n_output_per_feature

            # Initialize output array
            result = np.zeros(
                (batch_size, n_windows, n_output_features),
                dtype=np.float32,
            )

            # Process each sample in the batch
            for b in range(batch_size):
                # Process each window
                for w in range(n_windows):
                    start_idx = w * stride
                    end_idx = start_idx + window_size

                    feature_idx = 0
                    # Process each input feature
                    for f in range(n_features):
                        # Get window data for this feature
                        window_data = inputs_np[b, start_idx:end_idx, f]

                        # Extract features
                        feature_values = self._compute_features(window_data)

                        # Store in result
                        for value in feature_values:
                            if isinstance(value, np.ndarray):
                                for v in value:
                                    result[b, w, feature_idx] = v
                                    feature_idx += 1
                            else:
                                result[b, w, feature_idx] = value
                                feature_idx += 1

            # If window_size = time_steps, squeeze out the window dimension
            if self.window_size is None:
                result = result.reshape(batch_size, n_output_features)

            # Apply normalization if requested
            if self.normalize:
                # Normalize each feature column separately
                for i in range(n_output_features):
                    feature_col = (
                        result[:, :, i]
                        if self.window_size is not None
                        else result[:, i]
                    )
                    feature_min = np.min(feature_col)
                    feature_max = np.max(feature_col)
                    if feature_max > feature_min:
                        if self.window_size is not None:
                            result[:, :, i] = (feature_col - feature_min) / (
                                feature_max - feature_min
                            )
                        else:
                            result[:, i] = (feature_col - feature_min) / (
                                feature_max - feature_min
                            )

            return result

        # Apply the function
        result = tf.py_function(extract_tsfresh_features, [inputs], tf.float32)

        # Set the shape
        if self.window_size is None:
            # Single window case
            n_output_features = self._get_n_output_features()

            if len(inputs.shape) == 2:
                # Single feature input
                result.set_shape([inputs.shape[0], n_output_features])
            else:
                # Multi-feature input
                result.set_shape([inputs.shape[0], inputs.shape[2] * n_output_features])
        else:
            # Multiple windows case
            n_output_features = self._get_n_output_features()
            time_steps = inputs.shape[1]
            n_windows = (time_steps - self.window_size) // self.stride + 1

            if len(inputs.shape) == 2:
                # Single feature input
                result.set_shape([inputs.shape[0], n_windows, n_output_features])
            else:
                # Multi-feature input
                result.set_shape(
                    [inputs.shape[0], n_windows, inputs.shape[2] * n_output_features],
                )

        return result

    def _compute_features(self, series) -> list:
        """Compute statistical features for a single time series."""
        results = []

        # Handle NaN values: drop them, or replace them with 0.
        series = (
            series[~np.isnan(series)]
            if self.drop_na
            else np.nan_to_num(series, nan=0.0)
        )

        # Skip empty series
        if len(series) == 0:
            return [0.0] * self._get_n_output_features()

        # Precompute common statistics
        series_mean = np.mean(series)
        series_std = np.std(series)
        series_min = np.min(series)
        series_max = np.max(series)
        series_median = np.median(series)

        # Extract requested features
        for feature in self.features:
            if feature == "mean":
                results.append(series_mean)

            elif feature == "std":
                results.append(series_std)

            elif feature == "min":
                results.append(series_min)

            elif feature == "max":
                results.append(series_max)

            elif feature == "median":
                results.append(series_median)

            elif feature == "sum":
                results.append(np.sum(series))

            elif feature == "energy":
                results.append(np.sum(series**2))

            elif feature == "iqr":
                q75, q25 = np.percentile(series, [75, 25])
                results.append(q75 - q25)

            elif feature == "kurtosis":
                # Kurtosis (using Fisher's definition)
                if len(series) > 3 and series_std > 0:
                    n = len(series)
                    m4 = np.sum((series - series_mean) ** 4) / n
                    kurt = m4 / (series_std**4) - 3  # Excess kurtosis
                    results.append(kurt)
                else:
                    results.append(0.0)

            elif feature == "skewness":
                # Skewness
                if len(series) > 2 and series_std > 0:
                    n = len(series)
                    m3 = np.sum((series - series_mean) ** 3) / n
                    skew = m3 / (series_std**3)
                    results.append(skew)
                else:
                    results.append(0.0)

            elif feature == "abs_energy":
                results.append(np.sum(np.abs(series)))

            elif feature == "abs_mean":
                results.append(np.mean(np.abs(series)))

            elif feature == "count_above_mean":
                results.append(np.sum(series > series_mean))

            elif feature == "count_below_mean":
                results.append(np.sum(series < series_mean))

            elif feature == "first_location_of_max":
                results.append(np.argmax(series) / len(series))

            elif feature == "first_location_of_min":
                results.append(np.argmin(series) / len(series))

            elif feature.startswith("quantile_"):
                q = int(feature.split("_")[1]) / 100.0
                results.append(np.percentile(series, q * 100))

            elif feature == "linear_trend_coef":
                # Linear trend coefficients
                x = np.arange(len(series))
                if len(x) > 1:
                    # Add a column of ones for the intercept
                    design_matrix = np.vstack([x, np.ones(len(x))]).T

                    # Solve the least squares problem
                    try:
                        slope, intercept = np.linalg.lstsq(
                            design_matrix,
                            series,
                            rcond=None,
                        )[0]
                        results.append(np.array([slope, intercept]))
                    except np.linalg.LinAlgError:
                        results.append(np.array([0.0, 0.0]))
                else:
                    results.append(np.array([0.0, 0.0]))

            elif feature == "peak_count":
                # Count peaks (local maxima)
                if len(series) > 2:
                    # A point is a peak if it's greater than both neighbors
                    peaks = np.where(
                        (series[1:-1] > series[:-2]) & (series[1:-1] > series[2:]),
                    )[0]
                    results.append(len(peaks) / len(series))
                else:
                    results.append(0.0)

            elif feature == "valley_count":
                # Count valleys (local minima)
                if len(series) > 2:
                    # A point is a valley if it's less than both neighbors
                    valleys = np.where(
                        (series[1:-1] < series[:-2]) & (series[1:-1] < series[2:]),
                    )[0]
                    results.append(len(valleys) / len(series))
                else:
                    results.append(0.0)

            elif feature.startswith("fft_coef_"):
                # Extract FFT coefficients
                n_coefs = int(feature.split("_")[-1])
                if len(series) > 1:
                    fft_values = np.fft.fft(series - np.mean(series))
                    amplitudes = np.abs(fft_values)[:n_coefs]
                    # Pad with zeros if needed
                    if len(amplitudes) < n_coefs:
                        amplitudes = np.pad(amplitudes, (0, n_coefs - len(amplitudes)))
                    # Normalize
                    if np.sum(amplitudes) > 0:
                        amplitudes = amplitudes / np.sum(amplitudes)
                    results.append(amplitudes)
                else:
                    results.append(np.zeros(n_coefs))

            elif feature.startswith("autocorrelation_lag_"):
                # Compute autocorrelation at the specified lag
                lag = int(feature.split("_")[-1])
                if len(series) > lag:
                    # Mean-center the series
                    centered = series - series_mean
                    # Compute autocorrelation
                    if np.sum(centered**2) > 0:
                        autocorr = np.correlate(centered, centered, mode="full")
                        # Normalize
                        autocorr = autocorr / np.max(autocorr)
                        # Extract the specified lag
                        middle = len(autocorr) // 2
                        lag_value = autocorr[middle + lag]
                        results.append(lag_value)
                    else:
                        results.append(0.0)
                else:
                    results.append(0.0)

        return results

    def _get_n_output_features(self) -> int:
        """Calculate the number of output features."""
        n_features = 0

        for feature in self.features:
            if feature == "linear_trend_coef":
                n_features += 2  # Slope and intercept
            elif feature.startswith("fft_coef_"):
                n_coefs = int(feature.split("_")[-1])
                n_features += n_coefs
            else:
                n_features += 1

        return n_features

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute the output shape of the layer."""
        n_output_features = self._get_n_output_features()

        if len(input_shape) == 2:
            batch_size, time_steps = input_shape
            n_input_features = 1
        else:
            batch_size, time_steps, n_input_features = input_shape

        n_output_features *= n_input_features

        if self.window_size is None:
            # Single window over entire series
            return (batch_size, n_output_features)
        else:
            # Multiple windows
            window_size = min(self.window_size, time_steps)
            n_windows = (time_steps - window_size) // self.stride + 1
            return (batch_size, n_windows, n_output_features)

    def get_config(self) -> dict:
        """Return the configuration of the layer."""
        config = {
            "features": self.features,
            "window_size": self.window_size,
            "stride": self.stride,
            "drop_na": self.drop_na,
            "normalize": self.normalize,
        }
        base_config = super().get_config()
        return {**base_config, **config}
