"""Distribution-aware encoder layer for TensorFlow models.

This implementation automatically detects various data distributions and applies
appropriate transformations for better model performance. It is built on Keras
without TensorFlow Probability dependencies, making it lighter and easier to deploy.

The layer can detect normal, heavy-tailed, multimodal, uniform, exponential,
log-normal, discrete, periodic, sparse, beta, gamma, poisson, cauchy,
zero-inflated, bounded, and ordinal distributions.

Example usage:
    ```python
    import keras

    from kdp.layers import DistributionAwareEncoder

    # Creating a model with automatic distribution detection
    inputs = keras.Input(shape=(10,))
    encoded = DistributionAwareEncoder(embedding_dim=16)(inputs)
    outputs = keras.layers.Dense(1)(encoded)
    model = keras.Model(inputs, outputs)

    # Save and load model with custom objects
    model.save("my_model.keras")
    custom_objects = DistributionAwareEncoder.get_custom_objects()
    loaded_model = keras.saving.load_model("my_model", custom_objects=custom_objects)
    ```
"""

from enum import Enum
from typing import Any

import keras
import tensorflow as tf
from loguru import logger
import inspect

try:
    from kdp.layers.distribution_transform_layer import DistributionTransformLayer
except ImportError:
    # Fallback path
    try:
        from ..distribution_transform_layer import DistributionTransformLayer
    except ImportError:
        raise ImportError(
            "Could not import DistributionTransformLayer. Please ensure it's in the correct path.",
        )


class DistributionType(str, Enum):
    """Supported distribution types for feature encoding."""

    NORMAL = "normal"
    HEAVY_TAILED = "heavy_tailed"
    MULTIMODAL = "multimodal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    LOG_NORMAL = "log_normal"
    DISCRETE = "discrete"
    PERIODIC = "periodic"
    SPARSE = "sparse"
    BETA = "beta"  # For bounded data with two shape parameters
    GAMMA = "gamma"  # For positive, right-skewed data
    POISSON = "poisson"  # For count data
    CAUCHY = "cauchy"  # For extremely heavy-tailed data
    ZERO_INFLATED = "zero_inflated"  # For data with excess zeros
    BOUNDED = "bounded"  # For data with known bounds
    ORDINAL = "ordinal"  # For ordered categorical data

    # BOUNDED, ORDINAL and POISSON can be requested explicitly, and the encoder
    # will honour them, but `distribution_type="auto"` never returns them: the
    # detector scores no evidence for those three, so they cannot come out of
    # automatic detection.


@keras.saving.register_keras_serializable(package="kdp.layers")
class DistributionAwareEncoder(keras.layers.Layer):
    """An advanced layer that adapts its encoding based on the input distribution.

    This layer automatically detects and handles various distribution types using
    the DistributionTransformLayer. It identifies the data distribution type and
    applies appropriate transformations for better model performance.

    Key features:
    1. Auto-detection of distribution types to apply optimal transformations
    2. Periodic feature detection and automatic encoding with Fourier features (sin/cos)
    3. Optional projection to a fixed embedding dimension
    4. Distribution-specific embeddings that can be added to the outputs
    5. Graph mode compatibility for use in TensorFlow's static graph execution

    Supported distributions include:
    - Normal distributions: For normally distributed data
    - Heavy-tailed distributions: For data with heavier tails than normal
    - Multimodal distributions: For data with multiple peaks
    - Uniform distributions: For evenly distributed data
    - Exponential distributions: For data with exponential decay
    - Log-normal distributions: For data that is normal after log transform
    - Discrete distributions: For data with finite distinct values
    - Periodic distributions: For data with cyclic patterns (adds sin/cos features)
    - Sparse distributions: For data with many zeros
    - Beta distributions: For bounded data between 0 and 1
    - Gamma distributions: For positive, right-skewed data
    - Poisson distributions: For count data
    - Cauchy distributions: For extremely heavy-tailed data
    - Zero-inflated distributions: For data with excess zeros
    - Bounded distributions: For data with known bounds
    - Ordinal distributions: For ordered categorical data

    The layer uses pure TensorFlow operations without dependencies on TensorFlow Probability,
    and is compatible with both eager execution and graph mode.
    """

    def __init__(
        self,
        embedding_dim=None,
        epsilon=1e-6,
        detect_periodicity=True,
        handle_sparsity=True,
        auto_detect=True,
        distribution_type="unknown",
        transform_type="auto",
        add_distribution_embedding=False,
        name="distribution_aware_encoder",
        trainable=True,
        # Legacy parameters for backward compatibility
        num_bins=None,
        adaptive_binning=None,
        mixture_components=None,
        prefered_distribution=None,
        **kwargs,
    ):
        """Initialize the DistributionAwareEncoder.

        Args:
            embedding_dim: Optional output dimension for feature projection. If specified,
                a Dense layer will project the transformed features to this dimension.
                If None, the original feature dimension is preserved. Default is None.
            epsilon: Small value to prevent numerical issues. Default is 1e-6.
            detect_periodicity: If True, the layer will check for and handle periodic patterns
                by adding sin/cos features, increasing the output dimension. When True, periodic
                data will have 3x the original feature dimension. Default is True.
            handle_sparsity: Accepted and not used. Sparse data is
                detected and handled as one of the distributions the
                encoder recognises, whatever this says.
            auto_detect: If True, automatically detect the distribution type during training.
                If False, the specified distribution_type will be used. Default is True.
            distribution_type: The specific distribution type to use if auto_detect is False.
                Must be one of the values in DistributionType. Default is "unknown".
            transform_type: Type of transformation to apply via DistributionTransformLayer.
                Options include "none", "log", "sqrt", "box-cox", etc. Default is "auto".
            add_distribution_embedding: If True, adds a learned embedding for the detected
                distribution type to the output, increasing the output dimension. Default is False.
            name: Name for the layer. Default is "distribution_aware_encoder".
            trainable: Whether the layer is trainable. Default is True.

            # Legacy parameters (maintained for backward compatibility)
            num_bins: Number of bins for legacy histogram-based encoding. Not used in current implementation.
            adaptive_binning: Whether to use adaptive binning. Not used in current implementation.
            mixture_components: Number of mixture components. Not used in current implementation.
            prefered_distribution: Legacy way to specify distribution_type. If provided, auto_detect
                will be set to False and this value will be used as distribution_type.
            **kwargs: Additional arguments forwarded to `keras.layers.Layer`.

        Note on output dimensions:
            - If detect_periodicity=True and periodic features are detected/forced:
                output_dim = input_dim * 3 (original + sin + cos features)
            - If embedding_dim is specified:
                output_dim = embedding_dim
            - If add_distribution_embedding=True:
                output_dim += 8 (distribution embedding dimension)
        """
        super().__init__(name=name, trainable=trainable, **kwargs)

        # Initialize class attributes to avoid attribute errors
        self._added_periodic_features = False
        self._current_dist_type = None
        self._is_initialized = False

        # Handle legacy parameters
        if prefered_distribution is not None:
            # If prefered_distribution is specified, use it and disable auto-detect
            self.distribution_type = prefered_distribution
            self.auto_detect = False
            logger.info(
                f"Using specified distribution type: {prefered_distribution} (legacy parameter)",
            )
        else:
            self.distribution_type = distribution_type
            self.auto_detect = auto_detect

        # Store other parameters
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.detect_periodicity = detect_periodicity
        self.handle_sparsity = handle_sparsity
        self.transform_type = transform_type
        self.add_distribution_embedding = add_distribution_embedding

        # Store legacy parameters (not used, but needed for get_config)
        self.num_bins = num_bins
        self.adaptive_binning = adaptive_binning
        self.mixture_components = mixture_components
        self.prefered_distribution = prefered_distribution

        # Define valid distribution types
        self._valid_distributions = [dist.value for dist in DistributionType]

        # Validate distribution_type
        if (
            self.distribution_type not in self._valid_distributions
            and self.distribution_type != "unknown"
        ):
            logger.warning(
                f"Unknown distribution type: {self.distribution_type}. Defaulting to 'normal'",
            )
            self.distribution_type = "normal"

        # Initialize layers
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize internal layers."""
        # Check accepted parameters for DistributionTransformLayer
        transform_kwargs = {}
        transform_kwargs["transform_type"] = self.transform_type
        # Use epsilon if it's a valid parameter
        try:
            transform_signature = inspect.signature(DistributionTransformLayer.__init__)
            if "epsilon" in transform_signature.parameters:
                transform_kwargs["epsilon"] = self.epsilon
        except (TypeError, ValueError) as exc:
            logger.debug(
                f"Could not inspect DistributionTransformLayer signature ({exc}); "
                "falling back to its default parameters.",
            )

        # The DistributionTransformLayer handles most of the distribution transformations
        self.distribution_transform = DistributionTransformLayer(
            **transform_kwargs,
            name=f"{self.name}_transform",
        )

        # Projection layer for embedding dimension if specified
        if self.embedding_dim is not None:
            self.projection = keras.layers.Dense(
                self.embedding_dim,
                activation="relu",
                name=f"{self.name}_projection",
            )
        else:
            self.projection = None

    def build(self, input_shape) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of input tensor
        """
        # Create a variable to store the detected distribution
        if self.auto_detect:
            self.detected_distribution = self.add_weight(
                name="detected_distribution",
                shape=(1,),
                dtype="int32",
                trainable=False,
                initializer="zeros",
            )

        # Distribution embedding if needed
        if self.add_distribution_embedding:
            num_distributions = len(self._valid_distributions)
            self.distribution_embedding = self.add_weight(
                name="distribution_embedding",
                shape=(
                    num_distributions,
                    8,
                ),  # 8-dimensional embedding for each distribution
                initializer="glorot_uniform",
                trainable=True,
            )

        # For periodic data handling
        if self.detect_periodicity:
            self.frequency = self.add_weight(
                name="frequency",
                shape=(1,),
                initializer="ones",
                trainable=True,
            )
            self.phase = self.add_weight(
                name="phase",
                shape=(1,),
                initializer="zeros",
                trainable=True,
            )

        # Build the distribution transform layer
        self.distribution_transform.build(input_shape)

        # Build the projection layer if needed
        if self.projection is not None:
            transformed_shape = self.distribution_transform.compute_output_shape(
                input_shape,
            )
            self.projection.build(transformed_shape)

        # Fix the super().build call
        super().build(input_shape)

    def _calculate_statistics(self, x) -> dict:
        """Calculate statistics for distribution detection.

        Args:
            x: Input tensor

        Returns:
            Dictionary of statistics
        """
        # Basic statistics
        mean = tf.reduce_mean(x, axis=0)
        variance = tf.math.reduce_variance(x)
        std = tf.sqrt(variance + self.epsilon)

        # Standardize for higher moments
        x_std = (x - mean) / (std + self.epsilon)

        # Calculate skewness: E[(x-μ)³]/σ³
        skewness = tf.reduce_mean(tf.pow(x_std, 3), axis=0)

        # Calculate kurtosis: E[(x-μ)⁴]/σ⁴
        kurtosis = tf.reduce_mean(tf.pow(x_std, 4), axis=0)

        # Range statistics
        min_val = tf.reduce_min(x)
        max_val = tf.reduce_max(x)

        # Check for zeros and sparsity
        is_zero = tf.abs(x) < self.epsilon
        zero_ratio = tf.reduce_mean(tf.cast(is_zero, tf.float32))

        # Check for discreteness
        unique_values, _ = tf.unique(tf.reshape(x, [-1]))
        unique_ratio = tf.cast(tf.size(unique_values), tf.float32) / tf.cast(
            tf.size(x),
            tf.float32,
        )

        return {
            "mean": mean,
            "variance": variance,
            "std": std,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "min": min_val,
            "max": max_val,
            "zero_ratio": zero_ratio,
            "unique_ratio": unique_ratio,
            "has_negative": tf.reduce_any(x < 0),
            "is_bounded_01": tf.logical_and(
                tf.reduce_all(x >= 0),
                tf.reduce_all(x <= 1),
            ),
        }

    def _detect_distribution(self, x) -> tf.Tensor:
        """Detect the distribution type of the data.

        Args:
            x: Input tensor

        Returns:
            int: Index of the detected distribution type
        """
        # Calculate statistics
        stats = self._calculate_statistics(x)

        # Extract key metrics
        skewness = tf.reduce_mean(tf.abs(stats["skewness"]))
        kurtosis = tf.reduce_mean(stats["kurtosis"])
        zero_ratio = stats["zero_ratio"]
        has_negative = stats["has_negative"]
        is_bounded_01 = stats["is_bounded_01"]
        unique_ratio = stats["unique_ratio"]

        # Check for periodicity if enabled - store as a tensor for graph compatibility
        is_periodic = tf.constant(False, dtype=tf.bool)
        if self.detect_periodicity:
            is_periodic = tf.cast(self._check_periodicity(x), tf.bool)

        # Check for multimodality - store as a tensor for graph compatibility
        is_multimodal = tf.cast(self._check_multimodality(x, stats), tf.bool)

        # Use a scoring system to determine distribution type
        # Initialize scores for each distribution
        dist_scores = {}
        for dist_type in self._valid_distributions:
            dist_scores[dist_type] = tf.constant(0.0, dtype=tf.float32)

        # Sparse score - Use tf.cond
        dist_scores["sparse"] = tf.cond(
            zero_ratio > 0.5,
            lambda: tf.constant(100.0),
            lambda: tf.constant(0.0),
        )

        # Periodic score - Use tf.cond for tensor values
        dist_scores["periodic"] = tf.cond(
            is_periodic,
            lambda: tf.constant(90.0),
            lambda: tf.constant(0.0),
        )

        # Discrete score - Use tf.cond
        dist_scores["discrete"] = tf.cond(
            unique_ratio < 0.1,
            lambda: tf.constant(80.0),
            lambda: tf.constant(0.0),
        )

        # Beta score
        bounded_and_skewed = tf.logical_and(is_bounded_01, skewness > 0.5)
        dist_scores["beta"] = tf.cond(
            bounded_and_skewed,
            lambda: tf.constant(75.0),
            lambda: tf.constant(0.0),
        )

        # Uniform score
        dist_scores["uniform"] = tf.cond(
            is_bounded_01,
            lambda: tf.constant(70.0),
            lambda: tf.constant(0.0),
        )

        # Log-normal score
        positive_and_very_skewed = tf.logical_and(
            tf.logical_not(has_negative),
            skewness > 2.0,
        )
        dist_scores["log_normal"] = tf.cond(
            positive_and_very_skewed,
            lambda: tf.constant(65.0),
            lambda: tf.constant(0.0),
        )

        # Exponential score
        positive_and_skewed = tf.logical_and(
            tf.logical_not(has_negative),
            skewness > 1.0,
        )
        dist_scores["exponential"] = tf.cond(
            positive_and_skewed,
            lambda: tf.constant(60.0),
            lambda: tf.constant(0.0),
        )

        # Gamma score
        positive_and_mild_skew = tf.logical_and(
            tf.logical_not(has_negative),
            skewness > 0.5,
        )
        dist_scores["gamma"] = tf.cond(
            positive_and_mild_skew,
            lambda: tf.constant(55.0),
            lambda: tf.constant(0.0),
        )

        # Cauchy score
        dist_scores["cauchy"] = tf.cond(
            kurtosis > 10.0,
            lambda: tf.constant(50.0),
            lambda: tf.constant(0.0),
        )

        # Heavy-tailed score
        dist_scores["heavy_tailed"] = tf.cond(
            kurtosis > 4.0,
            lambda: tf.constant(45.0),
            lambda: tf.constant(0.0),
        )

        # Multimodal score - Use tf.cond for tensor values
        dist_scores["multimodal"] = tf.cond(
            is_multimodal,
            lambda: tf.constant(40.0),
            lambda: tf.constant(0.0),
        )

        # Zero-inflated score
        moderate_zeros = tf.logical_and(zero_ratio > 0.3, zero_ratio <= 0.5)
        dist_scores["zero_inflated"] = tf.cond(
            moderate_zeros,
            lambda: tf.constant(35.0),
            lambda: tf.constant(0.0),
        )

        # Normal score - default option if nothing else fits well
        normal_like = tf.logical_and(skewness < 0.5, tf.abs(kurtosis - 3.0) < 1.0)
        dist_scores["normal"] = tf.cond(
            normal_like,
            lambda: tf.constant(30.0),
            lambda: tf.constant(
                20.0,
            ),  # Give some baseline score to normal distribution
        )

        # Find the distribution with the highest score using TensorFlow operations
        # Create tensors for scores and distribution types
        score_values = []
        dist_types = []

        for dist_type in self._valid_distributions:
            score_values.append(dist_scores[dist_type])
            dist_types.append(dist_type)

        # Convert to tensors
        score_values_tensor = tf.stack(score_values)

        # Find the index of the maximum score
        max_score_idx = tf.argmax(score_values_tensor)

        # Get the selected distribution type
        # For graph mode compatibility, we'll handle this differently
        if tf.executing_eagerly():
            # In eager mode, we can directly use the index
            selected_dist = dist_types[max_score_idx.numpy()]
            dist_idx = self._valid_distributions.index(selected_dist)
        else:
            # In graph mode, we'll use the index directly
            dist_idx = max_score_idx
            # For logging, use the first valid distribution as a fallback
            selected_dist = self._valid_distributions[0]

        # For logging purposes only, calculate these values
        if tf.executing_eagerly():
            logger.debug(
                f"Detected distribution: {selected_dist} with "
                f"skewness={skewness.numpy():.2f}, kurtosis={kurtosis.numpy():.2f}, "
                f"zero_ratio={zero_ratio.numpy():.2f}, unique_ratio={unique_ratio.numpy():.2f}",
            )

        return dist_idx

    def _check_periodicity(self, x, max_lag=20, threshold=0.2) -> tf.Tensor:
        """Check if the input data has periodicity by analyzing autocorrelation.

        This method detects periodic patterns in the data by:
        1. Computing the autocorrelation function (ACF) up to max_lag
        2. Finding peaks in the ACF that exceed the threshold
        3. Checking if these peaks have a consistent spacing (indicating periodicity)

        Args:
            x: Input tensor of shape (batch_size, features)
            max_lag: Maximum lag to consider for autocorrelation. Default is 20.
            threshold: Threshold for peak detection in ACF. Default is 0.2.

        Returns:
            Boolean tensor indicating if periodicity was detected

        Note:
            When periodicity is detected, the encoder will add sine and cosine features
            to the output, effectively tripling the feature dimension.
        """
        # Flatten and ensure float type with proper shape
        x_flat = tf.cast(tf.reshape(x, [-1]), tf.float32)
        n = tf.shape(x_flat)[0]

        # For graph mode compatibility, we need to avoid using tensor values in conditionals
        # Instead, we'll use tf.cond for all conditionals

        # If data is too short, return False
        return tf.cond(
            tf.less(n, 3 * max_lag),
            lambda: tf.constant(False),
            lambda: self._compute_periodicity(x_flat, n, max_lag, threshold),
        )

    def _compute_periodicity(self, x_flat, n, max_lag, threshold) -> tf.Tensor:
        """Helper function to compute periodicity for graph mode compatibility."""
        # Center the data
        x_centered = x_flat - tf.reduce_mean(x_flat)

        # Calculate variance
        variance = tf.reduce_sum(tf.square(x_centered))

        # Skip if no variation
        return tf.cond(
            tf.less(variance, self.epsilon),
            lambda: tf.constant(False),
            lambda: self._compute_autocorrelation(
                x_centered,
                n,
                max_lag,
                variance,
                threshold,
            ),
        )

    def _compute_autocorrelation(
        self,
        x_centered,
        n,
        max_lag,
        variance,
        threshold,
    ) -> tf.Tensor:
        """Helper function to compute autocorrelation."""
        # Use TensorArray to collect autocorrelation values
        acf_array = tf.TensorArray(tf.float32, size=max_lag)

        # Compute autocorrelation for different lags
        i = tf.constant(0)

        # Use while_loop for graph compatibility
        def loop_cond(i, *args: Any) -> tf.Tensor:
            return tf.logical_and(tf.less(i, max_lag), tf.less(i, n // 3))

        def loop_body(i, acf_array) -> list:
            # Make sure we don't go out of bounds
            lag = i + 1  # Start from lag 1

            # Calculate correlation at this lag
            y1 = x_centered[lag:]
            y2 = x_centered[:-lag]

            # Ensure y1 and y2 have the same shape
            min_len = tf.minimum(tf.shape(y1)[0], tf.shape(y2)[0])
            y1 = y1[:min_len]
            y2 = y2[:min_len]

            corr = tf.reduce_sum(y1 * y2) / (variance + self.epsilon)
            acf_array = acf_array.write(i, corr)

            return [i + 1, acf_array]

        # Run the loop
        _, acf_array = tf.while_loop(loop_cond, loop_body, [i, acf_array])

        # Convert to tensor
        acf_tensor = acf_array.stack()

        # Check if we collected any values
        return tf.cond(
            tf.equal(tf.shape(acf_tensor)[0], 0),
            lambda: tf.constant(False),
            lambda: self._find_peaks(acf_tensor, threshold),
        )

    def _find_peaks(self, acf_tensor, threshold) -> tf.Tensor:
        """Find peaks in autocorrelation values."""
        # Need at least 3 points for meaningful peaks
        return tf.cond(
            tf.less(tf.shape(acf_tensor)[0], 3),
            lambda: tf.constant(False),
            lambda: self._check_significant_peaks(acf_tensor, threshold),
        )

    def _check_significant_peaks(self, acf_tensor, threshold) -> tf.Tensor:
        """Check if there are significant peaks in the autocorrelation."""
        # Create is_greater_left and is_greater_right tensors
        is_greater_left = tf.concat([[True], acf_tensor[1:] > acf_tensor[:-1]], axis=0)
        is_greater_right = tf.concat([acf_tensor[:-1] > acf_tensor[1:], [True]], axis=0)
        is_peak = tf.logical_and(is_greater_left, is_greater_right)

        # Check if peaks are above threshold
        significant_peaks = tf.logical_and(is_peak, acf_tensor > threshold)
        num_peaks = tf.reduce_sum(tf.cast(significant_peaks, tf.int32))

        # Consider periodic if at least one significant peak
        return num_peaks >= 1

    def _check_multimodality(self, x, stats, num_bins=100) -> tf.Tensor:
        """Check if the data has a multimodal distribution by analyzing its histogram.

        This method detects multimodality by:
        1. Creating a histogram of the data with num_bins bins
        2. Finding peaks in the histogram
        3. Checking if there are multiple significant peaks separated by valleys

        Args:
            x: Input tensor of shape (batch_size, features)
            stats: Dictionary of pre-computed statistics from _calculate_statistics
            num_bins: Number of bins to use for histogram analysis. Default is 100.

        Returns:
            Boolean tensor indicating if multimodality was detected

        Note:
            Multimodal distributions have multiple peaks, indicating that the data
            might come from a mixture of different distributions.
        """
        # Use a histogram approach to detect multiple modes
        # Flatten the input
        x_flat = tf.reshape(x, [-1])

        # `histogram_fixed_width` requires value_range[0] < value_range[1]. A
        # constant batch -- every single-row prediction, and any feature with no
        # spread -- makes min == max, and a fixed epsilon vanishes into float32
        # rounding at large magnitudes (50000 + 1e-6 == 50000). Widen the upper
        # bound by an epsilon scaled to the magnitude so the range is always
        # representable.
        lower = tf.cast(stats["min"], tf.float32)
        upper = tf.cast(stats["max"], tf.float32)
        min_span = tf.maximum(tf.abs(lower) * 1e-5, self.epsilon)
        upper = tf.maximum(upper, lower + min_span)

        hist = tf.histogram_fixed_width(
            x_flat,
            [lower, upper],
            nbins=num_bins,
        )

        # Smooth the histogram with a simple average filter
        # Avoid using loops for graph mode compatibility
        hist_float = tf.cast(hist, tf.float32)

        # Pad with zeros for simple window-based smoothing
        padded_hist = tf.pad(hist_float, [[1, 1]])

        # Create a simple 3-point average filter
        # For each position i, we compute the average of elements i-1, i, and i+1
        hist_padded_shifted_left = padded_hist[:-2]  # Elements i-1
        hist_padded_center = padded_hist[1:-1]  # Elements i
        hist_padded_shifted_right = padded_hist[2:]  # Elements i+1

        # Compute the average
        smoothed_hist = (
            hist_padded_shifted_left + hist_padded_center + hist_padded_shifted_right
        ) / 3.0

        # Find peaks (local maxima)
        # A peak is where the value is greater than its neighbors
        is_greater_left = tf.concat(
            [[True], smoothed_hist[1:] > smoothed_hist[:-1]],
            axis=0,
        )
        is_greater_right = tf.concat(
            [smoothed_hist[:-1] > smoothed_hist[1:], [True]],
            axis=0,
        )
        is_peak = tf.logical_and(is_greater_left, is_greater_right)

        # Count significant peaks (ignore small bumps)
        peak_threshold = tf.reduce_max(smoothed_hist) * 0.1  # 10% of max
        significant_peaks = tf.logical_and(is_peak, smoothed_hist > peak_threshold)
        num_peaks = tf.reduce_sum(tf.cast(significant_peaks, tf.int32))

        # Consider multimodal if at least two significant peaks
        return num_peaks >= 2

    def call(self, inputs, training=None) -> tf.Tensor:
        """Apply distribution-aware encoding to the inputs.

        This method:
        1. Detects the distribution type of input data (if auto_detect=True and in training mode)
        2. Applies appropriate transformations based on the detected distribution
        3. Optionally adds distribution embedding vectors
        4. Optionally adds periodic features (sin/cos transformations) for periodic data
        5. Optionally projects to the specified embedding dimension

        Args:
            inputs: Input tensor with shape (batch_size, ..., features)
            training: Boolean indicating if in training mode (True) or inference mode (False).
                      When True and auto_detect=True, distribution type is detected.
                      When False, uses previously detected distribution type.

        Returns:
            Transformed tensor with shape depending on configuration:
            - Base case: Same shape as input
            - With periodic features: (batch_size, ..., features*3)
            - With embedding_dim: (batch_size, ..., embedding_dim)
            - With distribution_embedding: Output has 8 additional dimensions

        Note:
            During inference (training=False), the layer uses the distribution type
            detected during training. This ensures consistent behavior between training
            and inference.
        """
        # Cast inputs to float32
        x = tf.cast(inputs, tf.float32)

        # Initialize flag for tracking if periodic features were added
        self._added_periodic_features = False

        # Handle distribution type
        if self.auto_detect:
            # Use the provided training flag or default to True if not specified
            is_training = training if training is not None else True

            # Use tf.cond for graph mode compatibility
            def detect_distribution_fn() -> tf.Tensor:
                # During training or first call, detect the distribution
                dist_idx = self._detect_distribution(x)

                # Store the detected distribution
                if hasattr(self, "detected_distribution"):
                    self.detected_distribution.assign(tf.reshape(dist_idx, [1]))

                # For graph mode compatibility, directly use the index
                # Ensure consistent type (int32)
                return tf.cast(dist_idx, tf.int32)

            def use_stored_distribution_fn() -> tf.Tensor:
                # During inference, use the stored distribution
                if hasattr(self, "detected_distribution"):
                    dist_idx = self.detected_distribution[0]
                else:
                    # Default to normal if no stored distribution
                    dist_idx = tf.constant(
                        self._valid_distributions.index("normal"),
                        dtype=tf.int32,
                    )
                return tf.cast(dist_idx, tf.int32)  # Ensure consistent type

            # Use the appropriate function based on training
            is_first_call = (
                not hasattr(self, "_is_initialized") or not self._is_initialized
            )
            should_detect = tf.logical_and(
                tf.cast(is_training, tf.bool),
                tf.logical_or(tf.cast(is_first_call, tf.bool), tf.constant(False)),
            )

            dist_idx = tf.cond(
                should_detect,
                detect_distribution_fn,
                use_stored_distribution_fn,
            )

            # Mark as initialized
            if is_first_call:
                self._is_initialized = True

            # Get distribution type from index - ensure index is within valid range
            safe_idx = tf.clip_by_value(dist_idx, 0, len(self._valid_distributions) - 1)
            dist_type = tf.gather(self._valid_distributions, safe_idx)
        else:
            # Use the specified distribution type
            dist_type = self.distribution_type
            # Get the index for consistency - ensure it's int32
            dist_idx = tf.constant(
                self._valid_distributions.index(dist_type),
                dtype=tf.int32,
            )

        # Store the current distribution type for use in compute_output_shape
        self._current_dist_type = dist_type
        self._current_dist_idx = dist_idx

        # Apply general distribution transformation
        transformed = self.distribution_transform(x, training=training)

        # Handle periodic features separately using a consistent approach for gradient flow
        # First, apply transformation to all inputs
        transformed = tf.identity(transformed)  # Ensure fresh tensor for gradient flow

        # Now conditionally add periodic features if needed
        if self.detect_periodicity:
            # Check if this distribution is periodic
            is_periodic = tf.equal(dist_type, "periodic")

            if tf.executing_eagerly():
                # In eager mode, we can use Python control flow
                if is_periodic:
                    # Add sin and cos features
                    self._added_periodic_features = True
                    mean = tf.reduce_mean(transformed)
                    std = tf.math.reduce_std(transformed) + self.epsilon
                    normalized = (
                        (transformed - mean)
                        / std
                        * tf.constant(3.14159, dtype=tf.float32)
                    )

                    sin_feature = tf.sin(self.frequency * normalized + self.phase)
                    cos_feature = tf.cos(self.frequency * normalized + self.phase)

                    # Concat along feature dimension
                    transformed = tf.concat(
                        [transformed, sin_feature, cos_feature],
                        axis=-1,
                    )
            else:
                # In graph mode, use tf.cond with carefully managed shapes
                def add_periodic_features() -> tf.Tensor:
                    mean = tf.reduce_mean(transformed)
                    std = tf.math.reduce_std(transformed) + self.epsilon
                    normalized = (
                        (transformed - mean)
                        / std
                        * tf.constant(3.14159, dtype=tf.float32)
                    )

                    sin_feature = tf.sin(self.frequency * normalized + self.phase)
                    cos_feature = tf.cos(self.frequency * normalized + self.phase)

                    # Set flag (has no effect in graph mode, just for documentation)
                    self._added_periodic_features = True

                    # Concat with the transformed features
                    return tf.concat([transformed, sin_feature, cos_feature], axis=-1)

                def keep_original() -> tf.Tensor:
                    return transformed

                # Use tf.cond to conditionally add periodic features
                transformed = tf.cond(is_periodic, add_periodic_features, keep_original)

        # Apply projection if provided
        if self.projection is not None:
            transformed = self.projection(transformed)

        # Add distribution embedding if needed
        if self.add_distribution_embedding and hasattr(self, "distribution_embedding"):
            # Get the distribution embedding for the current type
            embedding = tf.gather(self.distribution_embedding, dist_idx)

            # Tile embedding to match batch dimension
            batch_size = tf.shape(transformed)[0]
            tiled_embedding = tf.tile(tf.expand_dims(embedding, 0), [batch_size, 1])

            # Concatenate with transformed features
            transformed = tf.concat([transformed, tiled_embedding], axis=-1)

        return transformed

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute the output shape of the layer based on input shape and layer configuration.

        Args:
            input_shape: Shape tuple (tuple of integers) or TensorShape

        Returns:
            Output shape tuple or TensorShape

        Notes:
            The output shape depends on several factors:
            1. If detect_periodicity=True and periodicity is detected/forced:
               - Feature dimension is multiplied by 3 (original + sin + cos features)
            2. If embedding_dim is specified:
               - Output feature dimension will be embedding_dim
            3. If add_distribution_embedding=True:
               - 8 dimensions are added for the distribution embedding

            These transformations are applied in sequence, if applicable.
        """
        # For better graph compatibility, handle both TensorShape and tuple/list inputs
        input_shape = tf.TensorShape(input_shape)
        output_shape = input_shape

        # If we're going to add periodic features, output shape will change
        if self.detect_periodicity and self._added_periodic_features:
            # Expand the last dimension by 3x for the original + sin + cos features
            # Ensure we create a new shape tuple with the modified last dimension
            output_shape = output_shape[:-1].concatenate([output_shape[-1] * 3])

        # For projection, the output shape changes to the embedding dim
        if self.embedding_dim is not None:
            # Replace the last dimension with embedding_dim
            output_shape = output_shape[:-1].concatenate([self.embedding_dim])

        # If adding distribution embedding, output shape increases
        if self.add_distribution_embedding:
            # Add 8 to the last dimension for the embedding
            output_shape = output_shape[:-1].concatenate([output_shape[-1] + 8])

        return output_shape

    def get_config(self) -> dict:
        """Get the layer configuration for serialization.

        This method enables serialization and deserialization of the layer via
        `keras.saving.save_model()` and `keras.saving.load_model()`.

        Returns:
            Configuration dictionary containing all parameters needed to reconstruct the layer.

        Note:
            When saving a model containing a DistributionAwareEncoder layer, use the
            `get_custom_objects()` function to provide the necessary custom objects dictionary:

            ```python
            model.save("my_model.keras")
            custom_objects = get_custom_objects()
            loaded_model = keras.saving.load_model(
                "my_model", custom_objects=custom_objects
            )
            ```
        """
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "epsilon": self.epsilon,
                "detect_periodicity": self.detect_periodicity,
                "handle_sparsity": self.handle_sparsity,
                "auto_detect": self.auto_detect,
                "distribution_type": self.distribution_type,
                "transform_type": self.transform_type,
                "add_distribution_embedding": self.add_distribution_embedding,
                # Include legacy parameters for backward compatibility
                "num_bins": self.num_bins,
                "adaptive_binning": self.adaptive_binning,
                "mixture_components": self.mixture_components,
                "prefered_distribution": self.prefered_distribution,
            },
        )
        return config


# Note: This implementation replaces the previous version that relied on TensorFlow Probability.
# If you need the specific probability distributions from TensorFlow Probability, you may need to
# add those dependencies separately.


def get_custom_objects() -> dict:
    """Return custom objects dictionary needed for model loading.

    This function returns a dictionary of custom objects that need to be provided
    when loading a model that contains DistributionAwareEncoder layers.

    Returns:
        Dictionary mapping class names to class objects for all custom classes
        used by the DistributionAwareEncoder.

    Example:
        ```python
        from kdp.layers import DistributionAwareEncoder, get_custom_objects

        # Save the model
        model.save("my_model.keras")

        # Load the model
        custom_objects = get_custom_objects()
        loaded_model = keras.saving.load_model("my_model", custom_objects=custom_objects)
        ```
    """
    return {
        "DistributionAwareEncoder": DistributionAwareEncoder,
        "DistributionTransformLayer": DistributionTransformLayer,
    }
