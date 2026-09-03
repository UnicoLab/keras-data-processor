"""This module implements a DistributionTransformLayer that applies various transformations
to make data more normally distributed or to handle specific distribution types better.
It's particularly useful for preprocessing data before anomaly detection or other statistical analyses.
"""

import keras
from typing import Any
from loguru import logger
from keras import ops, KerasTensor
import tensorflow as tf


@keras.saving.register_keras_serializable(package="kdp.layers")
class DistributionTransformLayer(keras.layers.Layer):
    """Layer for transforming data distributions to improve anomaly detection.

    This layer applies various transformations to make data more normally distributed
    or to handle specific distribution types better. Supported transformations include
    log, square root, Box-Cox, Yeo-Johnson, arcsinh, cube-root, logit, quantile,
    robust-scale, and min-max.

    When transform_type is set to 'auto', the layer automatically selects the most
    appropriate transformation based on the data characteristics during training.

    Args:
        transform_type: Type of transformation to apply. Options are 'none', 'log', 'sqrt',
            'box-cox', 'yeo-johnson', 'arcsinh', 'cube-root', 'logit', 'quantile',
            'robust-scale', 'min-max', or 'auto'. Default is 'none'.
        lambda_param: Parameter for parameterized transformations like Box-Cox and Yeo-Johnson.
            Default is 0.0.
        epsilon: Small value added to prevent numerical issues like log(0). Default is 1e-10.
        min_value: Low end of the range 'min-max' scales onto. Default is 0.0.
        max_value: High end of the range 'min-max' scales onto. Default is 1.0.
        clip_values: Whether 'min-max' clips its result into
            [min_value, max_value]. Default is True.
        auto_candidates: list of transformation types to consider when transform_type is 'auto'.
            If None, all available transformations will be considered. Default is None.
        name: Optional name for the layer.

    Input shape:
        N-D tensor with shape: (batch_size, ..., features)

    Output shape:
        Same shape as input: (batch_size, ..., features)

    Example:
        ```python
        import keras
        import numpy as np
        from kmr.layers import DistributionTransformLayer

        # Create sample input data with skewed distribution
        x = keras.random.exponential((32, 10))  # 32 samples, 10 features

        # Apply log transformation
        log_transform = DistributionTransformLayer(transform_type="log")
        y = log_transform(x)
        print("Transformed output shape:", y.shape)  # (32, 10)

        # Apply Box-Cox transformation with lambda=0.5
        box_cox = DistributionTransformLayer(transform_type="box-cox", lambda_param=0.5)
        z = box_cox(x)

        # Apply arcsinh transformation (handles both positive and negative values)
        arcsinh_transform = DistributionTransformLayer(transform_type="arcsinh")
        a = arcsinh_transform(x)

        # Apply min-max scaling to range [0, 1]
        min_max = DistributionTransformLayer(
            transform_type="min-max", min_value=0.0, max_value=1.0
        )
        b = min_max(x)

        # Use automatic transformation selection
        auto_transform = DistributionTransformLayer(transform_type="auto")
        c = auto_transform(x)  # Will select the best transformation during training
        ```
    """

    def __init__(
        self,
        transform_type: str = "none",
        lambda_param: float = 0.0,
        epsilon: float = 1e-10,
        min_value: float = 0.0,
        max_value: float = 1.0,
        clip_values: bool = True,
        auto_candidates: list[str] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Call parent's __init__ first
        """Initialize the DistributionTransformLayer.

        See the class docstring for the accepted arguments and what
        each one controls.
        """
        super().__init__(name=name, **kwargs)

        # Set private attributes
        self._transform_type = transform_type
        self._lambda_param = lambda_param
        self._epsilon = epsilon
        self._min_value = min_value
        self._max_value = max_value
        self._clip_values = clip_values
        self._auto_candidates = auto_candidates

        # Set public attributes
        self.transform_type = self._transform_type
        self.lambda_param = self._lambda_param
        self.epsilon = self._epsilon
        self.min_value = self._min_value
        self.max_value = self._max_value
        self.clip_values = self._clip_values
        self.auto_candidates = self._auto_candidates

        # Define valid transformations
        self._valid_transforms = [
            "none",
            "log",
            "sqrt",
            "box-cox",
            "yeo-johnson",
            "arcsinh",
            "cube-root",
            "logit",
            "quantile",
            "robust-scale",
            "min-max",
            "auto",
        ]

        # Set default auto candidates if not provided
        if self.auto_candidates is None and self.transform_type == "auto":
            # Exclude 'none' and 'auto' from candidates
            self.auto_candidates = [
                t for t in self._valid_transforms if t not in ["none", "auto"]
            ]

        # Validate parameters
        self._validate_params()

        # Initialize auto-mode variables
        self._selected_transform = None
        self._is_initialized = False

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if self.transform_type not in self._valid_transforms:
            raise ValueError(
                f"transform_type must be one of {self._valid_transforms}, "
                f"got {self.transform_type}",
            )

        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")

        if self.min_value >= self.max_value:
            raise ValueError(
                f"min_value must be less than max_value, got min_value={self.min_value}, "
                f"max_value={self.max_value}",
            )

        if self.transform_type == "auto" and self.auto_candidates:
            for candidate in self.auto_candidates:
                if candidate not in self._valid_transforms or candidate in [
                    "auto",
                    "none",
                ]:
                    raise ValueError(
                        f"Invalid transformation candidate: {candidate}. "
                        f"Candidates must be valid transformations excluding 'auto' and 'none'.",
                    )

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Builds the layer with the given input shape.

        Args:
            input_shape: tuple of integers defining the input shape.
        """
        # For auto mode, we need to store the selected transformation
        if self.transform_type == "auto":
            # Create a variable to store the selected transformation index
            self._selected_transform_idx = self.add_weight(
                name="selected_transform_idx",
                shape=(1,),
                dtype="int32",
                trainable=False,
                initializer="zeros",
            )

            # Create a variable to store the selected lambda parameter
            self._selected_lambda = self.add_weight(
                name="selected_lambda",
                shape=(1,),
                dtype="float32",
                trainable=False,
                initializer="zeros",
            )

        logger.debug(
            f"DistributionTransformLayer built with transform_type={self.transform_type}, "
            f"lambda_param={self.lambda_param}",
        )
        super().build(input_shape)

    def _percentile(self, x_sorted: tf.Tensor, q: float) -> tf.Tensor:
        """Compute a percentile of a pre-sorted tensor along its first axis.

        Uses linear interpolation between the two neighbouring ranks, matching
        NumPy's default ``percentile`` behaviour. The computation stays free of
        Python-level branching so that it also works in graph mode, where the
        batch size is a tensor rather than an ``int``.

        Args:
            x_sorted: Tensor already sorted along axis 0.
            q: Percentile to compute, expressed as a fraction in [0, 1].

        Returns:
            Tensor of per-feature percentiles, with the batch axis removed.
        """
        n = tf.shape(x_sorted)[0]
        position = tf.cast(q, tf.float32) * tf.cast(n - 1, tf.float32)
        lower_idx = tf.cast(tf.floor(position), tf.int32)
        upper_idx = tf.minimum(lower_idx + 1, n - 1)
        fraction = position - tf.floor(position)

        lower = tf.gather(x_sorted, lower_idx, axis=0)
        upper = tf.gather(x_sorted, upper_idx, axis=0)
        return lower + (upper - lower) * tf.cast(fraction, x_sorted.dtype)

    def _compute_robust_statistics(
        self,
        x: KerasTensor,
    ) -> tuple[KerasTensor, KerasTensor, KerasTensor, KerasTensor]:
        """Compute order statistics for the input tensor.

        Args:
            x: Input tensor

        Returns:
            tuple of (min, max, median, interquartile_range), each computed
            per feature along the batch axis.
        """
        # Compute min and max along each feature dimension
        x_min = ops.min(x, axis=0, keepdims=True)
        x_max = ops.max(x, axis=0, keepdims=True)

        # Median and quartiles are read off the sorted values.
        x_sorted = tf.sort(x, axis=0)
        median = self._percentile(x_sorted, 0.5)
        q1 = self._percentile(x_sorted, 0.25)
        q3 = self._percentile(x_sorted, 0.75)

        # Add small epsilon to IQR to avoid division by zero
        iqr = ops.maximum(q3 - q1, self.epsilon)

        return x_min, x_max, median, iqr

    def _compute_statistics(self, x) -> dict:
        """Compute basic statistics for a tensor in a graph-mode compatible way.

        Args:
            x: Input tensor

        Returns:
            Dictionary of statistics including skewness, kurtosis,
            has_negative, has_zeros, and is_bounded_01
        """
        # Ensure we're working with float32
        x = tf.cast(x, tf.float32)

        # Get basic moments
        mean = tf.reduce_mean(x, axis=0)
        variance = tf.reduce_mean(tf.square(x - mean), axis=0)
        std_dev = tf.sqrt(variance + self.epsilon)

        # Compute skewness
        diff = x - mean
        third_moment = tf.reduce_mean(tf.pow(diff, 3), axis=0)
        skewness = third_moment / tf.pow(std_dev, 3)

        # Compute kurtosis
        fourth_moment = tf.reduce_mean(tf.pow(diff, 4), axis=0)
        kurtosis = fourth_moment / tf.pow(variance + self.epsilon, 2)

        # Check for negative values - as tensor
        min_val = tf.reduce_min(x)
        has_negative = min_val < 0

        # Check for zeros - as tensor
        # Approximate check using small epsilon
        has_zeros = tf.reduce_any(tf.abs(x) < self.epsilon)

        # Check if values are bounded between 0 and 1
        is_bounded_01 = tf.logical_and(min_val >= 0, tf.reduce_max(x) <= 1)

        # Return all statistics as a dictionary
        return {
            "skewness": skewness,
            "kurtosis": kurtosis,
            "has_negative": has_negative,
            "has_zeros": has_zeros,
            "is_bounded_01": is_bounded_01,
        }

    def _select_best_transformation(self, x: KerasTensor) -> tuple[str, float]:
        """Select the best transformation based on data characteristics."""
        # Compute basic statistics
        stats = self._compute_statistics(x)

        # Extract key metrics
        avg_skewness = tf.reduce_mean(tf.abs(stats["skewness"]))
        avg_kurtosis = tf.reduce_mean(stats["kurtosis"])
        is_bounded_01 = stats["is_bounded_01"]
        has_negative = stats["has_negative"]
        has_zeros = stats["has_zeros"]

        # Initialize with default values
        transform_type = self.transform_type
        lambda_param = self.lambda_param

        # If we're not selecting automatically, use the specified transform
        if transform_type != "auto":
            return transform_type, lambda_param

        # Define candidates list if not provided
        candidates = self.auto_candidates
        if not candidates:
            # Start with empty list
            candidates = []

            # Use tf.cond for all conditional logic to make graph-compatible
            # For bounded data in (0, 1)
            candidates = tf.cond(
                is_bounded_01,
                lambda: candidates + ["logit", "arcsinh", "min-max", "quantile"],
                lambda: candidates,
            )

            # For positive data with zeros
            candidates = tf.cond(
                tf.logical_and(tf.logical_not(has_negative), has_zeros),
                lambda: candidates + ["sqrt", "cube-root", "arcsinh"],
                lambda: candidates,
            )

            # For strictly positive data (no zeros)
            candidates = tf.cond(
                tf.logical_and(tf.logical_not(has_negative), tf.logical_not(has_zeros)),
                lambda: candidates + ["log", "sqrt", "box-cox", "arcsinh", "cube-root"],
                lambda: candidates,
            )

            # For mixed positive and negative data
            candidates = tf.cond(
                has_negative,
                lambda: candidates
                + ["yeo-johnson", "arcsinh", "cube-root", "robust-scale", "quantile"],
                lambda: candidates,
            )

            # Add general transformations that work for most data
            candidates = candidates + [
                "arcsinh",
                "yeo-johnson",
                "robust-scale",
                "quantile",
            ]

            # Remove duplicates while preserving order (for Python list only)
            if tf.executing_eagerly():
                candidates = list(dict.fromkeys(candidates))

        # Simple heuristic for transformation selection based on skewness and kurtosis
        abs_skewness = tf.abs(avg_skewness)
        abs_kurtosis = tf.abs(avg_kurtosis)

        # For graph mode compatibility, use a sequence of tf.cond operations
        # Initial default transformation
        transform_type = "none"

        # For bounded data in (0, 1)
        transform_type = tf.cond(
            is_bounded_01,
            lambda: tf.cond(
                abs_skewness > 0.5,
                lambda: tf.constant("logit"),
                lambda: tf.constant("min-max"),
            ),
            lambda: transform_type,
        )

        # For positive data
        transform_type = tf.cond(
            tf.logical_not(has_negative),
            lambda: tf.cond(
                abs_skewness > 1.0,
                lambda: tf.cond(
                    has_zeros,
                    lambda: tf.constant("arcsinh"),
                    lambda: tf.constant("log"),
                ),
                lambda: tf.cond(
                    abs_skewness > 0.5,
                    lambda: tf.constant("sqrt"),
                    lambda: tf.constant("arcsinh"),
                ),
            ),
            lambda: transform_type,
        )

        # For mixed positive and negative data
        transform_type = tf.cond(
            has_negative,
            lambda: tf.cond(
                tf.logical_or(abs_skewness > 1.0, abs_kurtosis > 3.0),
                lambda: tf.constant("yeo-johnson"),
                lambda: tf.cond(
                    abs_skewness > 0.5,
                    lambda: tf.constant("arcsinh"),
                    lambda: tf.constant("cube-root"),
                ),
            ),
            lambda: transform_type,
        )

        # Set lambda parameter based on transformation type and skewness direction
        # Use tf.cond to select lambda based on skewness
        lambda_param = tf.cond(
            tf.logical_or(
                tf.equal(transform_type, "box-cox"),
                tf.equal(transform_type, "yeo-johnson"),
            ),
            lambda: tf.cond(
                avg_skewness > 0,
                lambda: tf.constant(0.0, dtype=tf.float32),
                lambda: tf.constant(2.0, dtype=tf.float32),
            ),
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

        return transform_type, lambda_param

    def _apply_transform(self, x) -> tf.Tensor:
        """Apply the selected transformation to the input tensor in a graph-compatible way.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor
        """

        # Use tf.case to select the transformation type
        # Define functions for each transformation type
        def apply_none() -> tf.Tensor:
            return x

        def apply_log() -> tf.Tensor:
            # Add epsilon to avoid log(0)
            return tf.math.log(x + self.epsilon)

        def apply_sqrt() -> tf.Tensor:
            # Ensure non-negative values
            return tf.sqrt(tf.maximum(x, 0.0) + self.epsilon)

        def apply_box_cox() -> tf.Tensor:
            # Box-Cox transformation: (x^lambda - 1)/lambda if lambda != 0, log(x) if lambda == 0
            # Ensure x is positive
            x_pos = tf.maximum(x, self.epsilon)

            return tf.cond(
                tf.abs(self.lambda_param) < self.epsilon,
                lambda: tf.math.log(x_pos),  # For lambda ≈ 0, use log transform
                lambda: (tf.pow(x_pos, self.lambda_param) - 1.0)
                / self.lambda_param,  # Standard Box-Cox formula
            )

        def apply_yeo_johnson() -> tf.Tensor:
            # Yeo-Johnson works for both positive and negative values
            lambda_p = self.lambda_param

            # Create masks for positive and negative values
            pos_mask = x >= 0
            neg_mask = x < 0

            # Handle positive values
            pos_values = tf.cond(
                tf.abs(lambda_p) < self.epsilon,
                lambda: tf.math.log(x + 1.0),  # For lambda ≈ 0 and x ≥ 0
                lambda: (tf.pow(x + 1.0, lambda_p) - 1.0)
                / lambda_p,  # For other lambda and x ≥ 0
            )

            # Handle negative values
            neg_values = tf.cond(
                tf.abs(lambda_p - 2.0) < self.epsilon,
                lambda: -tf.math.log(-x + 1.0),  # For lambda ≈ 2 and x < 0
                lambda: -(
                    (tf.pow(-x + 1.0, 2.0 - lambda_p) - 1.0) / (2.0 - lambda_p)
                ),  # For other lambda and x < 0
            )

            # Combine results
            return tf.where(
                pos_mask,
                pos_values,
                tf.where(neg_mask, neg_values, tf.zeros_like(x)),
            )

        def apply_arcsinh() -> tf.Tensor:
            # Inverse hyperbolic sine transformation
            # Works well for both positive and negative values with heavy tails
            return tf.math.log(x + tf.sqrt(tf.square(x) + 1.0))

        def apply_cube_root() -> tf.Tensor:
            # Cube root transformation
            # Works well for both positive and negative values
            pos_mask = x >= 0
            neg_mask = x < 0

            # Handle positive values
            pos_values = tf.pow(x + self.epsilon, 1.0 / 3.0)

            # Handle negative values
            neg_values = -tf.pow(-x + self.epsilon, 1.0 / 3.0)

            # Combine results
            return tf.where(
                pos_mask,
                pos_values,
                tf.where(neg_mask, neg_values, tf.zeros_like(x)),
            )

        def apply_logit() -> tf.Tensor:
            # Logit transformation: log(x / (1 - x))
            # Clip values to the valid range with a small epsilon
            x_clipped = tf.clip_by_value(x, self.epsilon, 1.0 - self.epsilon)
            return tf.math.log(x_clipped / (1.0 - x_clipped))

        def apply_min_max() -> tf.Tensor:
            # Scale each feature's own range onto [min_value, max_value].
            #
            # Those two used to be read as the *source* range instead --
            # `(x - min_value) / (max_value - min_value)` -- so with their
            # defaults of 0 and 1 the transform returned its input untouched,
            # and no setting could scale onto a range other than [0, 1].
            # `clip_values` chose between that and the data's own range, which
            # is not what "whether to clip values to the specified range" says
            # and left the output far outside the range it names.
            #
            # Per feature, like `robust-scale` and `quantile` below: a single
            # min and max across the whole tensor would let one wide column
            # flatten every other one.
            data_min = tf.reduce_min(x, axis=0, keepdims=True)
            data_max = tf.reduce_max(x, axis=0, keepdims=True)

            spread = data_max - data_min
            spread = tf.where(
                tf.abs(spread) < self.epsilon,
                tf.ones_like(spread),
                spread,
            )

            low = tf.cast(self.min_value, x.dtype)
            high = tf.cast(self.max_value, x.dtype)
            scaled = low + (x - data_min) / spread * (high - low)

            if self.clip_values:
                scaled = tf.clip_by_value(scaled, low, high)
            return scaled

        def apply_robust_scale() -> tf.Tensor:
            # Robust scaling: centre on the per-feature median and divide by the
            # per-feature interquartile range, so outliers do not dominate.
            _, _, median, iqr = self._compute_robust_statistics(x)

            # Avoid division by zero for constant features.
            iqr = tf.where(tf.abs(iqr) < self.epsilon, tf.ones_like(iqr), iqr)

            return (x - median) / iqr

        def apply_quantile() -> tf.Tensor:
            # Rank-based quantile transformation, computed per feature so that
            # features on different scales are not pooled into a single ranking.
            # Ranks are mapped to [-1, 1] rather than through an exact normal
            # inverse CDF: the order is preserved and the output stays bounded.
            ranks = tf.argsort(tf.argsort(x, axis=0), axis=0)
            n = tf.shape(x)[0]
            quantiles = (tf.cast(ranks, tf.float32) + 0.5) / tf.cast(n, tf.float32)
            return tf.cast(2.0 * quantiles - 1.0, x.dtype)  # Map [0,1] to [-1,1]

        # Create a dictionary of transform types to functions
        transform_functions = {
            "none": apply_none,
            "log": apply_log,
            "sqrt": apply_sqrt,
            "box-cox": apply_box_cox,
            "yeo-johnson": apply_yeo_johnson,
            "arcsinh": apply_arcsinh,
            "cube-root": apply_cube_root,
            "logit": apply_logit,
            "min-max": apply_min_max,
            "robust-scale": apply_robust_scale,
            "quantile": apply_quantile,
        }

        # When using in graph mode, we need a dynamic approach
        result = x  # Default result

        # In graph mode, we need to compare the transform_type string to each valid option
        if isinstance(self.transform_type, tf.Tensor):
            # Create a series of tf.cond operations to select the right function
            for transform_name, transform_func in transform_functions.items():
                result = tf.cond(
                    tf.equal(self.transform_type, transform_name),
                    transform_func,
                    lambda current_result=result: current_result,
                )
            return result
        else:
            # In eager mode, we can directly access the transform_type string
            try:
                if self.transform_type in transform_functions:
                    return transform_functions[self.transform_type]()
                else:
                    # This should never happen due to validation in __init__
                    raise ValueError(
                        f"Unknown transformation type: {self.transform_type}",
                    )
            except Exception as e:
                # Add more context to the error
                raise type(e)(
                    f"Error in {self.transform_type} transformation: {str(e)}",
                ) from e

    def call(self, inputs, training=None) -> tf.Tensor:
        """Apply the selected transformation to the inputs.

        Args:
            inputs: Input tensor
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode

        Returns:
            Transformed tensor with the same shape as input
        """
        # Ensure inputs are cast to float32
        x = tf.cast(inputs, dtype=tf.float32)

        # Handle auto transformation mode
        if self.transform_type == "auto":
            if training or not self._is_initialized:
                # During training or first call, select the best transformation
                best_transform, best_lambda = self._select_best_transformation(x)

                # For graph mode compatibility, we need to handle the index computation differently
                # Map the transform types to indices using a series of tf.cond operations
                transform_idx = tf.constant(
                    0,
                    dtype=tf.int32,
                )  # Default to first transform

                # For graph mode, we need to compare the best_transform with each valid transform
                # and accumulate the index
                for i, transform_type in enumerate(self._valid_transforms):
                    transform_idx = tf.cond(
                        tf.equal(best_transform, transform_type),
                        lambda i=i: tf.constant(i, dtype=tf.int32),
                        lambda previous=transform_idx: previous,
                    )

                # Store the selected transformation
                if hasattr(self, "_selected_transform_idx"):
                    # Use the variable's assign method directly
                    self._selected_transform_idx.assign(tf.reshape(transform_idx, [1]))
                    self._selected_lambda.assign(tf.reshape(best_lambda, [1]))

                # Set the transformation type and lambda for this forward pass
                temp_transform_type = self.transform_type
                temp_lambda_param = self.lambda_param

                self.transform_type = best_transform
                self.lambda_param = best_lambda

                # Apply the transformation
                result = self._apply_transform(x)

                # Restore original values
                self.transform_type = temp_transform_type
                self.lambda_param = temp_lambda_param

                # Mark as initialized
                self._is_initialized = True

                return result
            else:
                # During inference, use the stored transformation
                # Get the transformation index in a way that works in both eager and graph mode
                if tf.executing_eagerly():
                    # For eager mode, we can use numpy() safely
                    transform_idx = int(self._selected_transform_idx.numpy()[0])
                    lambda_param = float(self._selected_lambda.numpy()[0])
                    transform_type = self._valid_transforms[transform_idx]
                else:
                    # For graph mode, we need a different approach
                    # Get the index from the variable
                    transform_idx = self._selected_transform_idx[0]
                    lambda_param = self._selected_lambda[0]
                    # Map the index to the transform type using tf.gather
                    transform_type = tf.gather(
                        tf.constant(self._valid_transforms),
                        transform_idx,
                    )

                # Set the transformation type and lambda for this forward pass
                temp_transform_type = self.transform_type
                temp_lambda_param = self.lambda_param

                # Use the transform type from the mapping
                self.transform_type = transform_type
                self.lambda_param = lambda_param

                # Apply the transformation
                result = self._apply_transform(x)

                # Restore original values
                self.transform_type = temp_transform_type
                self.lambda_param = temp_lambda_param

                return result
        else:
            # For non-auto modes, just apply the transformation
            return self._apply_transform(x)

    def get_config(self) -> dict[str, Any]:
        """Get the layer configuration.

        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update(
            {
                "transform_type": self.transform_type,
                "lambda_param": float(self.lambda_param),
                "epsilon": float(self.epsilon),
                "min_value": float(self.min_value),
                "max_value": float(self.max_value),
                "clip_values": self.clip_values,
                "auto_candidates": self.auto_candidates,
            },
        )
        return config

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple (tuple of integers) or TF TensorShape

        Returns:
            Output shape (tuple of integers or TensorShape)
        """
        # For most transformations, output shape is the same as input shape
        return input_shape
