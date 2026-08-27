import tensorflow as tf
from tensorflow.keras.layers import Layer


@tf.keras.utils.register_keras_serializable(package="kdp.layers")
class RollingStatsLayer(Layer):
    """Layer for computing rolling statistics on time series data.

    This layer computes various statistics (mean, std, min, max, sum)
    over a rolling window of the specified size.

    Args:
        window_size: Size of the rolling window
        statistics: List of statistics to compute (supported: "mean", "std", "min", "max", "sum")
        window_stride: Step size for moving the window (default=1)
        drop_na: Boolean indicating whether to drop rows with insufficient history
        pad_value: Value to use for padding when drop_na=False
        keep_original: Whether to include the original values in the output
    """

    def __init__(
        self,
        window_size,
        statistics,
        window_stride=1,
        drop_na=True,
        pad_value=0.0,
        keep_original=False,
        **kwargs,
    ):
        """Initialize the RollingStatsLayer.

        See the class docstring for the accepted arguments and what
        each one controls.
        """
        super().__init__(**kwargs)
        self.window_size = window_size
        self.statistics = statistics if isinstance(statistics, list) else [statistics]
        self.window_stride = window_stride
        self.drop_na = drop_na
        self.pad_value = pad_value
        self.keep_original = keep_original

        # For backward compatibility - if stat_name is passed, use it
        self.stat_name = self.statistics[0] if len(self.statistics) > 0 else "mean"

        # Validate window_size
        if self.window_size <= 0:
            raise ValueError(f"Window size must be positive. Got {window_size}")

        # Validate statistics
        valid_stats = ["mean", "std", "min", "max", "sum"]
        for stat in self.statistics:
            if stat not in valid_stats:
                raise ValueError(f"Statistic must be one of {valid_stats}. Got {stat}")

    def build(self, input_shape) -> None:
        """Build the layer's weights for a given input shape.

        Args:
            input_shape: Shape of the input tensor.
        """
        super().build(input_shape)

    def call(self, inputs) -> tf.Tensor:
        """Apply the rolling statistic computation.

        Args:
            inputs: Input tensor of shape ``(time_steps,)`` or
                ``(time_steps, features)``. Time runs along axis 0.

        Returns:
            Tensor with original values and/or rolling statistics depending on
            configuration.
        """
        inputs = tf.convert_to_tensor(inputs)
        # Static rank: `tf.rank` returns a tensor, which cannot drive a Python
        # `if` once the layer is traced into a graph.
        input_is_1d = inputs.shape.rank == 1
        if input_is_1d:
            inputs = tf.reshape(inputs, (-1, 1))

        result_tensors = []
        if self.keep_original:
            result_tensors.append(self._aligned_originals(inputs))
        for stat in self.statistics:
            result_tensors.append(self._compute_statistic(inputs, stat))

        if len(result_tensors) > 1:
            # Align on the shortest so every column refers to the same windows.
            min_length = tf.reduce_min([tf.shape(t)[0] for t in result_tensors])
            result_tensors = [t[:min_length] for t in result_tensors]

        result = tf.concat(result_tensors, axis=-1)

        # A 1-D series producing a single column is returned as 1-D, matching
        # the rank the caller passed in.
        if input_is_1d and len(self.statistics) == 1 and not self.keep_original:
            result = tf.reshape(result, [-1])

        return result

    def _aligned_originals(self, x) -> tf.Tensor:
        """Select the original rows that line up with each computed window.

        Args:
            x: Input tensor of shape ``(time_steps, features)``.

        Returns:
            The original values trimmed (and strided) to match the statistics.
        """
        if not self.drop_na:
            return x
        # One original per window, taken at the window's last time step.
        windows = tf.signal.frame(
            x,
            frame_length=self.window_size,
            frame_step=self.window_stride,
            axis=0,
        )
        return windows[:, -1, :]

    def _compute_statistic(self, x, stat_name: str) -> tf.Tensor:
        """Compute one rolling statistic over the input.

        Args:
            x: Input tensor of shape ``(time_steps, features)``.
            stat_name: Name of the statistic to compute.

        Returns:
            Tensor of rolling statistics. Windows shorter than ``window_size``
            are dropped when ``drop_na`` is set, and filled with ``pad_value``
            otherwise.
        """
        # `tf.signal.frame` builds every window in one op and applies the stride
        # exactly once. It yields zero frames when the series is shorter than
        # the window, which is the correct "no full window" answer.
        windows = tf.signal.frame(
            x,
            frame_length=self.window_size,
            frame_step=self.window_stride,
            axis=0,
        )
        # windows: (num_windows, window_size, features) -> reduce over the window.
        stats = self._calculate_stat(windows, stat_name, axis=1)

        if self.drop_na:
            return stats

        padding = tf.fill(
            [self.window_size - 1, tf.shape(x)[1]],
            tf.cast(self.pad_value, x.dtype),
        )
        return tf.concat([padding, stats], axis=0)

    def _calculate_stat(self, window, stat_name: str, axis: int = 0) -> tf.Tensor:
        """Calculate the specified statistic on a window.

        Args:
            window: Tensor holding the window values.
            stat_name: Name of the statistic to compute.
            axis: Axis along which the window extends.

        Returns:
            Tensor with the computed statistic.

        Raises:
            ValueError: If ``stat_name`` is not a supported statistic.
        """
        reducers = {
            "mean": tf.reduce_mean,
            "std": tf.math.reduce_std,
            "min": tf.reduce_min,
            "max": tf.reduce_max,
            "sum": tf.reduce_sum,
        }
        if stat_name not in reducers:
            raise ValueError(f"Unknown statistic: {stat_name}")
        return reducers[stat_name](window, axis=axis)

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

        if time_steps is not None:
            n_windows = max(
                (time_steps - self.window_size) // self.window_stride + 1,
                0,
            )
            time_steps = n_windows if self.drop_na else self.window_size - 1 + n_windows
            if self.keep_original and not self.drop_na:
                time_steps = min(time_steps, input_shape[0])

        n_columns = n_features * (
            len(self.statistics) + (1 if self.keep_original else 0)
        )

        if input_is_1d and len(self.statistics) == 1 and not self.keep_original:
            return (time_steps,)
        return (time_steps, n_columns)

    def get_config(self) -> dict:
        """Return the configuration needed to re-create this layer.

        Returns:
            The layer configuration.
        """
        config = {
            "window_size": self.window_size,
            "statistics": self.statistics,
            "window_stride": self.window_stride,
            "drop_na": self.drop_na,
            "pad_value": self.pad_value,
            "keep_original": self.keep_original,
        }
        base_config = super().get_config()
        return {**base_config, **config}
