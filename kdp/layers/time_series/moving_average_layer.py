import tensorflow as tf
from tensorflow.keras.layers import Layer


@tf.keras.utils.register_keras_serializable(package="kdp.layers")
class MovingAverageLayer(Layer):
    """Layer for computing moving averages of time series data.

    This layer computes simple moving averages over various periods.
    It's useful for smoothing and identifying longer-term trends.

    Args:
        periods: List of integers indicating the periods for the moving averages
        drop_na: Boolean indicating whether to drop rows with insufficient history
        pad_value: Value to use for padding when drop_na=False
        keep_original: Whether to include the original values in the output
    """

    def __init__(
        self,
        periods,
        drop_na=True,
        pad_value=0.0,
        keep_original=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.periods = periods if isinstance(periods, list) else [periods]
        self.drop_na = drop_na
        self.pad_value = pad_value
        self.keep_original = keep_original

        # Validate periods
        for period in self.periods:
            if period <= 0:
                raise ValueError(f"Period must be positive. Got {period}")

    def build(self, input_shape) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of the input tensor.
        """
        # Store the input shape for reshaping operations
        self.input_dims = len(input_shape)
        self.feature_size = input_shape[-1] if self.input_dims > 1 else 1

        super().build(input_shape)

    def _compute_ma(self, x, period: int) -> tf.Tensor:
        """Compute the moving average of a series over one period.

        Args:
            x: Input tensor of shape ``(time_steps, features)``. Time runs along
                axis 0, matching how the preprocessing pipeline feeds sorted
                time series through this layer.
            period: Window length.

        Returns:
            Tensor of shape ``(time_steps - period + 1, features)`` when
            ``drop_na`` is set, otherwise ``(time_steps, features)`` with the
            leading positions filled by expanding-window averages.
        """
        cumsum = tf.cumsum(x, axis=0)
        # Prepending a zero row turns every window sum into a single subtraction.
        cumsum_padded = tf.concat([tf.zeros_like(x[:1]), cumsum], axis=0)
        full_windows = (cumsum_padded[period:] - cumsum_padded[:-period]) / tf.cast(
            period,
            x.dtype,
        )

        if self.drop_na:
            return full_windows

        # Positions before the first full window use the average of everything
        # seen so far, so the output keeps the length of the input.
        head = cumsum[: period - 1]
        counts = tf.range(1, tf.shape(head)[0] + 1, dtype=x.dtype)
        expanding = head / tf.expand_dims(counts, axis=-1)
        return tf.concat([expanding, full_windows], axis=0)

    def call(self, inputs) -> tf.Tensor:
        """Apply the moving average computation.

        Args:
            inputs: Input tensor of shape ``(time_steps,)`` or
                ``(time_steps, features)``.

        Returns:
            Tensor with original values and/or moving averages depending on
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
            result_tensors.append(inputs)
        for period in self.periods:
            result_tensors.append(self._compute_ma(inputs, period))

        if len(result_tensors) > 1:
            # Longer windows drop more leading rows; align on the shortest so
            # every column refers to the same time steps. Taking the tail keeps
            # the rows where every requested window is defined.
            lengths = [tf.shape(t)[0] for t in result_tensors]
            min_length = tf.reduce_min(lengths)
            result_tensors = [t[tf.shape(t)[0] - min_length :] for t in result_tensors]

        result = tf.concat(result_tensors, axis=-1)

        # A 1-D series producing a single column is returned as 1-D, matching
        # the rank the caller passed in.
        if input_is_1d and not self.keep_original and len(self.periods) == 1:
            result = tf.reshape(result, [-1])

        return result

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

        if self.drop_na and time_steps is not None:
            time_steps = max(time_steps - (max(self.periods) - 1), 0)

        n_columns = n_features * (len(self.periods) + (1 if self.keep_original else 0))

        if input_is_1d and not self.keep_original and len(self.periods) == 1:
            return (time_steps,)
        return (time_steps, n_columns)

    def get_config(self) -> dict:
        config = {
            "periods": self.periods,
            "drop_na": self.drop_na,
            "pad_value": self.pad_value,
            "keep_original": self.keep_original,
        }
        base_config = super().get_config()
        return {**base_config, **config}
