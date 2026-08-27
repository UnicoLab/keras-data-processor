import tensorflow as tf
from kdp.layers.gated_linear_unit_layer import GatedLinearUnit


@tf.keras.utils.register_keras_serializable(package="kdp.layers")
class GatedResidualNetwork(tf.keras.layers.Layer):
    """GatedResidualNetwork is a custom Keras layer that implements a gated residual network.

    This layer applies a series of transformations to the input tensor and combines it with the original input
    using a residual connection. The transformations include a dense layer with ELU activation, a dense linear
    layer, a dropout layer, a gated linear unit layer, layer normalization, and a final dense layer.

    Args:
        units (int): Positive integer, dimensionality of the output space.
        dropout_rate (float): Float between 0 and 1. Fraction of the input units to drop.

    Returns:
        tf.Tensor: Output tensor of the GatedResidualNetwork layer.
    """

    def __init__(self, units: int, dropout_rate: float = 0.2, **kwargs: dict) -> None:
        """Initialize the GatedResidualNetwork layer.

        Args:
            units (int): Dimensionality of the output space.
            dropout_rate (float, optional): Fraction of the input units to drop. Defaults to 0.2.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.elu_dense = tf.keras.layers.Dense(units, activation="elu")
        self.linear_dense = tf.keras.layers.Dense(units)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units=units)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.project = tf.keras.layers.Dense(units)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs (tf.Tensor): Input tensor.
            training (bool, optional): Whether in training mode. Defaults to False.

        Returns:
            tf.Tensor: Output tensor after applying gated residual transformations.
        """
        # Cast inputs to float32 at the start
        inputs = tf.cast(inputs, tf.float32)

        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x, training=training)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x

    def get_config(self) -> dict:
        """Get layer configuration.

        Returns:
            dict: Layer configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "dropout_rate": self.dropout_rate,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "GatedResidualNetwork":
        """Create a layer instance from its config.

        Args:
            config (dict): Layer configuration dictionary.

        Returns:
            GatedResidualNetwork: A new instance of the layer.
        """
        return cls(**config)
