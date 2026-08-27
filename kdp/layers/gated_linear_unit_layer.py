import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="kdp.layers")
class GatedLinearUnit(tf.keras.layers.Layer):
    """GatedLinearUnit is a custom Keras layer that implements a gated linear unit.

    This layer applies a dense linear transformation to the input tensor and multiplies the result with the output
    of a dense sigmoid transformation. The result is a tensor where the input data is filtered based on the learned
    weights and biases of the layer.

    Args:
        units (int): Positive integer, dimensionality of the output space.

    Returns:
        tf.Tensor: Output tensor of the GatedLinearUnit layer.
    """

    def __init__(self, units: int, **kwargs: dict) -> None:
        """Initialize the GatedLinearUnit layer.

        Args:
            units (int): Dimensionality of the output space.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.units = units
        self.linear = tf.keras.layers.Dense(units)
        self.sigmoid = tf.keras.layers.Dense(units, activation="sigmoid")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor after applying gated linear transformation.
        """
        return self.linear(inputs) * self.sigmoid(inputs)

    def get_config(self) -> dict:
        """Get layer configuration.

        Returns:
            dict: Layer configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "units": self.units,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "GatedLinearUnit":
        """Create a layer instance from its config.

        Args:
            config (dict): Layer configuration dictionary.

        Returns:
            GatedLinearUnit: A new instance of the layer.
        """
        return cls(**config)
