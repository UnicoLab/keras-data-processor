import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package="kdp.layers")
class PreserveDtypeLayer(keras.layers.Layer):
    """Custom Keras layer that preserves the original dtype of input tensors.

    This is useful for passthrough features where we want to maintain the original
    data type without any casting.
    """

    def __init__(self, target_dtype=None, **kwargs):
        """Initialize the layer.

        Args:
            target_dtype: Optional target dtype to cast to. If None, preserves original dtype.
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.target_dtype = target_dtype

    def call(self, inputs, **kwargs) -> tf.Tensor:
        """Process the input tensor, optionally casting to target_dtype.

        Args:
            inputs: Input tensor of any dtype
            **kwargs: Additional keyword arguments

        Returns:
            Tensor with preserved or target dtype
        """
        if self.target_dtype is not None:
            return tf.cast(inputs, self.target_dtype)
        return inputs

    def get_config(self) -> dict:
        """Return the config dictionary for serialization.

        Returns:
            A dictionary with the layer configuration
        """
        config = super().get_config()
        config.update({"target_dtype": self.target_dtype})
        return config

    @classmethod
    def from_config(cls, config) -> "PreserveDtypeLayer":
        """Create a new instance from the serialized configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            A new instance of the layer
        """
        return cls(**config)
