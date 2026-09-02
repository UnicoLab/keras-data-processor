import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="kdp.layers")
class TransformerBlock(keras.layers.Layer):
    """Class that implements a transformer block."""

    def __init__(
        self,
        dim_model: int = 32,
        num_heads: int = 3,
        ff_units: int = 16,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        """Initializes the transformer block.

        Args:
            dim_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            ff_units (int): Units in the feed-forward layer.
            dropout_rate (float): Dropout rate to apply.
            kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.d_model = dim_model
        self.num_heads = num_heads
        self.ff_units = ff_units
        self.dropout_rate = dropout_rate

        # Define layers
        self.multihead_attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim_model,
        )
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.add1 = keras.layers.Add()
        self.layer_norm1 = keras.layers.LayerNormalization()

        self.ff1 = keras.layers.Dense(ff_units, activation="relu")
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.ff2 = keras.layers.Dense(dim_model)
        self.add2 = keras.layers.Add()
        self.layer_norm2 = keras.layers.LayerNormalization()

    def build(self, input_shape: tuple) -> None:
        """Build the sub-layers so Keras does not mark the block falsely built.

        The sub-layers are created in `__init__`, so without this method Keras 3
        warns that the layer "does not have a `build()` method implemented and
        it looks like it has unbuilt state", and marks it built anyway. Building
        them here on the shape `call` actually sees keeps saving, loading and
        weight transfer working on the real structure.

        Args:
            input_shape: Shape of the input, 2D or 3D. A 2D input is treated as
                a single-step sequence, matching what `call` does to it.
        """
        shape = tuple(input_shape)
        if len(shape) == 2:
            shape = (shape[0], 1, shape[1])

        if shape[-1] is not None and shape[-1] != self.d_model:
            raise ValueError(
                f"TransformerBlock was built with dim_model={self.d_model} but "
                f"received inputs of width {shape[-1]}. The residual connection "
                "adds the block's output to its input, so the two must match.",
            )

        self.multihead_attention.build(query_shape=shape, value_shape=shape)
        self.dropout1.build(shape)
        self.add1.build([shape, shape])
        self.layer_norm1.build(shape)

        self.ff1.build(shape)
        ff_shape = (*shape[:-1], self.ff_units)
        self.dropout2.build(ff_shape)
        self.ff2.build(ff_shape)
        self.add2.build([shape, shape])
        self.layer_norm2.build(shape)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Defines the forward pass for the transformer block.

        Args:
            inputs (tf.Tensor): Input tensor for the block.

        Returns:
            tf.Tensor: Output tensor after processing.
        """
        # Reshape if needed
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=1)

        # Multi-head attention
        attention = self.multihead_attention(inputs, inputs)
        attention = self.dropout1(attention)
        attention = self.add1([inputs, attention])
        attention_norm = self.layer_norm1(attention)

        # Feed-forward layers
        ff = self.ff1(attention_norm)
        ff = self.dropout2(ff)
        ff = self.ff2(ff)
        ff = self.add2([attention_norm, ff])
        return self.layer_norm2(ff)

    def get_config(self) -> dict:
        """Return the constructor arguments needed to rebuild this block.

        Returns:
            dict: Serializable configuration for the block.
        """
        config = super().get_config()
        config.update(
            {
                "dim_model": self.d_model,
                "num_heads": self.num_heads,
                "ff_units": self.ff_units,
                "dropout_rate": self.dropout_rate,
            },
        )
        return config
