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
