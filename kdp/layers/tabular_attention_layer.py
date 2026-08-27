import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="kdp.layers")
class TabularAttention(tf.keras.layers.Layer):
    """Custom layer to apply inter-feature and inter-sample attention for tabular data.

    This layer implements a dual attention mechanism:
    1. Inter-feature attention: Captures dependencies between features for each sample
    2. Inter-sample attention: Captures dependencies between samples for each feature

    The layer uses MultiHeadAttention for both attention mechanisms and includes
    layer normalization, dropout, and a feed-forward network.
    """

    def __init__(
        self, num_heads: int, d_model: int, dropout_rate: float = 0.1, **kwargs
    ):
        """Initialize the TabularAttention layer.

        Args:
            num_heads (int): Number of attention heads
            d_model (int): Dimensionality of the attention model
            dropout_rate (float): Dropout rate for regularization
            **kwargs: Additional keyword arguments for the layer
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        # Attention layers
        self.feature_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model,
        )
        self.sample_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model,
        )

        # Feed-forward network
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(d_model, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ],
        )

        # Normalization and dropout
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.feature_layernorm = tf.keras.layers.LayerNormalization()
        self.feature_layernorm2 = tf.keras.layers.LayerNormalization()
        self.feature_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.feature_dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.sample_layernorm = tf.keras.layers.LayerNormalization()
        self.sample_layernorm2 = tf.keras.layers.LayerNormalization()
        self.sample_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.sample_dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.output_projection = tf.keras.layers.Dense(d_model)

    def build(self, input_shape: int) -> None:
        """Build the layer.

        Args:
            input_shape: Shape tuple (tuple of integers) or list of shape tuples
        """
        self.input_dim = input_shape[-1]
        self.input_projection = tf.keras.layers.Dense(self.d_model)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass for TabularAttention.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, num_samples, num_features)
            training (bool): Whether the layer is in training mode

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, num_samples, d_model)

        Raises:
            ValueError: If input tensor is not 3-dimensional
        """
        if len(inputs.shape) != 3:
            raise ValueError(
                "Input tensor must be 3-dimensional (batch_size, num_samples, num_features)"
            )

        # Project inputs to d_model dimension
        projected = self.input_projection(inputs)

        # Inter-feature attention: across columns (features)
        features = self.feature_attention(
            projected, projected, projected, training=training
        )
        features = self.feature_layernorm(
            projected + self.feature_dropout(features, training=training)
        )
        features_ffn = self.ffn(features)
        features = self.feature_layernorm2(
            features + self.feature_dropout2(features_ffn, training=training)
        )

        # Inter-sample attention: across rows (samples)
        samples = tf.transpose(
            features, perm=[0, 2, 1]
        )  # Transpose for sample attention
        samples = self.sample_attention(samples, samples, samples, training=training)
        samples = tf.transpose(samples, perm=[0, 2, 1])  # Transpose back
        samples = self.sample_layernorm(
            features + self.sample_dropout(samples, training=training)
        )
        samples_ffn = self.ffn(samples)
        outputs = self.sample_layernorm2(
            samples + self.sample_dropout2(samples_ffn, training=training)
        )

        return outputs

    def get_config(self) -> dict:
        """Returns the configuration of the layer.

        Returns:
            dict: Configuration dictionary
        """
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "d_model": self.d_model,
                "dropout_rate": self.dropout_rate,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "TabularAttention":
        """Creates a layer from its config.

        Args:
            config: Layer configuration dictionary

        Returns:
            TabularAttention: A new instance of the layer
        """
        return cls(**config)
