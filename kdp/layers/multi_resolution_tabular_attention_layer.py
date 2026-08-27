import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="kdp.layers")
class MultiResolutionTabularAttention(tf.keras.layers.Layer):
    """Multi-resolution attention layer for tabular data.

    This layer implements separate attention mechanisms for numerical and categorical features,
    along with cross-attention between them.

    Args:
        num_heads (int): Number of attention heads
        d_model (int): Dimension of the attention model for numerical features
        embedding_dim (int): Dimension for categorical feature embeddings
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.

    Call arguments:
        numerical_features: Tensor of shape `(batch_size, num_numerical, numerical_dim)`
        categorical_features: Tensor of shape `(batch_size, num_categorical, categorical_dim)`
        training: Boolean indicating whether in training mode

    Returns:
        tuple: (numerical_output, categorical_output)
            - numerical_output: Tensor of shape `(batch_size, num_numerical, d_model)`
            - categorical_output: Tensor of shape `(batch_size, num_categorical, d_model)`
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        embedding_dim: int,
        dropout_rate: float = 0.1,
        **kwargs,
    ) -> None:
        """Initialize the MultiResolutionTabularAttention layer.

        Args:
            num_heads (int): Number of attention heads
            d_model (int): Dimension of the attention model for numerical features
            embedding_dim (int): Dimension for categorical feature embeddings
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
            **kwargs: Additional layer arguments
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

        # Create projection layers during initialization
        self.numerical_projection = tf.keras.layers.Dense(d_model)
        self.categorical_projection = tf.keras.layers.Dense(embedding_dim)

        # Numerical attention
        self.numerical_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
        )
        self.numerical_ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(d_model * 2, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ],
        )
        self.numerical_layernorm1 = tf.keras.layers.LayerNormalization()
        self.numerical_layernorm2 = tf.keras.layers.LayerNormalization()
        self.numerical_dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.numerical_dropout2 = tf.keras.layers.Dropout(dropout_rate)

        # Categorical attention
        self.categorical_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout_rate,
        )
        self.categorical_ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(embedding_dim * 2, activation="relu"),
                tf.keras.layers.Dense(embedding_dim),
            ],
        )
        self.categorical_layernorm1 = tf.keras.layers.LayerNormalization()
        self.categorical_layernorm2 = tf.keras.layers.LayerNormalization()
        self.categorical_dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.categorical_dropout2 = tf.keras.layers.Dropout(dropout_rate)

        # Cross attention
        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
        )
        self.cross_layernorm = tf.keras.layers.LayerNormalization()
        self.cross_dropout = tf.keras.layers.Dropout(dropout_rate)

        # Final projections
        self.categorical_output_projection = tf.keras.layers.Dense(d_model)

    def call(
        self,
        numerical_features: tf.Tensor,
        categorical_features: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Process numerical and categorical features through multi-resolution attention.

        Args:
            numerical_features: Tensor of shape (batch_size, num_numerical, numerical_dim)
            categorical_features: Tensor of shape (batch_size, num_categorical, categorical_dim)
            training: Whether the layer is in training mode

        Returns:
            tuple[tf.Tensor, tf.Tensor]: A tuple containing:
                - numerical_output: Tensor of shape (batch_size, num_numerical, d_model)
                - categorical_output: Tensor of shape (batch_size, num_categorical, d_model)
        """
        # Use the pre-initialized projection layer
        numerical_projected = self.numerical_projection(numerical_features)
        # Now process with attention
        numerical_attn = self.numerical_attention(
            numerical_projected,
            numerical_projected,
            numerical_projected,
            training=training,
        )
        numerical_1 = self.numerical_layernorm1(
            numerical_projected
            + self.numerical_dropout1(numerical_attn, training=training),
        )
        numerical_ffn = self.numerical_ffn(numerical_1)
        numerical_2 = self.numerical_layernorm2(
            numerical_1 + self.numerical_dropout2(numerical_ffn, training=training),
        )

        # Process categorical features
        categorical_projected = self.categorical_projection(categorical_features)
        categorical_attn = self.categorical_attention(
            categorical_projected,
            categorical_projected,
            categorical_projected,
            training=training,
        )
        categorical_1 = self.categorical_layernorm1(
            categorical_projected
            + self.categorical_dropout1(categorical_attn, training=training),
        )
        categorical_ffn = self.categorical_ffn(categorical_1)
        categorical_2 = self.categorical_layernorm2(
            categorical_1
            + self.categorical_dropout2(categorical_ffn, training=training),
        )

        # Cross attention: numerical features attend to categorical features
        categorical_for_cross = self.categorical_output_projection(categorical_2)
        cross_attn = self.cross_attention(
            numerical_2,
            categorical_for_cross,
            categorical_for_cross,
            training=training,
        )
        numerical_output = self.cross_layernorm(
            numerical_2 + self.cross_dropout(cross_attn, training=training),
        )

        # Project categorical features to match numerical dimension
        categorical_output = self.categorical_output_projection(categorical_2)

        return numerical_output, categorical_output

    def get_config(self) -> dict:
        """Get the layer configuration.

        Returns:
            dict: Configuration dictionary containing the layer parameters
        """
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "d_model": self.d_model,
                "embedding_dim": self.embedding_dim,
                "dropout_rate": self.dropout_rate,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "MultiResolutionTabularAttention":
        """Create a layer from its config.

        Args:
            config: Configuration dictionary

        Returns:
            MultiResolutionTabularAttention: A new instance of the layer
        """
        return cls(**config)
