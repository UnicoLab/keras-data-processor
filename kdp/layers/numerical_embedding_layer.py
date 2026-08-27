import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="kdp.layers")
class NumericalEmbedding(tf.keras.layers.Layer):
    """Advanced numerical embedding layer for continuous features.

    This layer embeds each continuous numerical feature into a higher-dimensional space by
    combining two branches:

      1. Continuous Branch: Each feature is processed via a small MLP (using TimeDistributed layers).
      2. Discrete Branch: Each feature is discretized into bins using learnable min/max boundaries
         and then an embedding is looked up for its bin.

    A learnable gate (of shape (num_features, embedding_dim)) combines the two branch outputs
    per feature and per embedding dimension. Additionally, the continuous branch uses a residual
    connection and optional batch normalization to improve training stability.

    The layer supports inputs of shape (batch, num_features) for any number of features and returns
    outputs of shape (batch, num_features, embedding_dim).

    Args:
        embedding_dim (int): Output embedding dimension per feature.
        mlp_hidden_units (int): Hidden units for the continuous branch MLP.
        num_bins (int): Number of bins for discretization.
        init_min (float or list): Initial minimum values for discretization boundaries. If a scalar is
            provided, it is applied to all features.
        init_max (float or list): Initial maximum values for discretization boundaries.
        dropout_rate (float): Dropout rate applied to the continuous branch.
        use_batch_norm (bool): Whether to apply batch normalization to the continuous branch.

    """

    def __init__(
        self,
        embedding_dim: int = 8,
        mlp_hidden_units: int = 16,
        num_bins: int = 10,
        init_min: float | list[float] = -3.0,
        init_max: float | list[float] = 3.0,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        **kwargs,
    ):
        """Initialize the NumericalEmbedding layer.

        Args:
            embedding_dim: Dimension of the output embedding for each feature.
            mlp_hidden_units: Number of hidden units in the MLP.
            num_bins: Number of bins for discretization.
            init_min: Minimum value(s) for initialization. Can be a single float or list of floats.
            init_max: Maximum value(s) for initialization. Can be a single float or list of floats.
            dropout_rate: Dropout rate for regularization.
            use_batch_norm: Whether to use batch normalization.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.mlp_hidden_units = mlp_hidden_units
        self.num_bins = num_bins
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.init_min = init_min
        self.init_max = init_max

        if self.num_bins is None:
            raise ValueError(
                "num_bins must be provided to activate the discrete branch."
            )

    def build(self, input_shape):
        # input_shape: (batch, num_features)
        self.num_features = input_shape[-1]
        # Continuous branch: process each feature independently using TimeDistributed MLP.
        self.cont_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(self.mlp_hidden_units, activation="relu")
                ),
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(self.embedding_dim)
                ),
            ],
            name="cont_mlp",
        )
        self.dropout = (
            tf.keras.layers.Dropout(self.dropout_rate)
            if self.dropout_rate > 0
            else lambda x, training: x
        )
        if self.use_batch_norm:
            self.batch_norm = tf.keras.layers.TimeDistributed(
                tf.keras.layers.BatchNormalization(), name="cont_batch_norm"
            )
        # Residual projection to match embedding_dim.
        self.residual_proj = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.embedding_dim, activation=None),
            name="residual_proj",
        )
        # Discrete branch: Create one Embedding layer per feature.
        self.bin_embeddings = []
        for i in range(self.num_features):
            embed_layer = tf.keras.layers.Embedding(
                input_dim=self.num_bins,
                output_dim=self.embedding_dim,
                name=f"bin_embed_{i}",
            )
            self.bin_embeddings.append(embed_layer)
        # Learned bin boundaries for each feature, shape: (num_features,)
        init_min_tensor = tf.convert_to_tensor(self.init_min, dtype=tf.float32)
        init_max_tensor = tf.convert_to_tensor(self.init_max, dtype=tf.float32)
        if init_min_tensor.shape.ndims == 0:
            init_min_tensor = tf.fill([self.num_features], init_min_tensor)
        if init_max_tensor.shape.ndims == 0:
            init_max_tensor = tf.fill([self.num_features], init_max_tensor)

        if tf.executing_eagerly():
            init_min_value = init_min_tensor.numpy()
            init_max_value = init_max_tensor.numpy()
        else:
            # Fallback: if not executing eagerly, force conversion to list
            init_min_value = (
                init_min_tensor.numpy().tolist()
                if hasattr(init_min_tensor, "numpy")
                else self.init_min
            )
            init_max_value = (
                init_max_tensor.numpy().tolist()
                if hasattr(init_max_tensor, "numpy")
                else self.init_max
            )

        self.learned_min = self.add_weight(
            name="learned_min",
            shape=(self.num_features,),
            initializer=tf.constant_initializer(init_min_value),
            trainable=True,
        )
        self.learned_max = self.add_weight(
            name="learned_max",
            shape=(self.num_features,),
            initializer=tf.constant_initializer(init_max_value),
            trainable=True,
        )
        # Gate to combine continuous and discrete branches, shape: (num_features, embedding_dim)
        self.gate = self.add_weight(
            name="gate",
            shape=(self.num_features, self.embedding_dim),
            initializer="zeros",
            trainable=True,
        )

        # Explicitly build the sub-layers. Keras restores weights only for
        # sub-layers that are already built when a saved model is deserialized,
        # so leaving these to build lazily inside `call` breaks `load_model`
        # whenever this layer is nested inside another custom layer.
        batch = input_shape[0]
        branch_input_shape = (batch, self.num_features, 1)
        self.cont_mlp.build(branch_input_shape)
        self.residual_proj.build(branch_input_shape)
        if self.use_batch_norm:
            self.batch_norm.build((batch, self.num_features, self.embedding_dim))
        for embed_layer in self.bin_embeddings:
            embed_layer.build((batch,))

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Continuous branch.
        inputs_expanded = tf.expand_dims(inputs, axis=-1)  # (batch, num_features, 1)
        cont = self.cont_mlp(inputs_expanded)
        cont = self.dropout(cont, training=training)
        if self.use_batch_norm:
            cont = self.batch_norm(cont, training=training)
        # Residual connection.
        cont_res = self.residual_proj(inputs_expanded)
        cont = cont + cont_res  # (batch, num_features, embedding_dim)

        # Discrete branch.
        inputs_float = tf.cast(inputs, tf.float32)
        # Use learned min and max for scaling.
        scaled = (inputs_float - self.learned_min) / (
            self.learned_max - self.learned_min + 1e-6
        )
        # Compute bin indices.
        bin_indices = tf.floor(scaled * self.num_bins)
        bin_indices = tf.cast(bin_indices, tf.int32)
        bin_indices = tf.clip_by_value(bin_indices, 0, self.num_bins - 1)
        disc_embeddings = []
        for i in range(self.num_features):
            feat_bins = bin_indices[:, i]  # (batch,)
            feat_embed = self.bin_embeddings[i](
                feat_bins
            )  # i is a Python integer here.
            disc_embeddings.append(feat_embed)
        disc = tf.stack(disc_embeddings, axis=1)  # (batch, num_features, embedding_dim)

        # Combine branches via a per-feature, per-dimension gate.
        gate = tf.nn.sigmoid(self.gate)  # (num_features, embedding_dim)
        output = gate * cont + (1 - gate) * disc  # (batch, num_features, embedding_dim)
        # If only one feature is provided, squeeze the features axis.
        if self.num_features == 1:
            return tf.squeeze(output, axis=1)  # New shape: (batch, embedding_dim)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "mlp_hidden_units": self.mlp_hidden_units,
                "num_bins": self.num_bins,
                "init_min": self.init_min,
                "init_max": self.init_max,
                "dropout_rate": self.dropout_rate,
                "use_batch_norm": self.use_batch_norm,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
