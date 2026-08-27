import keras
import tensorflow as tf
import numpy as np
from kdp.layers.numerical_embedding_layer import NumericalEmbedding


@keras.saving.register_keras_serializable(package="kdp.layers")
class GlobalNumericalEmbedding(keras.layers.Layer):
    """Global NumericalEmbedding processes concatenated numeric features.
    It applies an inner NumericalEmbedding over the flattened input and then
    performs global pooling (average or max) to produce a compact representation.
    """

    def __init__(
        self,
        global_embedding_dim: int = 8,
        global_mlp_hidden_units: int = 16,
        global_num_bins: int = 10,
        global_init_min: float | list[float] = -3.0,
        global_init_max: float | list[float] = 3.0,
        global_dropout_rate: float = 0.1,
        global_use_batch_norm: bool = True,
        global_pooling: str = "average",
        **kwargs,
    ):
        """Initialize the GlobalNumericalEmbedding layer.

        Args:
            global_embedding_dim: Dimension of the final global embedding.
            global_mlp_hidden_units: Number of hidden units in the global MLP.
            global_num_bins: Number of bins for discretization.
            global_init_min: Minimum value(s) for initialization. Can be a single float or list of floats.
            global_init_max: Maximum value(s) for initialization. Can be a single float or list of floats.
            global_dropout_rate: Dropout rate for regularization.
            global_use_batch_norm: Whether to use batch normalization.
            global_pooling: Pooling method to use ("average" or "max").
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.global_embedding_dim = global_embedding_dim
        self.global_mlp_hidden_units = global_mlp_hidden_units
        self.global_num_bins = global_num_bins

        # Ensure initializer parameters are Python scalars, lists, or numpy arrays.
        if not isinstance(global_init_min, list | tuple | np.ndarray):
            try:
                global_init_min = float(global_init_min)
            except Exception:
                raise ValueError(
                    "init_min must be a Python scalar, list, tuple or numpy array",
                )
        if not isinstance(global_init_max, list | tuple | np.ndarray):
            try:
                global_init_max = float(global_init_max)
            except Exception:
                raise ValueError(
                    "init_max must be a Python scalar, list, tuple or numpy array",
                )
        self.global_init_min = global_init_min
        self.global_init_max = global_init_max
        self.global_dropout_rate = global_dropout_rate
        self.global_use_batch_norm = global_use_batch_norm
        self.global_pooling = global_pooling

        # Use the existing advanced numerical embedding block
        self.inner_embedding = NumericalEmbedding(
            embedding_dim=self.global_embedding_dim,
            mlp_hidden_units=self.global_mlp_hidden_units,
            num_bins=self.global_num_bins,
            init_min=self.global_init_min,
            init_max=self.global_init_max,
            dropout_rate=self.global_dropout_rate,
            use_batch_norm=self.global_use_batch_norm,
            name="global_numeric_emebedding",
        )
        if self.global_pooling == "average":
            self.global_pooling_layer = keras.layers.GlobalAveragePooling1D(
                name="global_avg_pool",
            )
        elif self.global_pooling == "max":
            self.global_pooling_layer = keras.layers.GlobalMaxPooling1D(
                name="global_max_pool",
            )
        else:
            raise ValueError(f"Unsupported pooling method: {self.global_pooling}")

    def build(self, input_shape) -> None:
        """Explicitly build the sub-layers so saved weights can be restored.

        Keras only restores the weights of sub-layers that already exist and are
        built when a model is deserialized. Building them lazily inside ``call``
        leaves them unbuilt at load time, which makes ``load_model`` fail, so the
        inner embedding and pooling layers are built here instead.

        Args:
            input_shape: Shape of the inputs, ``(batch, ...)``.
        """
        flat_dim = input_shape[-1]
        if len(input_shape) > 2:
            flat_dim = 1
            for dim in input_shape[1:]:
                flat_dim *= dim

        self.inner_embedding.build((input_shape[0], flat_dim))
        self.global_pooling_layer.build(
            (input_shape[0], flat_dim, self.global_embedding_dim),
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Expects inputs with shape (batch, ...) and flattens them (except for the batch dim).
        Then, the inner embedding produces a 3D output (batch, num_features, embedding_dim),
        which is finally pooled to yield (batch, embedding_dim).
        """
        # If inputs have more than 2 dimensions, flatten them (except for batch dimension).
        if len(inputs.shape) > 2:
            inputs = tf.reshape(inputs, (tf.shape(inputs)[0], -1))
        # Pass through the inner advanced embedding.
        x_embedded = self.inner_embedding(inputs, training=training)
        # Global pooling over numeric features axis.
        return self.global_pooling_layer(x_embedded)

    def compute_output_shape(self, input_shape) -> tuple:
        # Regardless of the input shape, the output shape is (batch_size, embedding_dim)
        """Compute the output shape for a given input shape.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            The corresponding output shape.
        """
        return (input_shape[0], self.global_embedding_dim)

    def get_config(self) -> dict:
        """Return the configuration needed to re-create this layer.

        Returns:
            The layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "global_embedding_dim": self.global_embedding_dim,
                "global_mlp_hidden_units": self.global_mlp_hidden_units,
                "global_num_bins": self.global_num_bins,
                "global_init_min": self.global_init_min,
                "global_init_max": self.global_init_max,
                "global_dropout_rate": self.global_dropout_rate,
                "global_use_batch_norm": self.global_use_batch_norm,
                "global_pooling": self.global_pooling,
            },
        )
        return config

    @classmethod
    def from_config(cls, config) -> "GlobalNumericalEmbedding":
        """Re-create the layer from its configuration.

        Args:
            config: Configuration dictionary produced by `get_config`.

        Returns:
            The reconstructed layer.
        """
        return cls(**config)
