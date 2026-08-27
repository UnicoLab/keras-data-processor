import keras
import tensorflow as tf
from kdp.layers.gated_residual_network_layer import GatedResidualNetwork


@keras.saving.register_keras_serializable(package="kdp.layers")
class VariableSelection(keras.layers.Layer):
    """VariableSelection is a custom Keras layer that implements a variable selection mechanism.

    This layer applies a gated residual network to each feature independently and concatenates the results.
    It then applies another gated residual network to the concatenated tensor and uses a softmax layer to
    calculate the weights for each feature. Finally, it combines the weighted features using matrix multiplication.

    Args:
        nr_features (int): Positive integer, number of input features.
        units (int): Positive integer, dimensionality of the output space.
        dropout_rate (float): Float between 0 and 1. Fraction of the input units to drop.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: A tuple containing:
            - selected_features: Output tensor after feature selection
            - feature_weights: Weights assigned to each feature
    """

    def __init__(
        self,
        nr_features: int,
        units: int,
        dropout_rate: float = 0.2,
        **kwargs: dict,
    ) -> None:
        """Initialize the VariableSelection layer.

        Args:
            nr_features (int): Number of input features.
            units (int): Dimensionality of the output space.
            dropout_rate (float, optional): Fraction of the input units to drop. Defaults to 0.2.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.nr_features = nr_features
        self.units = units
        self.dropout_rate = dropout_rate

        self.grns = []
        # Create a GRN for each feature independently
        for _ in range(nr_features):
            grn = GatedResidualNetwork(units=units, dropout_rate=dropout_rate)
            self.grns.append(grn)

        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(units=units, dropout_rate=dropout_rate)
        self.softmax = keras.layers.Dense(units=nr_features, activation="softmax")

    def call(
        self,
        inputs: list[tf.Tensor],
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Forward pass of the layer.

        Args:
            inputs (list[tf.Tensor]): List of input tensors.
            training (bool, optional): Whether in training mode. Defaults to False.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Tuple containing selected features and feature weights.
        """
        # Process concatenated features
        v = keras.layers.concatenate(inputs)
        v = self.grn_concat(v, training=training)
        feature_weights = self.softmax(v)
        feature_weights = tf.expand_dims(feature_weights, axis=-1)

        # Process each feature independently
        x = []
        for idx, feature_input in enumerate(inputs):
            x.append(self.grns[idx](feature_input, training=training))
        x = tf.stack(x, axis=1)

        # Apply feature selection weights
        selected_features = tf.squeeze(
            tf.matmul(feature_weights, x, transpose_a=True),
            axis=1,
        )
        return selected_features, feature_weights

    def get_config(self) -> dict:
        """Get layer configuration.

        Returns:
            dict: Layer configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "nr_features": self.nr_features,
                "units": self.units,
                "dropout_rate": self.dropout_rate,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "VariableSelection":
        """Create a layer instance from its config.

        Args:
            config (dict): Layer configuration dictionary.

        Returns:
            VariableSelection: A new instance of the layer.
        """
        return cls(**config)
