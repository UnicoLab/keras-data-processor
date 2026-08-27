"""Feature-wise Mixture of Experts implementation for Keras Data Processor.

This module implements a specialized routing mechanism that directs different
features to different "expert" networks based on their characteristics.
"""

import tensorflow as tf
from tensorflow import keras


@tf.keras.utils.register_keras_serializable(package="kdp.moe")
class StackFeaturesLayer(tf.keras.layers.Layer):
    """Layer to stack individual features along a new axis (dim 1) for use with Feature MoE."""

    def __init__(self, name="stack_features", trainable=True, dtype=None, **kwargs):
        """Initialize the layer.

        Args:
            name: Name of the layer
            trainable: Whether the layer is trainable
            dtype: Data type of the layer
            **kwargs: Additional keyword arguments passed to the parent class
        """
        super().__init__(name=name, trainable=trainable, dtype=dtype, **kwargs)

    def call(self, inputs) -> tf.Tensor:
        """Stack features along axis 1.

        Args:
            inputs: List of feature tensors of shape [batch_size, feature_dim]

        Returns:
            Stacked tensor of shape [batch_size, num_features, feature_dim]
        """
        return tf.stack(inputs, axis=1)

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute the output shape.

        Args:
            input_shape: List of input shapes

        Returns:
            Output shape
        """
        if not isinstance(input_shape, list):
            raise ValueError("Input must be a list of tensors")

        batch_size = input_shape[0][0]
        feature_dim = input_shape[0][-1]
        num_features = len(input_shape)

        return (batch_size, num_features, feature_dim)

    def get_config(self) -> dict:
        """Get layer configuration for serialization."""
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="kdp.moe")
class UnstackLayer(tf.keras.layers.Layer):
    """Layer to unstack features along an axis."""

    def __init__(
        self,
        axis=1,
        name="unstack_features",
        trainable=True,
        dtype=None,
        **kwargs,
    ):
        """Initialize the layer.

        Args:
            axis: Axis to unstack along
            name: Name of the layer
            trainable: Whether the layer is trainable
            dtype: Data type of the layer
            **kwargs: Additional keyword arguments passed to the parent class
        """
        super().__init__(name=name, trainable=trainable, dtype=dtype, **kwargs)
        self.axis = axis

    def call(self, inputs) -> list:
        """Unstack features along specified axis.

        Args:
            inputs: Tensor to unstack

        Returns:
            List of tensors unstacked along the specified axis
        """
        return tf.unstack(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute the output shape.

        Args:
            input_shape: Input shape

        Returns:
            List of output shapes
        """
        shapes = []
        for i in range(input_shape[self.axis]):
            shape = list(input_shape)
            del shape[self.axis]
            shapes.append(tuple(shape))
        return shapes

    def get_config(self) -> dict:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


@tf.keras.utils.register_keras_serializable(package="kdp.moe")
class ExpertBlock(keras.layers.Layer):
    """Expert network for processing a subset of features.

    Each expert specializes in handling certain types of features or patterns.
    """

    def __init__(
        self,
        expert_dim: int = 64,
        hidden_dims: list[int] = None,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_batch_norm: bool = True,
        name: str | None = None,
        trainable: bool = True,
        dtype=None,
        **kwargs,
    ):
        """Initialize an expert network.

        Args:
            expert_dim: The output dimension of the expert
            hidden_dims: List of hidden layer dimensions (if None, uses [expert_dim*2])
            activation: Activation function to use
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            name: Optional name for the layer
            trainable: Whether the layer is trainable
            dtype: Data type of the layer
            **kwargs: Additional keyword arguments passed to the parent class
        """
        super().__init__(name=name, trainable=trainable, dtype=dtype, **kwargs)
        self.expert_dim = expert_dim
        self.hidden_dims = hidden_dims or [expert_dim, expert_dim]
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Build the expert network
        self.hidden_layers = []

        for i, units in enumerate(self.hidden_dims):
            self.hidden_layers.append(
                keras.layers.Dense(units, activation=None, name=f"expert_dense_{i}"),
            )

            if self.use_batch_norm:
                self.hidden_layers.append(
                    keras.layers.BatchNormalization(name=f"expert_bn_{i}"),
                )

            self.hidden_layers.append(
                keras.layers.Activation(self.activation, name=f"expert_act_{i}"),
            )

            if self.dropout_rate > 0:
                self.hidden_layers.append(
                    keras.layers.Dropout(self.dropout_rate, name=f"expert_drop_{i}"),
                )

        # Output layer
        self.output_layer = keras.layers.Dense(
            self.expert_dim,
            activation=None,
            name="expert_output",
        )

    def call(self, inputs, training=None) -> tf.Tensor:
        """Forward pass through the expert network.

        Args:
            inputs: Input tensor
            training: Whether in training mode (affects dropout and batch norm)

        Returns:
            Expert output tensor
        """
        x = inputs

        for layer in self.hidden_layers:
            if isinstance(
                layer, keras.layers.Dropout | keras.layers.BatchNormalization
            ):
                x = layer(x, training=training)
            else:
                x = layer(x)

        return self.output_layer(x)

    def get_config(self) -> dict:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "expert_dim": self.expert_dim,
                "hidden_dims": self.hidden_dims,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
                "use_batch_norm": self.use_batch_norm,
            },
        )
        return config


@tf.keras.utils.register_keras_serializable(package="kdp.moe")
class FeatureMoE(keras.layers.Layer):
    """Feature-wise Mixture of Experts layer.

    Routes different features to different expert networks based on either:
    1. Learned routing (trained router network)
    2. Predefined assignments (manual specification)
    """

    def __init__(
        self,
        num_experts: int = 4,
        expert_dim: int = 64,
        expert_hidden_dims: list[int] = None,
        routing: str = "learned",
        sparsity: int = 2,
        routing_activation: str = "softmax",
        feature_names: list[str] | None = None,
        predefined_assignments: dict[str, int] | None = None,
        freeze_experts: bool = False,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = True,
        name: str | None = None,
        trainable: bool = True,
        dtype=None,
        **kwargs,
    ):
        """Initialize the Feature-wise MoE layer.

        Args:
            num_experts: Number of expert networks
            expert_dim: Output dimension of each expert
            expert_hidden_dims: Hidden dimensions for each expert
            routing: Routing mechanism - "learned" or "predefined"
            sparsity: Number of experts to use per feature (for sparse routing)
            routing_activation: Activation for routing weights ("softmax" or "sparsemax")
            feature_names: Names of input features (required for predefined routing)
            predefined_assignments: Mapping from feature name to expert index
            freeze_experts: Whether to freeze the expert weights during training
            dropout_rate: Dropout rate for the experts
            use_batch_norm: Whether to use batch normalization in experts
            name: Optional name for the layer
            trainable: Whether the layer is trainable
            dtype: Data type of the layer
            **kwargs: Additional keyword arguments passed to the parent class
        """
        super().__init__(name=name, trainable=trainable, dtype=dtype, **kwargs)
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.expert_hidden_dims = expert_hidden_dims
        self.routing = routing
        self.sparsity = min(sparsity, num_experts)
        self.routing_activation = routing_activation
        self.feature_names = feature_names
        self.predefined_assignments = predefined_assignments
        self.freeze_experts = freeze_experts
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Validate parameters
        if routing == "predefined" and (
            not feature_names or not predefined_assignments
        ):
            raise ValueError(
                "For predefined routing, feature_names and predefined_assignments must be provided",
            )

        # Initialize experts
        self.experts = [
            ExpertBlock(
                expert_dim=expert_dim,
                hidden_dims=expert_hidden_dims,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                name=f"expert_{i}",
            )
            for i in range(num_experts)
        ]

        # Set up routing mechanism
        if routing == "learned":
            # Router network maps feature representations to expert weights
            self.router = keras.layers.Dense(num_experts, use_bias=True, name="router")
        else:
            # Create a fixed assignment matrix for predefined routing
            self._create_assignment_matrix()

    def _create_assignment_matrix(self):
        """Create a fixed assignment matrix for predefined routing."""
        if not self.feature_names or not self.predefined_assignments:
            return

        # Create a mapping from feature index to expert index
        self.assignment_matrix = tf.zeros((len(self.feature_names), self.num_experts))

        for i, feature_name in enumerate(self.feature_names):
            if feature_name in self.predefined_assignments:
                expert_idx = self.predefined_assignments[feature_name]
                if isinstance(expert_idx, int):
                    # One expert per feature
                    self.assignment_matrix = tf.tensor_scatter_nd_update(
                        self.assignment_matrix,
                        [[i, expert_idx]],
                        [1.0],
                    )
                else:
                    # Multiple experts with weights
                    for expert_id, weight in expert_idx.items():
                        self.assignment_matrix = tf.tensor_scatter_nd_update(
                            self.assignment_matrix,
                            [[i, expert_id]],
                            [weight],
                        )

        # Convert to a constant tensor for efficiency
        self.assignment_matrix = tf.constant(self.assignment_matrix)

    def _compute_routing_weights(self, inputs, training=None):
        """Compute routing weights for each feature.

        Args:
            inputs: Input tensor of shape [batch_size, num_features, feature_dim]
            training: Whether in training mode

        Returns:
            Routing weights of shape [batch_size, num_features, num_experts]
        """
        if self.routing == "predefined":
            # Use fixed assignments; expand dims for broadcasting over the batch.
            return tf.expand_dims(self.assignment_matrix, 0)
        else:
            # Compute routing weights using the router network
            # Average the feature representations along the batch dimension
            # to get feature-level routing rather than instance-level
            feature_reprs = tf.reduce_mean(
                inputs,
                axis=0,
            )  # [num_features, feature_dim]

            # Get logits from router
            routing_logits = self.router(feature_reprs)  # [num_features, num_experts]

            # Apply activation
            if self.routing_activation == "softmax":
                weights = tf.nn.softmax(routing_logits, axis=-1)
            else:  # Implement sparse routing
                # Sort logits and keep only top-k
                top_logits, top_indices = tf.nn.top_k(
                    routing_logits,
                    k=self.sparsity,
                    sorted=True,
                )

                # Create a mask for the top-k logits
                batch_size = tf.shape(feature_reprs)[0]
                mask = tf.scatter_nd(
                    indices=tf.stack(
                        [
                            tf.repeat(tf.range(batch_size), self.sparsity),
                            tf.reshape(top_indices, [-1]),
                        ],
                        axis=1,
                    ),
                    updates=tf.ones_like(tf.reshape(top_logits, [-1])),
                    shape=tf.shape(routing_logits),
                )

                # Apply mask and softmax
                masked_logits = routing_logits * mask - 1e9 * (1.0 - mask)
                weights = tf.nn.softmax(masked_logits, axis=-1)

            # Expand dims for broadcasting
            return tf.expand_dims(weights, 0)  # [1, num_features, num_experts]

    def call(self, inputs, training=None) -> tf.Tensor:
        """Forward pass through the Feature-wise MoE.

        Args:
            inputs: Input tensor of shape [batch_size, num_features, feature_dim]
            training: Whether in training mode

        Returns:
            Output tensor of shape [batch_size, num_features, expert_dim]
        """
        # Get shapes - commenting out unused variables
        # batch_size = tf.shape(inputs)[0]
        # num_features = tf.shape(inputs)[1]
        # feature_dim = tf.shape(inputs)[2]

        # Compute routing weights
        routing_weights = self._compute_routing_weights(
            inputs,
            training,
        )  # [1, num_features, num_experts]

        # Apply each expert to all features
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = (
                expert(inputs, training=False)
                if self.freeze_experts
                else expert(inputs, training=training)
            )
            expert_outputs.append(expert_output)

        # Stack expert outputs along a new axis
        stacked_outputs = tf.stack(
            expert_outputs,
            axis=-2,
        )  # [batch_size, num_features, num_experts, expert_dim]

        # Weight expert outputs by routing weights
        routing_weights_expanded = tf.expand_dims(
            routing_weights,
            -1,
        )  # [1, num_features, num_experts, 1]
        weighted_outputs = (
            stacked_outputs * routing_weights_expanded
        )  # [batch_size, num_features, num_experts, expert_dim]

        # Sum over experts
        return tf.reduce_sum(
            weighted_outputs,
            axis=-2,
        )  # [batch_size, num_features, expert_dim]

    def get_expert_assignments(self) -> dict:
        """Get the current expert assignments for each feature.

        For predefined routing, this returns the predefined_assignments dictionary.
        For learned routing, this calculates the current assignments based on router weights.

        Returns:
            dict: Feature assignments to experts
        """
        if self.routing == "predefined":
            return self.predefined_assignments
        elif self.routing == "learned":
            # Extract feature to expert assignments from learned router
            # Get the router weights and determine dominant expert(s) for each feature
            # Commenting out unused variable
            # router_weights = self.router.kernel  # [feature_dim, num_experts]

            # For now, return empty dict - this is a placeholder for learned routing
            return {}
        else:
            return {}

    def get_config(self) -> dict:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "expert_dim": self.expert_dim,
                "expert_hidden_dims": self.expert_hidden_dims,
                "routing": self.routing,
                "sparsity": self.sparsity,
                "routing_activation": self.routing_activation,
                "freeze_experts": self.freeze_experts,
                "dropout_rate": self.dropout_rate,
                "use_batch_norm": self.use_batch_norm,
            },
        )

        # Only include feature_names and predefined_assignments if using predefined routing
        if self.routing == "predefined":
            config.update(
                {
                    "feature_names": self.feature_names,
                    "predefined_assignments": self.predefined_assignments,
                },
            )

        return config


# Utility function to add Feature-wise MoE to a model
def add_feature_moe_to_model(
    model: keras.Model,
    feature_inputs: dict[str, keras.layers.Layer],
    num_experts: int = 4,
    expert_dim: int = 64,
    expert_hidden_dims: list[int] = None,
    routing: str = "learned",
    sparsity: int = 2,
    predefined_assignments: dict[str, int] | None = None,
    use_residual: bool = True,
) -> keras.Model:
    """Add Feature-wise Mixture of Experts to an existing preprocessing model.

    Args:
        model: The existing preprocessing model
        feature_inputs: Dictionary mapping feature names to input tensors
        num_experts: Number of expert networks
        expert_dim: Output dimension of each expert
        expert_hidden_dims: Hidden dimensions for each expert
        routing: Routing mechanism - "learned" or "predefined"
        sparsity: Number of experts to use per feature (for sparse routing)
        predefined_assignments: Mapping from feature name to expert index
        use_residual: Whether to use residual connections

    Returns:
        Updated model with Feature-wise MoE
    """
    # Get feature names and representations
    feature_names = list(feature_inputs.keys())
    feature_outputs = [
        model.get_layer(f"preprocessed_{name}").output for name in feature_names
    ]

    # Stack feature representations
    stacked_features = StackFeaturesLayer()(feature_outputs)

    # Apply Feature-wise MoE
    moe = FeatureMoE(
        num_experts=num_experts,
        expert_dim=expert_dim,
        expert_hidden_dims=expert_hidden_dims,
        routing=routing,
        sparsity=sparsity,
        feature_names=feature_names,
        predefined_assignments=predefined_assignments,
        name="feature_moe",
    )

    moe_outputs = moe(stacked_features)

    # Unstack the outputs for each feature
    unstacked_outputs = UnstackLayer(axis=1)(moe_outputs)

    # Create new outputs with optional residual connections
    new_outputs = []
    for i, (feature_name, original_output) in enumerate(
        zip(feature_names, feature_outputs, strict=False),
    ):
        expert_output = unstacked_outputs[i]

        # Add residual connection if shapes match
        if use_residual and original_output.shape[-1] == expert_output.shape[-1]:
            combined = keras.layers.Add(name=f"{feature_name}_moe_residual")(
                [original_output, expert_output],
            )
        else:
            # Otherwise just use the expert output
            combined = keras.layers.Dense(
                expert_dim,
                name=f"{feature_name}_moe_projection",
            )(expert_output)

        new_outputs.append(combined)

    # Create a new model with updated outputs
    return keras.Model(
        inputs=model.inputs,
        outputs=new_outputs,
        name=f"{model.name}_with_moe",
    )
