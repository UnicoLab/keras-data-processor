"""Feature-wise Mixture of Experts implementation for Keras Data Processor.

This module implements a specialized routing mechanism that directs different
features to different "expert" networks based on their characteristics.
"""

import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package="kdp.moe")
class PadFeatureLayer(keras.layers.Layer):
    """Right-pad a feature with zeros so every feature stacks to one width.

    `StackFeaturesLayer` needs `[batch, num_features, feature_dim]`, which means
    every feature must be the same width. Real feature sets are not: a
    normalised float is one column and a discretised one is ten. Padding keeps
    each feature whole and identifiable, and costs no parameters -- a feature
    already at the target width is returned untouched.
    """

    def __init__(self, width: int, **kwargs):
        """Initialize the layer.

        Args:
            width: The width every feature is padded up to.
            **kwargs: Passed to the parent layer.
        """
        super().__init__(**kwargs)
        self.width = int(width)

    def call(self, inputs) -> tf.Tensor:
        """Pad the last axis up to `width`.

        Args:
            inputs: A tensor shaped `[batch, feature_dim]`.

        Returns:
            The tensor padded on the right to `[batch, width]`.
        """
        missing = self.width - int(inputs.shape[-1])
        if missing <= 0:
            return inputs
        return tf.pad(inputs, [[0, 0], [0, missing]])

    def compute_output_shape(self, input_shape) -> tuple:
        """Report the padded shape.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            The same shape with the last axis set to `width`.
        """
        return (*tuple(input_shape[:-1]), self.width)

    def get_config(self) -> dict:
        """Return the configuration needed to rebuild this layer.

        Returns:
            The layer configuration.
        """
        config = super().get_config()
        config.update({"width": self.width})
        return config


@keras.saving.register_keras_serializable(package="kdp.moe")
class PerFeatureDense(keras.layers.Layer):
    """Project every feature with its own weights, in a single layer.

    Dict-mode Feature MoE used one `Dense` per feature. `.keras` stores a
    layer's weights under a name derived from its class and the order it was
    built -- `dense`, `dense_1`, `dense_2` -- rather than the name it was given,
    and Keras reorders sibling layers when it rebuilds a functional graph from
    config. Four sibling `Dense` layers therefore came back holding each other's
    kernels: a reloaded model returned one feature's projection under another
    feature's name, silently.

    One layer holding a weight per feature is arithmetically the same and has no
    sibling to be confused with.
    """

    def __init__(self, units: int, activation=None, **kwargs):
        """Initialize the layer.

        Args:
            units: Width of each feature's projection.
            activation: Activation applied to the result, by name or callable.
            **kwargs: Passed to the parent layer.
        """
        super().__init__(**kwargs)
        self.units = int(units)
        self.activation = keras.activations.get(activation)

    def build(self, input_shape) -> None:
        """Create one kernel and bias per feature.

        Args:
            input_shape: `[batch, num_features, input_dim]`.

        Raises:
            ValueError: If the feature or input dimension is not known.
        """
        _, num_features, input_dim = tuple(input_shape)
        if num_features is None or input_dim is None:
            raise ValueError(
                "PerFeatureDense needs a known number of features and input "
                f"width, got {tuple(input_shape)}.",
            )
        self.kernel = self.add_weight(
            name="kernel",
            shape=(num_features, input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(num_features, self.units),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs) -> tf.Tensor:
        """Apply each feature's own projection.

        Args:
            inputs: `[batch, num_features, input_dim]`.

        Returns:
            `[batch, num_features, units]`.
        """
        projected = tf.einsum("bfi,fio->bfo", inputs, self.kernel) + self.bias
        if self.activation is not None:
            projected = self.activation(projected)
        return projected

    def compute_output_shape(self, input_shape) -> tuple:
        """Report the projected shape.

        Args:
            input_shape: `[batch, num_features, input_dim]`.

        Returns:
            The same shape with the last axis set to `units`.
        """
        batch, num_features, _ = tuple(input_shape)
        return (batch, num_features, self.units)

    def get_config(self) -> dict:
        """Return the configuration needed to rebuild this layer.

        Returns:
            The layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": keras.activations.serialize(self.activation),
            },
        )
        return config


@keras.saving.register_keras_serializable(package="kdp.moe")
class StackFeaturesLayer(keras.layers.Layer):
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

        # This reported the *first* feature's width whatever the others were,
        # so Keras built a graph on a shape `tf.stack` cannot produce and the
        # model failed on its first real batch instead of at build time.
        widths = {shape[-1] for shape in input_shape if shape[-1] is not None}
        if len(widths) > 1:
            raise ValueError(
                "Every feature must be the same width to stack. Got "
                f"{sorted(widths)}; pad them to a common width first.",
            )

        batch_size = input_shape[0][0]
        feature_dim = input_shape[0][-1]
        num_features = len(input_shape)

        return (batch_size, num_features, feature_dim)

    def get_config(self) -> dict:
        """Get layer configuration for serialization."""
        return super().get_config()


@keras.saving.register_keras_serializable(package="kdp.moe")
class UnstackLayer(keras.layers.Layer):
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
        for _ in range(input_shape[self.axis]):
            shape = list(input_shape)
            del shape[self.axis]
            shapes.append(tuple(shape))
        return shapes

    def get_config(self) -> dict:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


@keras.saving.register_keras_serializable(package="kdp.moe")
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

    def build(self, input_shape) -> None:
        """Build the stacked layers so Keras does not mark the block falsely built.

        The layers are created in `__init__`, so without this Keras 3 warns that
        the block "does not have a `build()` method implemented and it looks
        like it has unbuilt state", and marks it built anyway.

        Args:
            input_shape: Shape of the input to the expert.
        """
        shape = tuple(input_shape)
        for layer in self.hidden_layers:
            layer.build(shape)
            shape = layer.compute_output_shape(shape)
        self.output_layer.build(shape)
        super().build(input_shape)

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
                layer,
                keras.layers.Dropout | keras.layers.BatchNormalization,
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


@keras.saving.register_keras_serializable(package="kdp.moe")
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
            routing_activation: Accepted and not used. Routing weights are
                always a softmax over the logits; "sparsemax" is not
                implemented. To route each feature to fewer experts, set
                `sparsity`, which masks all but its top-k logits.
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
        # `tf.nn.top_k` rejects a float `k`, and a saved config can bring
        # this back as one, so it is pinned to an int here.
        self.sparsity = int(min(sparsity, num_experts))
        self.routing_activation = routing_activation
        self.feature_names = feature_names
        self.predefined_assignments = predefined_assignments
        self.freeze_experts = freeze_experts
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Validate parameters
        if routing == "predefined":
            if not feature_names or not predefined_assignments:
                raise ValueError(
                    "For predefined routing, feature_names and predefined_assignments must be provided",
                )
            self._validate_assignments(
                feature_names=feature_names,
                assignments=predefined_assignments,
                num_experts=num_experts,
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

        # `freeze_experts` only passed `training=False` to each expert, which
        # controls dropout and batch-norm behaviour, not whether the weights
        # receive gradients -- so the experts were still trained. Marking them
        # untrainable is what the option is documented to do.
        if freeze_experts:
            for expert in self.experts:
                expert.trainable = False

        # Set up routing mechanism. Learned routing keeps its logits in a
        # weight created in `build`, one row per feature; see
        # `_compute_routing_weights` for why they cannot come from the input.
        self.routing_logits = None
        if routing != "learned":
            # Create a fixed assignment matrix for predefined routing
            self._create_assignment_matrix()

    def build(self, input_shape) -> None:
        """Create the learned routing logits, one row per feature.

        Args:
            input_shape: Shape of the stacked features,
                `[batch, num_features, feature_dim]`.

        Raises:
            ValueError: If the number of features is not known statically, or
                does not match the names given for predefined routing.
        """
        num_features = input_shape[1]
        if num_features is None:
            raise ValueError(
                "FeatureMoE needs to know how many features it is routing. "
                f"Got an unknown second dimension in {tuple(input_shape)}.",
            )
        if self.feature_names and len(self.feature_names) != num_features:
            raise ValueError(
                f"FeatureMoE was given {len(self.feature_names)} feature names "
                f"but {num_features} features to route.",
            )

        if self.routing == "learned":
            self.routing_logits = self.add_weight(
                name="routing_logits",
                shape=(num_features, self.num_experts),
                initializer="glorot_uniform",
                trainable=True,
            )

        for expert in self.experts:
            if not expert.built:
                expert.build(input_shape)
        super().build(input_shape)

    @staticmethod
    def _validate_assignments(
        feature_names: list[str],
        assignments: dict[str, int | dict[int, float]],
        num_experts: int,
    ) -> None:
        """Reject predefined assignments that would silently zero out a feature.

        The assignment matrix doubles as the routing weights, so a feature with
        no entry gets an all-zero row and its whole representation is multiplied
        away. An out-of-range expert index is the same failure with a different
        cause. Both are configuration mistakes, so they are reported rather than
        absorbed.

        Args:
            feature_names: Features that will be routed through the mixture.
            assignments: The caller's feature -> expert mapping.
            num_experts: How many experts exist to route to.

        Raises:
            ValueError: If a feature is unassigned or an expert index is invalid.
        """
        missing = [name for name in feature_names if name not in assignments]
        if missing:
            raise ValueError(
                "Predefined routing needs an expert for every feature. "
                f"Missing assignments for: {sorted(missing)}. "
                "Unassigned features would be zeroed out by the router.",
            )

        unknown = [name for name in assignments if name not in feature_names]
        if unknown:
            raise ValueError(
                f"Predefined assignments name features the mixture never sees: {sorted(unknown)}. "
                f"Known features: {sorted(feature_names)}.",
            )

        for name in feature_names:
            target = assignments[name]
            indices = target.keys() if isinstance(target, dict) else [target]
            for index in indices:
                if not isinstance(index, int) or isinstance(index, bool):
                    raise ValueError(
                        f"Expert index for {name!r} must be an int, got {index!r}.",
                    )
                if not 0 <= index < num_experts:
                    raise ValueError(
                        f"Expert index {index} for {name!r} is out of range "
                        f"for a mixture of {num_experts} experts.",
                    )

    def _create_assignment_matrix(self) -> None:
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

    def _compute_routing_weights(self, inputs, training=None) -> tf.Tensor:
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
            # The logits used to come from a Dense layer fed the batch *mean* of
            # the features. Routing therefore depended on which rows happened to
            # share a batch: changing one row moved every row's output, and a
            # record scored alone did not match the same record scored in a
            # batch. Feature-level routing is a property of the feature, so it
            # lives in a learned weight and is identical for every row.
            routing_logits = self.routing_logits  # [num_features, num_experts]

            # `sparsity` names how many experts each feature may use, and the
            # top-k mask below is what enforces it. It used to sit behind
            # `routing_activation != "softmax"`, a knob `PreprocessingModel`
            # never exposed, so the branch was unreachable and the documented
            # "use only top k experts" never happened: every feature was routed
            # densely to all of them.
            if self.sparsity < self.num_experts:
                # Sort logits and keep only top-k
                top_logits, top_indices = tf.nn.top_k(
                    routing_logits,
                    k=self.sparsity,
                    sorted=True,
                )

                # Create a mask for the top-k logits
                num_features = tf.shape(routing_logits)[0]
                mask = tf.scatter_nd(
                    indices=tf.stack(
                        [
                            tf.repeat(tf.range(num_features), self.sparsity),
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
            else:
                weights = tf.nn.softmax(routing_logits, axis=-1)

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
        for expert in self.experts:
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

    def get_expert_assignments(self, inputs=None) -> dict:
        """Report how much of each feature each expert handles.

        Learned routing used to return an empty dictionary here, so the
        documented way to see which expert handles which feature reported
        nothing at all for the default routing mode. The router decides from
        the feature representations rather than from its weights alone, so a
        batch is needed to answer the question.

        Args:
            inputs: A batch shaped like the one `call` receives,
                `[batch_size, num_features, feature_dim]`. Required for learned
                routing; ignored for predefined routing, whose assignments are
                fixed.

        Returns:
            dict: `{feature_name: {expert_index: weight}}`, keeping only the
            experts with a non-zero share. Feature names fall back to
            `feature_0`, `feature_1`, ... when the layer was built without them.

        Raises:
            ValueError: If learned routing is in use and no batch is given.
        """
        if self.routing == "predefined":
            assignments = {}
            for name, target in (self.predefined_assignments or {}).items():
                if isinstance(target, dict):
                    assignments[name] = {
                        int(index): float(weight) for index, weight in target.items()
                    }
                else:
                    assignments[name] = {int(target): 1.0}
            return assignments

        if inputs is None:
            raise ValueError(
                "Learned routing decides the assignments from the data, so a "
                "batch of stacked features is needed to report them. Pass the "
                "same tensor you would pass to the layer.",
            )

        weights = self._compute_routing_weights(inputs, training=False)[0]
        rows = weights.numpy()
        names = self.feature_names or [f"feature_{i}" for i in range(len(rows))]
        return {
            name: {
                index: float(weight)
                for index, weight in enumerate(row)
                if float(weight) > 0
            }
            for name, row in zip(names, rows, strict=False)
        }

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

    # Features rarely come out of preprocessing the same width, and stacking
    # needs them equal. `PreprocessingModel` pads them for exactly this reason;
    # doing it here too means this helper is not limited to the case where
    # every feature happens to match. The widest is padded by zero columns, so
    # nothing is added to it.
    widest = max(int(output.shape[-1]) for output in feature_outputs)
    padded_outputs = [
        PadFeatureLayer(width=widest, name=f"{name}_moe_pad")(output)
        for name, output in zip(feature_names, feature_outputs, strict=True)
    ]

    # Stack feature representations
    stacked_features = StackFeaturesLayer()(padded_outputs)

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
