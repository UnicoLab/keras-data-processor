# FeatureMoE

Feature-wise Mixture of Experts layer.

Routes different features to different expert networks based on- **either**: 1. Learned routing (trained router network)
2. Predefined assignments (manual specification)


## Constructor

```python
__init__(self, num_experts: int = 4, expert_dim: int = 64, expert_hidden_dims: list[int] = None, routing: str = 'learned', sparsity: int = 2, routing_activation: str = 'softmax', feature_names: list[str] | None = None, predefined_assignments: dict[str, int] | None = None, freeze_experts: bool = False, dropout_rate: float = 0.0, use_batch_norm: bool = True, name: str | None = None, trainable: bool = True, dtype=None, **kwargs)
```

Initialize the Feature-wise MoE layer.

### Parameters- **num_experts**: Number of expert networks
- **expert_dim**: Output dimension of each expert
- **expert_hidden_dims**: Hidden dimensions for each expert
- **routing**: Routing mechanism - "learned" or "predefined"
- **sparsity**: Number of experts to use per feature (for sparse routing)
- **routing_activation**: Accepted and not used. Routing weights are
        always a softmax over the logits; "sparsemax" is not
        implemented. To route each feature to fewer experts, set
        `sparsity`, which masks all but its top-k logits.
- **feature_names**: Names of input features (required for predefined routing)
- **predefined_assignments**: Mapping from feature name to expert index
- **freeze_experts**: Whether to freeze the expert weights during training
- **dropout_rate**: Dropout rate for the experts
- **use_batch_norm**: Whether to use batch normalization in experts
- **name**: Optional name for the layer
- **trainable**: Whether the layer is trainable
- **dtype**: Data type of the layer
    **kwargs: Additional keyword arguments passed to the parent class


---

## add_loss

```python
add_loss(self, loss)
```

Can be called inside of the `call()` method to add a scalar loss.

### Examples


```python
class- **MyLayer(Layer)**: ...
    def call(self, x):
        self.add_loss(ops.sum(x))
        return x
```


---

## add_variable

```python
add_variable(self, shape, initializer, dtype=None, trainable=True, autocast=True, regularizer=None, constraint=None, name=None)
```

Add a weight variable to the layer.

Alias of `add_weight()`.


---

## add_weight

```python
add_weight(self, shape=None, initializer=None, dtype=None, trainable=True, autocast=True, regularizer=None, constraint=None, aggregation='none', overwrite_with_gradient=False, name=None)
```

Add a weight variable to the layer.

### Parameters- **shape**: Shape tuple for the variable. Must be fully-defined
        (no `None` entries). Defaults to `()` (scalar) if unspecified.
- **initializer**: Initializer object to use to populate the initial
        variable value, or string name of a built-in initializer
        (e.g. `"random_normal"`). If unspecified, defaults to
        `"glorot_uniform"` for floating-point variables and to `"zeros"`
        for all other types (e.g. int, bool).
- **dtype**: Dtype of the variable to create, e.g. `"float32"`. If
        unspecified, defaults to the layer's variable dtype
        (which itself defaults to `"float32"` if unspecified).
- **trainable**: Boolean, whether the variable should be trainable via
        backprop or whether its updates are managed manually. Defaults
        to `True`.
- **autocast**: Boolean, whether to autocast layers variables when
        accessing them. Defaults to `True`.
- **regularizer**: Regularizer object to call to apply penalty on the
        weight. These penalties are summed into the loss function
        during optimization. Defaults to `None`.
- **constraint**: Contrainst object to call on the variable after any
        optimizer update, or string name of a built-in constraint.
        Defaults to `None`.
- **aggregation**: Optional string, one of `None`, `"none"`, `"mean"`,
        `"sum"` or `"only_first_replica"`. Annotates the variable with
        the type of multi-replica aggregation to be used for this
        variable when writing custom data parallel training loops.
        Defaults to `"none"`.
- **overwrite_with_gradient**: Boolean, whether to overwrite the variable
        with the computed gradient. This is useful for float8 training.
        Defaults to `False`.
- **name**: String name of the variable. Useful for debugging purposes.


---

## build

```python
build(self, input_shape) -> None
```

Create the learned routing logits, one row per feature.

### Parameters- **input_shape**: Shape of the stacked features,
        `[batch, num_features, feature_dim]`.

### Raises
- **ValueError**: If the number of features is not known statically, or
        does not match the names given for predefined routing.


---

## build_from_config

```python
build_from_config(self, config)
```

Builds the layer's states with the supplied config dict.

By default, this method calls the `build(config["input_shape"])` method,
which creates weights based on the layer's input shape in the supplied
config. If your config contains other information needed to load the
layer's state, you should override this method.

### Parameters- **config**: Dict containing the input shape associated with this layer.


---

## call

```python
call(self, inputs, training=None) -> tensorflow.python.framework.tensor.Tensor
```

Forward pass through the Feature-wise MoE.

### Parameters- **inputs**: Input tensor of shape [batch_size, num_features, feature_dim]
- **training**: Whether in training mode

### Returns

    Output tensor of shape [batch_size, num_features, expert_dim]


---

## count_params

```python
count_params(self)
```

Count the total number of scalars composing the weights.

### Returns

    An integer count.


---

## get_build_config

```python
get_build_config(self)
```

Returns a dictionary with the layer's input shape.

This method returns a config dict that can be used by
`build_from_config(config)` to create all states (e.g. Variables and
Lookup tables) needed by the layer.

By default, the config only contains the input shape that the layer
was built with. If you're writing a custom layer that creates state in
an unusual way, you should override this method to make sure this state
is already created when Keras attempts to load its value upon model
loading.

### Returns

    A dict containing the input shape associated with the layer.


---

## get_config

```python
get_config(self) -> dict
```

Get layer configuration for serialization.

---

## get_expert_assignments

```python
get_expert_assignments(self, inputs=None) -> dict
```

Report how much of each feature each expert handles.

Learned routing used to return an empty dictionary here, so the
documented way to see which expert handles which feature reported
nothing at all for the default routing mode. The router decides from
the feature representations rather than from its weights alone, so a
batch is needed to answer the question.

### Parameters- **inputs**: A batch shaped like the one `call` receives,
        `[batch_size, num_features, feature_dim]`. Required for learned
        routing; ignored for predefined routing, whose assignments are
        fixed.

### Returns
- **dict**: `{feature_name: {expert_index: weight}}`, keeping only the
    experts with a non-zero share. Feature names fall back to
    `feature_0`, `feature_1`, ... when the layer was built without them.

### Raises
- **ValueError**: If learned routing is in use and no batch is given.


---

## get_weights

```python
get_weights(self)
```

Return the values of `layer.weights` as a list of NumPy arrays.

---

## load_own_variables

```python
load_own_variables(self, store)
```

Loads the state of the layer.

You can override this method to take full control of how the state of
the layer is loaded upon calling `keras.models.load_model()`.

### Parameters- **store**: Dict from which the state of the model will be loaded.


---

## rematerialized_call

```python
rematerialized_call(self, layer_call, *args, **kwargs)
```

Enable rematerialization dynamically for layer's call method.

### Parameters- **layer_call**: The original `call` method of a layer.

### Returns

    Rematerialized layer's `call` method.


---

## save_own_variables

```python
save_own_variables(self, store)
```

Saves the state of the layer.

You can override this method to take full control of how the state of
the layer is saved upon calling `model.save()`.

### Parameters- **store**: Dict where the state of the model will be saved.


---

## set_weights

```python
set_weights(self, weights)
```

Sets the values of `layer.weights` from a list of NumPy arrays.

---

## stateless_call

```python
stateless_call(self, trainable_variables, non_trainable_variables, *args, return_losses=False, **kwargs)
```

Call the layer without any side effects.

### Parameters- **trainable_variables**: List of trainable variables of the model.
- **non_trainable_variables**: List of non-trainable variables of the
        model.
    *args: Positional arguments to be passed to `call()`.
- **return_losses**: If `True`, `stateless_call()` will return the list of
        losses created during `call()` as part of its return values.
    **kwargs: Keyword arguments to be passed to `call()`.

### Returns

    A tuple. By default, returns `(outputs, non_trainable_variables)`.
        If `return_losses = True`, then returns
        `(outputs, non_trainable_variables, losses)`.
- **Note**: `non_trainable_variables` include not only non-trainable weights
such as `BatchNormalization` statistics, but also RNG seed state
(if there are any random operations part of the layer, such as dropout),
and `Metric` state (if there are any metrics attached to the layer).
These are all elements of state of the layer.

### Examples


```python
model = ...
data = ...
trainable_variables = model.trainable_variables
non_trainable_variables = model.non_trainable_variables
# Call the model with zero side effects
outputs, non_trainable_variables = model.stateless_call(
    trainable_variables,
    non_trainable_variables,
    data,
)
# Attach the updated state to the model
# (until you do this, the model is still in its pre-call state).
for ref_var, value in zip(
    model.non_trainable_variables, non_trainable_variables
):
    ref_var.assign(value)
```


---
