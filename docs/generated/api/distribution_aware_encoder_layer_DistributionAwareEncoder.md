# DistributionAwareEncoder

An advanced layer that adapts its encoding based on the input distribution.

This layer automatically detects and handles various distribution types using
the DistributionTransformLayer. It identifies the data distribution type and
applies appropriate transformations for better model performance.

Key- **features**: 1. Auto-detection of distribution types to apply optimal transformations
2. Periodic feature detection and automatic encoding with Fourier features (sin/cos)
3. Optional projection to a fixed embedding dimension
4. Distribution-specific embeddings that can be added to the outputs
5. Graph mode compatibility for use in TensorFlow's static graph execution

Supported distributions include:
- Normal distributions: For normally distributed data
- Heavy-tailed distributions: For data with heavier tails than normal
- Multimodal distributions: For data with multiple peaks
- Uniform distributions: For evenly distributed data
- Exponential distributions: For data with exponential decay
- Log-normal distributions: For data that is normal after log transform
- Discrete distributions: For data with finite distinct values
- Periodic distributions: For data with cyclic patterns (adds sin/cos features)
- Sparse distributions: For data with many zeros
- Beta distributions: For bounded data between 0 and 1
- Gamma distributions: For positive, right-skewed data
- Poisson distributions: For count data
- Cauchy distributions: For extremely heavy-tailed data
- Zero-inflated distributions: For data with excess zeros
- Bounded distributions: For data with known bounds
- Ordinal distributions: For ordered categorical data

The layer uses pure TensorFlow operations without dependencies on TensorFlow Probability,
and is compatible with both eager execution and graph mode.


## Constructor

```python
__init__(self, embedding_dim=None, epsilon=1e-06, detect_periodicity=True, handle_sparsity=True, auto_detect=True, distribution_type='unknown', transform_type='auto', add_distribution_embedding=False, name='distribution_aware_encoder', trainable=True, num_bins=None, adaptive_binning=None, mixture_components=None, prefered_distribution=None, **kwargs)
```

Initialize the DistributionAwareEncoder.

### Parameters- **embedding_dim**: Optional output dimension for feature projection. If specified,
        a Dense layer will project the transformed features to this dimension.
        If None, the original feature dimension is preserved. Default is None.
- **epsilon**: Small value to prevent numerical issues. Default is 1e-6.
- **detect_periodicity**: If True, the layer will check for and handle periodic patterns
        by adding sin/cos features, increasing the output dimension. When True, periodic
        data will have 3x the original feature dimension. Default is True.
- **handle_sparsity**: If True, special handling for sparse data (many zeros). Default is True.
- **auto_detect**: If True, automatically detect the distribution type during training.
        If False, the specified distribution_type will be used. Default is True.
- **distribution_type**: The specific distribution type to use if auto_detect is False.
        Must be one of the values in DistributionType. Default is "unknown".
- **transform_type**: Type of transformation to apply via DistributionTransformLayer.
        Options include "none", "log", "sqrt", "box-cox", etc. Default is "auto".
- **add_distribution_embedding**: If True, adds a learned embedding for the detected
        distribution type to the output, increasing the output dimension. Default is False.
- **name**: Name for the layer. Default is "distribution_aware_encoder".
- **trainable**: Whether the layer is trainable. Default is True.

    # Legacy parameters (maintained for backward compatibility)
- **num_bins**: Number of bins for legacy histogram-based encoding. Not used in current implementation.
- **adaptive_binning**: Whether to use adaptive binning. Not used in current implementation.
- **mixture_components**: Number of mixture components. Not used in current implementation.
- **prefered_distribution**: Legacy way to specify distribution_type. If provided, auto_detect
        will be set to False and this value will be used as distribution_type.
    **kwargs: Additional arguments forwarded to `keras.layers.Layer`.

Note on output dimensions:
    - If detect_periodicity=True and periodic features are detected/forced:
        output_dim = input_dim * 3 (original + sin + cos features)
    - If embedding_dim is specified:
        output_dim = embedding_dim
    - If add_distribution_embedding=True:
        output_dim += 8 (distribution embedding dimension)


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

Build the layer.

### Parameters- **input_shape**: Shape of input tensor


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

Apply distribution-aware encoding to the inputs.

This- **method**: 1. Detects the distribution type of input data (if auto_detect=True and in training mode)
2. Applies appropriate transformations based on the detected distribution
3. Optionally adds distribution embedding vectors
4. Optionally adds periodic features (sin/cos transformations) for periodic data
5. Optionally projects to the specified embedding dimension

### Parameters
- **inputs**: Input tensor with shape (batch_size, ..., features)
- **training**: Boolean indicating if in training mode (True) or inference mode (False).
              When True and auto_detect=True, distribution type is detected.
              When False, uses previously detected distribution type.

### Returns

    Transformed tensor with shape depending on configuration:
    - Base case: Same shape as input
    - With periodic features: (batch_size, ..., features*3)
    - With embedding_dim: (batch_size, ..., embedding_dim)
    - With distribution_embedding: Output has 8 additional dimensions

### Notes

    During inference (training=False), the layer uses the distribution type
    detected during training. This ensures consistent behavior between training
    and inference.


---

## compute_output_shape

```python
compute_output_shape(self, input_shape) -> tuple
```

Compute the output shape of the layer based on input shape and layer configuration.

### Parameters- **input_shape**: Shape tuple (tuple of integers) or TensorShape

### Returns

    Output shape tuple or TensorShape
- **Notes**: The output shape depends on several factors:
    1. If detect_periodicity=True and periodicity is detected/forced:
       - Feature dimension is multiplied by 3 (original + sin + cos features)
    2. If embedding_dim is specified:
       - Output feature dimension will be embedding_dim
    3. If add_distribution_embedding=True:
       - 8 dimensions are added for the distribution embedding

    These transformations are applied in sequence, if applicable.


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

Get the layer configuration for serialization.

This method enables serialization and deserialization of the layer via
`keras.saving.save_model()` and `keras.saving.load_model()`.

### Returns

    Configuration dictionary containing all parameters needed to reconstruct the layer.

### Notes

    When saving a model containing a DistributionAwareEncoder layer, use the
    `get_custom_objects()` function to provide the necessary custom objects- **dictionary**: ```python
    model.save("my_model.keras")
    custom_objects = get_custom_objects()
    loaded_model = keras.saving.load_model(
        "my_model", custom_objects=custom_objects
    )
    ```


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

