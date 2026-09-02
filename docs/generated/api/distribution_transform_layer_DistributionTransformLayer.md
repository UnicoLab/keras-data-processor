# DistributionTransformLayer

Layer for transforming data distributions to improve anomaly detection.

This layer applies various transformations to make data more normally distributed
or to handle specific distribution types better. Supported transformations include
log, square root, Box-Cox, Yeo-Johnson, arcsinh, cube-root, logit, quantile,
robust-scale, and min-max.

When transform_type is set to 'auto', the layer automatically selects the most
appropriate transformation based on the data characteristics during training.

### Parameters- **transform_type**: Type of transformation to apply. Options are 'none', 'log', 'sqrt',
        'box-cox', 'yeo-johnson', 'arcsinh', 'cube-root', 'logit', 'quantile',
        'robust-scale', 'min-max', or 'auto'. Default is 'none'.
- **lambda_param**: Parameter for parameterized transformations like Box-Cox and Yeo-Johnson.
        Default is 0.0.
- **epsilon**: Small value added to prevent numerical issues like log(0). Default is 1e-10.
- **min_value**: Minimum value for min-max scaling. Default is 0.0.
- **max_value**: Maximum value for min-max scaling. Default is 1.0.
- **clip_values**: Whether to clip values to the specified range in min-max scaling. Default is True.
- **auto_candidates**: list of transformation types to consider when transform_type is 'auto'.
        If None, all available transformations will be considered. Default is None.
- **name**: Optional name for the layer.

Input shape:
    N-D tensor with shape: (batch_size, ..., features)

Output shape:
    Same shape as input: (batch_size, ..., features)

### Examples

    ```python
    import keras
    import numpy as np
    from kmr.layers import DistributionTransformLayer

    # Create sample input data with skewed distribution
    x = keras.random.exponential((32, 10))  # 32 samples, 10 features

    # Apply log transformation
    log_transform = DistributionTransformLayer(transform_type="log")
    y = log_transform(x)
    print("Transformed output shape:", y.shape)  # (32, 10)

    # Apply Box-Cox transformation with lambda=0.5
    box_cox = DistributionTransformLayer(transform_type="box-cox", lambda_param=0.5)
    z = box_cox(x)

    # Apply arcsinh transformation (handles both positive and negative values)
    arcsinh_transform = DistributionTransformLayer(transform_type="arcsinh")
    a = arcsinh_transform(x)

    # Apply min-max scaling to range [0, 1]
    min_max = DistributionTransformLayer(
        transform_type="min-max", min_value=0.0, max_value=1.0
    )
    b = min_max(x)

    # Use automatic transformation selection
    auto_transform = DistributionTransformLayer(transform_type="auto")
    c = auto_transform(x)  # Will select the best transformation during training
    ```


## Constructor

```python
__init__(self, transform_type: str = 'none', lambda_param: float = 0.0, epsilon: float = 1e-10, min_value: float = 0.0, max_value: float = 1.0, clip_values: bool = True, auto_candidates: list[str] | None = None, name: str | None = None, **kwargs: Any) -> None
```

Initialize the DistributionTransformLayer.

See the class docstring for the accepted arguments and what
each one controls.


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
build(self, input_shape: tuple[int, ...]) -> None
```

Builds the layer with the given input shape.

### Parameters- **input_shape**: tuple of integers defining the input shape.


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

Apply the selected transformation to the inputs.

### Parameters- **inputs**: Input tensor
- **training**: Boolean indicating whether the layer should behave in
        training mode or inference mode

### Returns

    Transformed tensor with the same shape as input


---

## compute_output_shape

```python
compute_output_shape(self, input_shape) -> tuple
```

Compute the output shape of the layer.

### Parameters- **input_shape**: Shape tuple (tuple of integers) or TF TensorShape

### Returns

    Output shape (tuple of integers or TensorShape)


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
get_config(self) -> dict[str, typing.Any]
```

Get the layer configuration.

### Returns

    Configuration dictionary


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

