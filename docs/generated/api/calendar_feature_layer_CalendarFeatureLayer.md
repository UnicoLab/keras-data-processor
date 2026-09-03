# CalendarFeatureLayer

Layer for generating calendar features from date or timestamp inputs.

This layer extracts calendar features like day of week, month, is_weekend,
etc. from date or timestamp inputs. These features can help models
learn seasonal patterns related to the calendar.

### Parameters- **features**: List of calendar features to extract. Options:
        - 'year': Year as a float
        - 'month': Month of year (1-12)
        - 'day': Day of month (1-31)
        - 'day_of_week': Day of week (0-6, 0 is Monday)
        - 'day_of_year': Day of year (1-366)
        - 'week_of_year': Week of year (1-53)
        - 'is_weekend': Binary indicator for weekend
        - 'quarter': Quarter of year (1-4)
        - 'is_month_start': Binary indicator for first day of month
        - 'is_month_end': Binary indicator for last day of month
        - 'is_quarter_start': Binary indicator for first day of quarter
        - 'is_quarter_end': Binary indicator for last day of quarter
        - 'is_year_start': Binary indicator for first day of year
        - 'is_year_end': Binary indicator for last day of year
        - 'month_sin': Sinusoidal encoding of month
        - 'month_cos': Cosinusoidal encoding of month
        - 'day_sin': Sinusoidal encoding of day of month
        - 'day_cos': Cosinusoidal encoding of day of month
        - 'day_of_week_sin': Sinusoidal encoding of day of week
        - 'day_of_week_cos': Cosinusoidal encoding of day of week
- **cyclic_encoding**: Deprecated and ignored. It never changed the output.
        Ask for the sin/cos components by name -- 'month_sin',
        'month_cos', 'day_of_week_sin', 'day_of_week_cos' -- to get them.
- **input_format**: Format of the input date string. Default is '%Y-%m-%d'.
- **normalize**: Whether to normalize numeric features to [0, 1] range.
- **onehot_categorical**: Accepted and not used. Every requested
        feature comes back as a single numeric column whatever this
        says.


## Constructor

```python
__init__(self, features=None, cyclic_encoding=<unset>, input_format='%Y-%m-%d', normalize=True, onehot_categorical=False, **kwargs)
```

Initialize the CalendarFeatureLayer.

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
build(self, input_shape) -> None
```

Build the layer's weights for a given input shape.

### Parameters- **input_shape**: Shape of the input tensor.


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

Extract calendar features from date inputs.

### Parameters- **inputs**: Input tensor of shape (batch_size, 1) or (batch_size,) with date strings
- **training**: Boolean tensor indicating whether the call is for training

### Returns

    Tensor with extracted calendar features


---

## compute_output_shape

```python
compute_output_shape(self, input_shape) -> tuple
```

Compute the output shape of the layer.

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

Return the configuration of the layer.

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

