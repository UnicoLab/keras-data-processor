# FeaturePreprocessor

## Constructor

```python
__init__(self, name: str, use_dynamic: bool = False) -> None
```

Initializes a feature preprocessor.

### Parameters- **name (str)**: The name of the feature preprocessor.
    use_dynamic (bool): Whether to use the dynamic preprocessing pipeline.


---

## add_processing_step

```python
add_processing_step(self, layer_creator: collections.abc.Callable[..., keras.src.layers.layer.Layer] = None, **layer_kwargs) -> None
```

Add a preprocessing layer to the feature preprocessor pipeline.
If using the standard pipeline, a ProcessingStep is added.
Otherwise, the layer is added to a list for dynamic handling.

### Parameters- **layer_creator (Callable[..., keras.layers.Layer])**: A callable that creates a layer.
        If not provided, the default layer creator is used.
    **layer_kwargs: Additional keyword arguments for the layer creator.


---

## chain

```python
chain(self, input_layer) -> keras.src.layers.layer.Layer
```

Chains the processing steps starting from the given input_layer.

For a static pipeline, this delegates to the internal Pipeline's chain() method.
For the dynamic pipeline, it constructs the dynamic pipeline on the fly.


---

## transform

```python
transform(self, input_data: tensorflow.python.framework.tensor.Tensor) -> tensorflow.python.framework.tensor.Tensor
```

Process the input data through the pipeline.
For the dynamic pipeline, wrap input in a dictionary and extract final output.

### Parameters- **input_data**: The input data to process.

### Returns

    tf.Tensor: The processed data.


---
