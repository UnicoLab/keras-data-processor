# ProcessingStep

## Constructor

```python
__init__(self, layer_creator: collections.abc.Callable[..., keras.src.layers.layer.Layer], **layer_kwargs) -> None
```

Initialize a processing step.

### Parameters- **layer_creator (Callable[..., keras.layers.Layer])**: A callable that creates a layer.
    **layer_kwargs: Additional keyword arguments for the layer creator.


---

## connect

```python
connect(self, input_layer) -> keras.src.layers.layer.Layer
```

Connect this step's layer to an input layer and return the output layer.

### Parameters- **input_layer**: The input layer to connect to.


---

## process

```python
process(self, input_data) -> keras.src.layers.layer.Layer
```

Apply the processing step to the input data.

### Parameters- **input_data**: The input data to process.


---
