# DynamicPreprocessingPipeline

Chains Keras preprocessing layers over a dictionary of features.

Every layer is addressed by its ``name``. When the pipeline runs, a layer
reads its input from the entry that shares its name if that entry exists,
and otherwise from the output of the layer that precedes it. The result is
always written back under the layer's own name, so intermediate results
stay available to later layers and to the caller.

That single rule covers both common- **shapes**: * supplying one entry per layer runs the layers independently over their
  own inputs;
* supplying only the first layer's entry chains the layers, each one
  consuming what the previous layer produced.

### Examples

    ```python
    pipeline = DynamicPreprocessingPipeline(
        [ScalingLayer(name="scaling"), LogLayer(name="log")]
    )
    # "log" is absent from the data, so it consumes the scaling output.
    out = pipeline.transform({"scaling": tf.constant([[1.0], [2.0]])})
    out["scaling"], out["log"]
    ```


## Constructor

```python
__init__(self, layers: list) -> None
```

Initializes the pipeline with a list of preprocessing layers.

### Parameters- **layers (list)**: A list of TensorFlow preprocessing layers. Each layer
        must have a unique ``name``, which doubles as the key it reads
        from and writes to.

### Raises
- **ValueError**: If two layers share the same name.


---

## initialize_and_transform

```python
initialize_and_transform(self, features: dict) -> dict
```

Applies every layer to the feature dictionary.

Kept as an alias- **of**: meth:`transform` for backwards compatibility.

### Parameters

    features (dict): Mapping of feature names to tensors.

### Returns
- **dict**: The input entries plus one entry per layer.


---

## process

```python
process(self, dataset: tensorflow.python.data.ops.dataset_ops.DatasetV2) -> tensorflow.python.data.ops.dataset_ops.DatasetV2
```

Processes the dataset through the pipeline using the tf.data API.

### Parameters- **dataset (tf.data.Dataset)**: A dataset whose elements are dictionaries
        of features.

### Returns

    tf.data.Dataset: The processed dataset, with each layer's output
        stored under the layer's name.


---

## transform

```python
transform(self, features: dict) -> dict
```

Applies every layer to the feature dictionary.

### Parameters- **features (dict)**: Mapping of feature names to tensors. It is not
        modified; a new dictionary is returned.

### Returns
- **dict**: The input entries plus one entry per layer, holding that
        layer's output.

### Raises
- **KeyError**: If a layer has no entry to read from, i.e. neither its own
        key nor the previous layer's output is available.


---
