# Pipeline

## Constructor

```python
__init__(self, steps: list[kdp.pipeline.ProcessingStep] = None, name: str = '') -> None
```

Initialize a pipeline with a list of processing steps.

### Parameters- **steps (list[ProcessingStep])**: A list of processing steps.
    name (str): The name of the pipeline.


---

## add_step

```python
add_step(self, step: kdp.pipeline.ProcessingStep) -> None
```

Add a processing step to the pipeline.

### Parameters- **step (ProcessingStep)**: The processing step to add.


---

## chain

```python
chain(self, input_layer) -> keras.src.layers.layer.Layer
```

Chain the pipeline steps by connecting each step in sequence, starting from the input layer.

### Parameters- **input_layer**: The input layer to start the chain from.


---

## transform

```python
transform(self, input_data: tensorflow.python.framework.tensor.Tensor) -> tensorflow.python.framework.tensor.Tensor
```

Apply the pipeline to the input data.

### Parameters- **input_data**: The input data to process.

### Returns

    tf.Tensor: The processed data.


---
