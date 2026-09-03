# InferenceFormatter

Base class for formatting data for inference in various contexts.

This class provides common functionality for converting data to the format
required by preprocessors during inference, regardless of feature types.

Subclasses should implement specific formatting logic for different types
of features (time series, text, etc.).


## Constructor

```python
__init__(self, preprocessor)
```

Initialize the InferenceFormatter.

### Parameters- **preprocessor**: The trained preprocessor model to prepare data for


---

## prepare_inference_data

```python
prepare_inference_data(self, data: dict | pandas.core.frame.DataFrame, to_tensors: bool = False) -> dict | dict[str, tensorflow.python.framework.tensor.Tensor]
```

Prepare data for inference based on preprocessor requirements.

### Parameters- **data**: The data to make predictions on
- **to_tensors**: Whether to convert the output to TensorFlow tensors

### Returns

    Dict with properly formatted data for inference, either as Python types or as TensorFlow tensors


---
