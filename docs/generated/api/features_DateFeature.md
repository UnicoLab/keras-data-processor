# DateFeature

TextFeature with dynamic kwargs passing.

## Constructor

```python
__init__(self, name: str, feature_type: kdp.features.FeatureType = <FeatureType.DATE: 9>, **kwargs) -> None
```

Initializes a DateFeature instance.

### Parameters- **name (str)**: The name of the feature.
    feature_type (FeatureType): The type of the feature.
    **kwargs: Additional keyword arguments for the feature.


---

## add_preprocessor

```python
add_preprocessor(self, preprocessor: kdp.layers_factory.PreprocessorLayerFactory | typing.Any) -> None
```

Adds a preprocessor to the feature.

### Parameters- **preprocessor (Union[PreprocessorLayerFactory, Any])**: The preprocessor to add.


---

## from_string

```python
from_string(type_str: str) -> 'FeatureType'
```

Converts a string to a FeatureType.

### Parameters- **type_str (str)**: The string representation of the feature type.


---

## update_kwargs

```python
update_kwargs(self, **kwargs) -> None
```

Updates the kwargs with new or modified parameters.

### Parameters

    **kwargs: The new or modified parameters.


---

