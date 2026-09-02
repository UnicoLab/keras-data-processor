# NumericalFeature

NumericalFeature with dynamic kwargs passing and embedding support.

## Constructor

```python
__init__(self, name: str, feature_type: kdp.features.FeatureType = <FeatureType.FLOAT_NORMALIZED: 2>, preferred_distribution: kdp.features.DistributionType | None = None, use_embedding: bool = False, embedding_dim: int = 8, num_bins: int = 10, **kwargs) -> None
```

Initializes a NumericalFeature instance.

### Parameters- **name (str)**: The name of the feature.
    feature_type (FeatureType): The type of the feature.
    preferred_distribution (DistributionType | None): The preferred distribution type.
    use_embedding (bool): Whether to use advanced numerical embedding.
    embedding_dim (int): Dimension of the embedding space.
    num_bins (int): Number of bins for discretization.
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

## get_embedding_layer

```python
get_embedding_layer(self, input_shape: tuple | None = None) -> keras.src.layers.layer.Layer
```

Creates and returns a NumericalEmbedding layer configured for this feature.

### Parameters- **input_shape**: Unused. `NumericalEmbedding` derives the feature count
        in its own `build`, so nothing here depends on the shape. The
        parameter is kept, and optional, so existing callers that pass
        it keep working.

### Returns

    A `NumericalEmbedding` layer built from this feature's settings.


---

## update_kwargs

```python
update_kwargs(self, **kwargs) -> None
```

Updates the kwargs with new or modified parameters.

### Parameters

    **kwargs: The new or modified parameters.


---

