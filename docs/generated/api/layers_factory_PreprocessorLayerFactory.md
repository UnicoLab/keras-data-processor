# PreprocessorLayerFactory

## cast_to_float32_layer

```python
cast_to_float32_layer(name: str = 'cast_to_float32', **kwargs: dict) -> keras.src.layers.layer.Layer
```

Create a CastToFloat32Layer layer.

### Parameters- **name**: The name of the layer.
    **kwargs: Additional keyword arguments to pass to the layer constructor.

### Returns

    An instance of the CastToFloat32Layer layer.


---

## create_layer

```python
create_layer(layer_class: str | object, name: str = None, **kwargs: Any) -> keras.src.layers.layer.Layer
```

Create a layer using the layer class name, automatically filtering kwargs based on the layer class.

### Parameters- **layer_class (str | Class Object)**: The name of the layer class to be created
        (e.g., 'Normalization', 'Rescaling') or the class object itself.
    name (str): The name of the layer. Optional.
    **kwargs: Additional keyword arguments to pass to the layer constructor.

### Returns

    An instance of the specified layer class.


---

## date_encoding_layer

```python
date_encoding_layer(name: str = 'date_encoding_layer', **kwargs: dict) -> keras.src.layers.layer.Layer
```

Create a DateEncodingLayer layer.

### Parameters- **name**: The name of the layer.
    **kwargs: Additional keyword arguments to pass to the layer constructor.

### Returns

    An instance of the DateEncodingLayer layer.


---

## date_parsing_layer

```python
date_parsing_layer(name: str = 'date_parsing_layer', **kwargs: dict) -> keras.src.layers.layer.Layer
```

Create a DateParsingLayer layer.

### Parameters- **name**: The name of the layer.
    **kwargs: Additional keyword arguments to pass to the layer constructor.

### Returns

    An instance of the DateParsingLayer layer.


---

## date_season_layer

```python
date_season_layer(name: str = 'date_season_layer', **kwargs: dict) -> keras.src.layers.layer.Layer
```

Create a SeasonLayer layer.

### Parameters- **name**: The name of the layer.
    **kwargs: Additional keyword arguments to pass to the layer constructor.

### Returns

    An instance of the SeasonLayer layer.


---

## differencing_layer

```python
differencing_layer(name: str = 'differencing', order: int = 1, fill_value: float = 0.0, drop_na: bool = True, **kwargs: dict) -> keras.src.layers.layer.Layer
```

Create a DifferencingLayer for differencing time series data to make it stationary.

### Parameters- **name**: Name of the layer.
- **order**: Order of differencing. Default is 1.
- **fill_value**: Value to use for filling initial values. Default is 0.0.
- **drop_na**: Whether to drop rows with NaN values. Default is True.
    **kwargs: Additional keyword arguments.

### Returns

    DifferencingLayer instance.


---

## distribution_aware_encoder

```python
distribution_aware_encoder(name: str = 'distribution_aware', num_bins: int = 1000, epsilon: float = 1e-06, detect_periodicity: bool = True, handle_sparsity: bool = True, adaptive_binning: bool = True, mixture_components: int = 3, prefered_distribution: 'DistributionType' = None, **kwargs: Any) -> keras.src.layers.layer.Layer
```

Create a DistributionAwareEncoder layer.

### Parameters- **name (str)**: Name of the layer
    num_bins (int): Number of bins for quantile encoding
    epsilon (float): Small value for numerical stability
    detect_periodicity (bool): Whether to detect and handle periodic patterns
    handle_sparsity (bool): Whether to handle sparse data specially
    adaptive_binning (bool): Whether to use adaptive binning
    mixture_components (int): Number of components for mixture modeling
    prefered_distribution (DistributionType): Optional specific distribution type to use.
        When given, automatic distribution detection is disabled.
    **kwargs: Additional keyword arguments

### Returns

    DistributionAwareEncoder layer


---

## distribution_transform_layer

```python
distribution_transform_layer(name: str = 'distribution_transform', transform_type: str = 'none', lambda_param: float = 0.0, epsilon: float = 1e-10, min_value: float = 0.0, max_value: float = 1.0, clip_values: bool = True, auto_candidates: list[str] = None, **kwargs: Any) -> keras.src.layers.layer.Layer
```

Create a DistributionTransformLayer layer.

### Parameters- **name (str)**: Name of the layer
    transform_type (str): Type of transformation to apply
    lambda_param (float): Parameter for parameterized transformations
    epsilon (float): Small value for numerical stability
    min_value (float): Minimum value for min-max scaling
    max_value (float): Maximum value for min-max scaling
    clip_values (bool): Whether to clip values to the specified range
    auto_candidates (list[str]): List of transformations to consider in auto mode
    **kwargs: Additional keyword arguments

### Returns

    DistributionTransformLayer layer


---

## gated_linear_unit_layer

```python
gated_linear_unit_layer(units: int, name: str = 'gated_linear_unit', **kwargs: dict) -> keras.src.layers.layer.Layer
```

Create a GatedLinearUnit layer.

### Parameters- **units (int)**: Dimensionality of the output space
    name (str): Name of the layer
    **kwargs: Additional arguments to pass to the layer

### Returns
- **GatedLinearUnit**: A GatedLinearUnit layer instance


---

## gated_residual_network_layer

```python
gated_residual_network_layer(units: int, dropout_rate: float = 0.2, name: str = 'gated_residual_network', **kwargs: dict) -> keras.src.layers.layer.Layer
```

Create a GatedResidualNetwork layer.

### Parameters- **units (int)**: Dimensionality of the output space
    dropout_rate (float): Fraction of the input units to drop
    name (str): Name of the layer
    **kwargs: Additional arguments to pass to the layer

### Returns
- **GatedResidualNetwork**: A GatedResidualNetwork layer instance


---

## global_numerical_embedding_layer

```python
global_numerical_embedding_layer(global_embedding_dim: int = 8, global_mlp_hidden_units: int = 16, global_num_bins: int = 10, global_init_min: float = -3.0, global_init_max: float = 3.0, global_dropout_rate: float = 0.1, global_use_batch_norm: bool = True, global_pooling: str = 'average', name: str = 'global_numerical_embedding', **kwargs: dict) -> keras.src.layers.layer.Layer
```

Create a GlobalNumericalEmbedding layer.

### Parameters- **global_embedding_dim (int)**: Dimension of the final global embedding
    global_mlp_hidden_units (int): Number of hidden units in the global MLP
    global_num_bins (int): Number of bins for discretization
    global_init_min (float): Minimum value for initialization
    global_init_max (float): Maximum value for initialization
    global_dropout_rate (float): Dropout rate for regularization
    global_use_batch_norm (bool): Whether to use batch normalization
    global_pooling (str): Pooling method to use ("average" or "max")
    name (str): Name of the layer
    **kwargs: Additional arguments to pass to the layer

### Returns
- **GlobalNumericalEmbedding**: A GlobalNumericalEmbedding layer instance


---

## lag_feature_layer

```python
lag_feature_layer(name: str = 'lag_feature', lags: list[int] = None, fill_value: float = 0.0, drop_na: bool = True, **kwargs: dict) -> keras.src.layers.layer.Layer
```

Create a LagFeatureLayer for generating lag features from time series data.

### Parameters- **name**: Name of the layer.
- **lags**: List of lag values to create. Default is [1] (one step back).
- **fill_value**: Value to use for filling NaN values. Default is 0.0.
- **drop_na**: Whether to drop rows with NaN values. Default is True.
    **kwargs: Additional keyword arguments.

### Returns

    LagFeatureLayer instance.


---

## moving_average_layer

```python
moving_average_layer(name: str = 'moving_average', periods: list[int] = None, pad_value: float = 0.0, keep_original: bool = True, **kwargs: dict) -> keras.src.layers.layer.Layer
```

Create a MovingAverageLayer for computing moving averages to smooth time series data.

### Parameters- **name**: Name of the layer.
- **periods**: List of periods (window sizes) for moving averages. Default is [7] (7-period MA).
- **pad_value**: Value to use for padding. Default is 0.0.
- **keep_original**: Whether to keep the original series alongside MAs. Default is True.
    **kwargs: Additional keyword arguments.

### Returns

    MovingAverageLayer instance.


---

## multi_resolution_attention_layer

```python
multi_resolution_attention_layer(num_heads: int, d_model: int, embedding_dim: int = 32, name: str = 'multi_resolution_attention', **kwargs: dict) -> keras.src.layers.layer.Layer
```

Create a MultiResolutionTabularAttention layer.

### Parameters- **num_heads (int)**: Number of attention heads
    d_model (int): Dimensionality of the attention model
    embedding_dim (int): Dimension for categorical embeddings
    name (str): Name of the layer
    **kwargs: Additional arguments to pass to the layer

### Returns
- **MultiResolutionTabularAttention**: A MultiResolutionTabularAttention layer instance


---

## numerical_embedding_layer

```python
numerical_embedding_layer(embedding_dim: int = 8, mlp_hidden_units: int = 16, num_bins: int = 10, init_min: float = -3.0, init_max: float = 3.0, dropout_rate: float = 0.1, use_batch_norm: bool = True, name: str = 'numerical_embedding', **kwargs: dict) -> keras.src.layers.layer.Layer
```

Create a NumericalEmbedding layer.

### Parameters- **embedding_dim (int)**: Dimension of the output embedding
    mlp_hidden_units (int): Number of hidden units in the MLP
    num_bins (int): Number of bins for discretization
    init_min (float): Minimum value for initialization
    init_max (float): Maximum value for initialization
    dropout_rate (float): Dropout rate for regularization
    use_batch_norm (bool): Whether to use batch normalization
    name (str): Name of the layer
    **kwargs: Additional arguments to pass to the layer

### Returns
- **NumericalEmbedding**: A NumericalEmbedding layer instance


---

## preserve_dtype_layer

```python
preserve_dtype_layer(name: str = 'preserve_dtype', target_dtype: tensorflow.python.framework.dtypes.DType | None = None, **kwargs: dict) -> keras.src.layers.layer.Layer
```

Create a PreserveDtypeLayer layer.

### Parameters- **name**: The name of the layer.
- **target_dtype**: Optional target dtype to cast to. If None, preserves original dtype.
    **kwargs: Additional keyword arguments to pass to the layer constructor.

### Returns

    An instance of the PreserveDtypeLayer layer.


---

## rolling_stats_layer

```python
rolling_stats_layer(window_size: int, name: str = 'rolling_stats', statistics: list[str] = None, window_stride: int = 1, pad_value: float = 0.0, **kwargs: dict) -> keras.src.layers.layer.Layer
```

Create a RollingStatsLayer for computing rolling statistics over a sliding window.

### Parameters- **window_size**: Size of the sliding window.
- **name**: Name of the layer.
- **statistics**: List of statistics to compute. Options: 'mean', 'std', 'min', 'max',
               'sum', 'median', 'range', 'variance'. Default is ['mean'].
- **window_stride**: Stride of the sliding window. Default is 1.
- **pad_value**: Value to use for padding. Default is 0.0.
    **kwargs: Additional keyword arguments.

### Returns

    RollingStatsLayer instance.


---

## tabular_attention_layer

```python
tabular_attention_layer(num_heads: int, d_model: int, name: str = 'tabular_attention', **kwargs: dict) -> keras.src.layers.layer.Layer
```

Create a TabularAttention layer.

### Parameters- **num_heads (int)**: Number of attention heads
    d_model (int): Dimensionality of the attention model
    name (str): Name of the layer
    **kwargs: Additional arguments to pass to the layer

### Returns
- **TabularAttention**: A TabularAttention layer instance


---

## text_preprocessing_layer

```python
text_preprocessing_layer(name: str = 'text_preprocessing', **kwargs: dict) -> keras.src.layers.layer.Layer
```

Create a TextPreprocessingLayer layer.

### Parameters- **name**: The name of the layer.
    **kwargs: Additional keyword arguments to pass to the layer constructor.

### Returns

    An instance of the TextPreprocessingLayer layer.


---

## transformer_block_layer

```python
transformer_block_layer(name: str = 'transformer', **kwargs: dict) -> keras.src.layers.layer.Layer
```

Create a TransformerBlock layer.

### Parameters- **name**: The name of the layer.
    **kwargs: Additional keyword arguments to pass to the layer constructor.

### Returns

    An instance of the TransformerBlock layer.


---

## variable_selection_layer

```python
variable_selection_layer(nr_features: int = None, units: int = 16, dropout_rate: float = 0.2, name: str = 'variable_selection', **kwargs: dict) -> keras.src.layers.layer.Layer
```

Create a VariableSelection layer.

### Parameters- **nr_features (int)**: Number of input features
    units (int): Dimensionality of the output space
    dropout_rate (float): Fraction of the input units to drop
    name (str): Name of the layer
    **kwargs: Additional arguments to pass to the layer

### Returns
- **VariableSelection**: A VariableSelection layer instance


---

