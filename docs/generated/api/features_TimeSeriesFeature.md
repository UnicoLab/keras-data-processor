# TimeSeriesFeature

TimeSeriesFeature with support for lag features and temporal processing.

## Constructor

```python
__init__(self, name: str, feature_type: kdp.features.FeatureType = <FeatureType.TIME_SERIES: 10>, lag_config: dict = None, rolling_stats_config: dict = None, differencing_config: dict = None, moving_average_config: dict = None, wavelet_transform_config: dict = None, tsfresh_feature_config: dict = None, calendar_feature_config: dict = None, sequence_length: int = None, sort_by: str = None, sort_ascending: bool = True, group_by: str = None, dtype: tensorflow.python.framework.dtypes.DType = tf.float32, is_target: bool = False, exclude_from_input: bool = False, input_type: str = 'continuous', **kwargs) -> None
```

Initializes a TimeSeriesFeature instance.

### Parameters- **name (str)**: The name of the feature.
    feature_type (FeatureType): The type of the feature.
    lag_config (dict): Configuration for lag features. If None, no lag features will be created.
- **Example**: {'lags': [1, 7, 14], 'drop_na': True}
    rolling_stats_config (dict): Configuration for rolling statistics.
- **Example**: {'window_size': 7, 'statistics': ['mean', 'std']}
    differencing_config (dict): Configuration for differencing.
- **Example**: {'order': 1}
    moving_average_config (dict): Configuration for moving averages.
- **Example**: {'periods': [7, 14, 30]}
    wavelet_transform_config (dict): Configuration for wavelet transform.
- **Example**: {'levels': 3, 'window_sizes': [4, 8, 16], 'flatten_output': True}
    tsfresh_feature_config (dict): Configuration for statistical feature extraction.
- **Example**: {'features': ['mean', 'std', 'min', 'max'], 'normalize': True}
    calendar_feature_config (dict): Configuration for calendar features.
- **Example**: {'features': ['month', 'day', 'day_of_week'], 'cyclic_encoding': True}
    sequence_length (int): Length of the sequence. If None, no sequence will be created.
    sort_by (str): Column name to sort the time series data by (typically a timestamp column).
        Required for proper time series ordering.
    sort_ascending (bool): Whether to sort in ascending order (True) or descending order (False).
        Default is True for chronological ordering.
    group_by (str): Optional column name to group time series data by. Useful for multiple series
        (e.g., data for different stores, customers, products, etc.)
    dtype (tf.DType): The data type of the feature.
    is_target (bool): Whether this feature is a target for prediction.
    exclude_from_input (bool): Whether to exclude this feature from the input.
    input_type (str): The input type of the feature (e.g., "continuous").
    **kwargs: Additional keyword arguments for the feature.


---

## add_preprocessor

```python
add_preprocessor(self, preprocessor: kdp.layers_factory.PreprocessorLayerFactory | typing.Any) -> None
```

Adds a preprocessor to the feature.

### Parameters- **preprocessor (Union[PreprocessorLayerFactory, Any])**: The preprocessor to add.


---

## build_layers

```python
build_layers(self, row_preserving: bool = True) -> list
```

Build the appropriate layers for this time series feature based on configuration.

### Parameters- **row_preserving**: When True (the default) the layers keep every input
        row, padding the warm-up positions instead of dropping them.
        A preprocessing model lays features out side by side, so a layer
        that removes its feature's leading rows leaves that column
        shorter than every other one and the concatenation fails. Pass
        False only when driving the returned layers directly.

### Returns
- **list**: List of TensorFlow layers for time series preprocessing


---

## from_string

```python
from_string(type_str: str) -> 'FeatureType'
```

Converts a string to a FeatureType.

### Parameters- **type_str (str)**: The string representation of the feature type.


---

## get_output_dim

```python
get_output_dim(self) -> int
```

Calculate the output dimension of this feature after all transformations.

The layers built- **by**: meth:`build_layers` are applied in sequence, and each
one that keeps its originals passes them through alongside its new
columns. The widths therefore compose multiplicatively, not additively:
two lags on a single column give 3 columns, and differencing that while
keeping the originals gives 6, not 4.

### Returns
- **int**: The number of columns the layer stack produces.


---

## to_dict

```python
to_dict(self) -> dict
```

Convert the feature configuration to a dictionary.

### Returns- **dict**: Dictionary representation of the feature


---

## update_kwargs

```python
update_kwargs(self, **kwargs) -> None
```

Updates the kwargs with new or modified parameters.

### Parameters

    **kwargs: The new or modified parameters.


---
