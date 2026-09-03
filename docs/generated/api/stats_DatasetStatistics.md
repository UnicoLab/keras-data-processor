# DatasetStatistics

## Constructor

```python
__init__(self, path_data: str, features_specs: dict[str, kdp.features.FeatureType | str] = None, numeric_features: list[kdp.features.NumericalFeature] = None, categorical_features: list[kdp.features.CategoricalFeature] = None, text_features: list[kdp.features.CategoricalFeature] = None, date_features: list[str] = None, time_series_features: list[kdp.features.TimeSeriesFeature] = None, features_stats_path: pathlib.Path = None, overwrite_stats: bool = False, batch_size: int = 50000) -> None
```

Initializes the statistics accumulators for numeric, categorical, text, and date features.

### Parameters- **path_data**: Path to the folder containing the CSV files.
- **batch_size**: The batch size to use when reading data from the dataset.
- **features_stats_path**: Path to the features statistics JSON file (defaults to None).
- **overwrite_stats**: Whether or not to overwrite existing statistics file (defaults to False).
- **features_specs**: A dictionary mapping feature names to feature specifications (defaults to None).
        Easier alternative to providing numerical and categorical lists.
- **numeric_features**: A list of numerical features to calculate statistics for (defaults to None).
- **categorical_features**: A list of categorical features to calculate statistics for (defaults to None).
- **text_features**: A list of text features to calculate statistics for (defaults to None).
- **date_features**: A list of date features to calculate statistics for (defaults to None).
- **time_series_features**: A list of time series features to calculate statistics for (defaults to None).


---

## calculate_dataset_statistics

```python
calculate_dataset_statistics(self, dataset: tensorflow.python.data.ops.dataset_ops.DatasetV2) -> dict[str, dict]
```

Calculate the statistics of the dataset.

### Parameters- **dataset**: The dataset to calculate statistics for.

### Returns

    Dictionary containing the computed statistics


---

## main

```python
main(self) -> dict
```

Calculates and returns final statistics for the dataset.

### Returns

    A dictionary containing the calculated statistics for the dataset.


---

## recommend_model_configuration

```python
recommend_model_configuration(self) -> dict
```

Analyze the computed dataset statistics and provide recommendations for optimal preprocessing.

This method leverages the ModelAdvisor to analyze feature characteristics and suggest
the best preprocessing strategies, layer configurations, and model parameters.

### Returns- **dict**: A dictionary containing feature-specific and global recommendations
         along with a ready-to-use code snippet.


---
