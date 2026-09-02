# PreprocessingModel

## Constructor

```python
__init__(self, features_stats: dict[str, typing.Any] = None, path_data: str = None, batch_size: int = 50000, feature_crosses: list[tuple[str, str, int]] = None, features_stats_path: str = None, output_mode: str = 'concat', overwrite_stats: bool = False, log_to_file: bool = False, features_specs: dict[str, kdp.features.FeatureType | str] = None, transfo_nr_blocks: int = None, transfo_nr_heads: int = 3, transfo_ff_units: int = 16, transfo_dropout_rate: float = 0.25, transfo_placement: str = 'categorical', tabular_attention: bool = False, tabular_attention_heads: int = 4, tabular_attention_dim: int = 64, tabular_attention_dropout: float = 0.1, tabular_attention_placement: str = 'all_features', tabular_attention_embedding_dim: int = 32, use_caching: bool = True, feature_selection_placement: str = 'none', use_distribution_aware: bool = False, distribution_aware_bins: int = 1000, feature_selection_units: int = 32, feature_selection_dropout: float = 0.2, use_advanced_numerical_embedding: bool = False, embedding_dim: int = 8, mlp_hidden_units: int = 16, num_bins: int = 10, init_min: float = -3.0, init_max: float = 3.0, dropout_rate: float = 0.1, use_batch_norm: bool = True, use_global_numerical_embedding: bool = False, global_embedding_dim: int = 8, global_mlp_hidden_units: int = 16, global_num_bins: int = 10, global_init_min: float = -3.0, global_init_max: float = 3.0, global_dropout_rate: float = 0.1, global_use_batch_norm: bool = True, global_pooling: str = 'average', use_feature_moe: bool = False, feature_moe_num_experts: int = 4, feature_moe_expert_dim: int = 64, feature_moe_hidden_dims: list[int] = None, feature_moe_routing: str = 'learned', feature_moe_sparsity: int = 2, feature_moe_assignments: dict[str, int] = None, feature_moe_dropout: float = 0.1, feature_moe_freeze_experts: bool = False, feature_moe_use_residual: bool = True, include_passthrough_in_output: bool = True, name: str = 'preprocessor') -> None
```

Initialize a preprocessing model.

### Parameters- **features_stats (dict[str, Any])**: A dictionary containing the statistics of the features.
    path_data (str): The path to the data from which estimate the statistics.
    batch_size (int): The batch size for the data iteration for stats estimation.
    feature_crosses (list[tuple[str, str, int]]):
        A list of tuples containing the names of the features to be crossed,
        and nr_bins to be used for hashing.
    features_stats_path (str): The path where to save/load features statistics.
    output_mode (str): The output mode of the model (concat | dict).
    overwrite_stats (bool): A boolean indicating whether to overwrite the statistics.
    log_to_file (bool): A boolean indicating whether to log to a file.
    features_specs (dict[str, FeatureType | str]): A dictionary containing the features and their types.
    transfo_nr_blocks (int): The number of transformer blocks for the transformer block
        (default=None, transformer block is disabled).
    transfo_nr_heads (int): The number of heads for the transformer block (categorical variables).
    transfo_ff_units (int): The number of feed forward units for the transformer
    transfo_dropout_rate (float): The dropout rate for the transformer block (default=0.25).
    transfo_placement (str): The placement of the transformer block (categorical | all_features).
    tabular_attention (bool): Whether to use tabular attention (default=False).
    tabular_attention_heads (int): Number of attention heads for tabular attention.
    tabular_attention_dim (int): Dimension of the attention model.
    tabular_attention_dropout (float): Dropout rate for tabular attention.
    tabular_attention_placement (str): Where to apply tabular attention (none|numeric|categorical|all_features).
    tabular_attention_embedding_dim (int): Dimension of the embedding for multi-resolution attention.
    use_caching (bool): Whether to cache preprocessed features (default=True).
    feature_selection_placement (str): Where to apply feature selection (none|numeric|categorical|all_features).
    feature_selection_units (int): Number of units for feature selection.
    feature_selection_dropout (float): Dropout rate for feature selection.
    use_distribution_aware (bool): Whether to use distribution-aware encoding for features.
    distribution_aware_bins (int): Number of bins to use for distribution-aware encoding.
    use_advanced_numerical_embedding (bool): Whether to use advanced numerical embedding.
    embedding_dim (int): Dimension of the embedding for advanced numerical embedding.
    mlp_hidden_units (int): Number of units for the MLP in advanced numerical embedding.
    num_bins (int): Number of bins for discretization in advanced numerical embedding.
    init_min (float): Minimum value for the embedding in advanced numerical embedding.
    init_max (float): Maximum value for the embedding in advanced numerical embedding.
    dropout_rate (float): Dropout rate for advanced numerical embedding.
    use_batch_norm (bool): Whether advanced numerical embedding applies batch normalization.
    use_global_numerical_embedding (bool): Whether to embed all numeric features jointly,
        in addition to (or instead of) per-feature embeddings.
    global_embedding_dim (int): Dimension of the global numerical embedding.
    global_mlp_hidden_units (int): Number of units for the MLP in the global numerical embedding.
    global_num_bins (int): Number of bins for discretization in the global numerical embedding.
    global_init_min (float): Minimum value for the global numerical embedding.
    global_init_max (float): Maximum value for the global numerical embedding.
    global_dropout_rate (float): Dropout rate for the global numerical embedding.
    global_use_batch_norm (bool): Whether the global numerical embedding applies batch normalization.
    global_pooling (str): Pooling applied over the global embedding (average | max).
    use_feature_moe (bool): Whether to enable the feature-wise mixture of experts.
    feature_moe_num_experts (int): Number of experts in the mixture.
    feature_moe_expert_dim (int): Output dimension of each expert.
    feature_moe_hidden_dims (list[int]): Hidden layer sizes inside each expert.
    feature_moe_routing (str): How features are routed to experts (learned | predefined).
    feature_moe_sparsity (int): Number of experts each feature is routed to when routing is sparse.
    feature_moe_assignments (dict[str, int]): Explicit feature-to-expert assignments,
        used when routing is "predefined".
    feature_moe_dropout (float): Dropout rate applied inside the mixture of experts.
    feature_moe_freeze_experts (bool): Whether expert weights are frozen during training.
    feature_moe_use_residual (bool): Whether to add a residual connection around the mixture.
    name (str): Name given to the built Keras model. Keras requires
        operation names to be unique within a graph, so give each
        preprocessor its own name when combining several in one model
        (a two-tower recommender, for example).
    include_passthrough_in_output (bool): Whether passthrough features appear in the model output.


---

## batch_predict

```python
batch_predict(self, dataset: tensorflow.python.data.ops.dataset_ops.DatasetV2) -> collections.abc.Generator
```

Process batches of data through the model.

### Parameters- **dataset**: TensorFlow dataset containing batches of input data
- **Yields**: Preprocessed batches

### Raises
- **ValueError**: If the model hasn't been built yet


---

## build_preprocessor

```python
build_preprocessor(self) -> dict
```

Building preprocessing model.

### Returns- **dict**: Dictionary containing:
        - model: The preprocessing model
        - inputs: Model inputs
        - signature: Model signature
        - output_dims: Output dimensions
        - feature_stats: Feature statistics

### Raises
- **ValueError**: If no features are specified or if required stats are missing


---

## get_feature_importances

```python
get_feature_importances(self, data: dict | None = None) -> dict
```

Get feature importance weights if feature selection was enabled.

The selection layer computes a softmax over features for every row, so
the importances depend on the data rather than being fixed weights.
Pass a batch to get numbers; without one there is nothing to score and
only a description of each weight tensor can be returned.

### Parameters- **data**: Optional mapping of feature name to a batch of values. When
        given, the model is run and the mean importance per feature is
        returned as a float.

### Returns
- **dict**: Feature name to mean importance (a float) when `data` is
        supplied, otherwise feature name to a description of its weight
        tensor.

### Raises
- **ValueError**: If feature selection was not enabled or model hasn't been built


---

## load_model

```python
load_model(load_path: str) -> tuple
```

Load a saved preprocessing model and its metadata.

### Parameters- **load_path**: Directory path where the model and metadata are saved

### Returns
- **tuple**: (loaded_model, metadata)

### Raises
- **ValueError**: If the model directory doesn't exist or is missing required files


---

## predict

```python
predict(self, data, **kwargs) -> Any
```

Predict using the preprocessor model.

### Parameters- **data**: The data to predict on, can be pandas DataFrame, dict, or TensorFlow dataset.
    **kwargs: Additional keyword arguments to pass to the model's predict method.

### Returns

    The prediction output.


---

## save_model

```python
save_model(self, save_path: str) -> None
```

Save the preprocessing model and its metadata.

This method saves both the TensorFlow model and additional metadata
needed to fully reconstruct the preprocessing pipeline.

### Parameters- **save_path**: Directory path where to save the model and metadata

### Raises
- **ValueError**: If the model hasn't been built yet


---

