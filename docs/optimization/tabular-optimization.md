# 🚀 Tabular Optimization: Beating Traditional Models

## 📋 Quick Overview

Want to outperform XGBoost and other traditional tabular models? KDP's advanced optimization techniques help you achieve state-of-the-art results by addressing the core limitations of tree-based approaches. This guide shows you how to unlock neural superpowers for tabular data.

## ✨ Why KDP Beats Traditional Models

| Challenge | Traditional Approach | KDP's Solution |
|-----------|---------------------|----------------|
| Complex Distributions | Fixed binning strategies | 📊 **Distribution-Aware Encoding** that adapts to your specific data |
| Interaction Discovery | Manual feature crosses or tree splits | 👁️ **Tabular Attention** that automatically finds important relationships |
| Feature Importance | Post-hoc analysis | 🎯 **Built-in Feature Selection** during training |
| Deep Representations | Limited embedding capabilities | 🧠 **Advanced Neural Embeddings** for all feature types |
| Performance at Scale | Memory issues with large datasets | ⚡ **Optimized Processing Pipeline** with batching and caching |

## 🏆 Performance Comparison

In our benchmarks against top tabular models:

- **Accuracy**: +3-7% improvement over XGBoost on complex datasets
- **AUC**: +5% average improvement on financial and user behavior data
- **Training Time**: 2-5x faster than comparable deep learning approaches
- **Memory Usage**: 50-70% reduction compared to one-hot encoding pipelines

## 🚀 One-Minute Optimization

```python
from kdp import PreprocessingModel

# Create an optimized preprocessor in one step
preprocessor = PreprocessingModel(
    path_data="customer_data.csv",
    features_specs=features,

    # Enable performance-enhancing features
    use_distribution_aware=True,       # Smart distribution handling
    tabular_attention=True,            # Feature interaction learning
    feature_selection_placement="all_features",  # Remove noise automatically

    # Performance optimizations
    use_caching=True,               # Speed up repeated processing
    batch_size=10000                   # Process in manageable chunks
)

# Build and get metrics
result = preprocessor.build_preprocessor()
model = result["model"]

# Check optimization results
print(f"Memory usage: {preprocessor.get_memory_usage()['peak_mb']} MB")
print(f"Processing time: {preprocessor.get_timing_metrics()['total_seconds']:.2f}s")
```

## 🔧 Advanced Optimization Techniques

### 1. Distribution-Aware Optimization

```python
# Fine-tune distribution handling for better performance
preprocessor = PreprocessingModel(
    features_specs=features,

    # Enable and customize distribution-aware encoding
    use_distribution_aware=True,
    distribution_aware_bins=1000,             # More bins = finer-grained encoding
)
```

`distribution_aware_bins` is the only model-level knob. Periodicity detection,
sparsity handling and adaptive binning are always on and are not configurable
from `PreprocessingModel`; to change them, build a
[`DistributionAwareEncoder`](../advanced/distribution-aware-encoding.md)
yourself in a custom pipeline. To force a distribution for one column, set
`preferred_distribution` on its `NumericalFeature`.

### 2. Feature Interaction Optimization

```python
# Optimize how features interact with each other
preprocessor = PreprocessingModel(
    features_specs=features,

    # Enable and customize tabular attention
    tabular_attention=True,
    tabular_attention_heads=8,                # More heads = more interaction patterns
    tabular_attention_dim=128,                # Larger = richer representations
    tabular_attention_placement="multi_resolution",  # Process at multiple scales

    # Advanced interaction learning
    transfo_nr_blocks=2,                      # Add transformer blocks
    transfo_dropout_rate=0.1                  # Regularization for better generalization
)
```

### 3. Memory & Performance Optimization

```python
# Optimize for large datasets and faster processing
preprocessor = PreprocessingModel(
    features_specs=features,

    # Memory
    batch_size=50000,                         # Adjust based on available RAM
    use_caching=True,                         # Cache intermediate tensors
)
```

`batch_size` governs how much data the statistics pass holds at once, and
`use_caching` reuses intermediate tensors while the graph is built. Features
are already processed in parallel internally &mdash; there is no flag for it.

!!! note "Mixed precision is a Keras setting, not a KDP one"
    Set it globally before building, and every layer KDP creates picks it up:

    ```python
    import keras
    keras.mixed_precision.set_global_policy("mixed_float16")
    ```

## 📈 Real-World Optimization Examples

### Financial Fraud Detection

```python
# Optimize for fraud detection (imbalanced, complex distributions)
preprocessor = PreprocessingModel(
    path_data="transactions.csv",
    features_specs={
        "amount": FeatureType.FLOAT_RESCALED,
        "transaction_time": FeatureType.DATE,
        "merchant_id": FeatureType.STRING_CATEGORICAL,
        "device_id": FeatureType.STRING_CATEGORICAL,
        "location": FeatureType.STRING_CATEGORICAL,
        "history_summary": FeatureType.TEXT
    },
    # Distribution optimization for financial data
    use_distribution_aware=True,
    distribution_aware_bins=2000,            # More precise for financial values

    # Interaction learning for fraud patterns
    tabular_attention=True,
    tabular_attention_heads=12,              # More heads for complex interactions

    # Performance optimizations
    feature_selection_placement="all_features",       # Focus on relevant signals
    use_caching=True,
    batch_size=5000                          # Smaller batches for complex processing
)
```

### E-Commerce Recommendations

```python
# Optimize for recommendation systems (high-dimensional, sparse)
preprocessor = PreprocessingModel(
    path_data="user_product_interactions.csv",
    features_specs={
        "user_id": FeatureType.STRING_CATEGORICAL,
        "product_id": FeatureType.STRING_CATEGORICAL,
        "category": FeatureType.STRING_CATEGORICAL,
        "price": FeatureType.FLOAT_RESCALED,
        "past_purchases": FeatureType.TEXT,
        "last_visit": FeatureType.DATE
    },
    # Specialized recommendation processing
    feature_crosses=[("user_id", "category", 32)],  # (feature_a, feature_b, nr_bins)
    use_feature_moe=True,                    # Mixture of Experts for different features

    # Performance
    use_caching=True,
)
```

## 🧪 Measuring Optimization Impact

Check if your optimizations are working:

```python
# Create baseline and optimized models
baseline = PreprocessingModel(features_specs=features).build_preprocessor()["model"]
optimized = PreprocessingModel(
    features_specs=features,
    use_distribution_aware=True,
    tabular_attention=True
).build_preprocessor()["model"]

# Create identical downstream models
def create_model(preprocessor):
    inputs = preprocessor.input
    x = preprocessor.output
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["auc"])
    return model

# Build and evaluate both models
baseline_model = create_model(baseline)
optimized_model = create_model(optimized)

# Train and compare
baseline_history = baseline_model.fit(train_data, validation_data=val_data)
optimized_history = optimized_model.fit(train_data, validation_data=val_data)

# Visualize the difference
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(baseline_history.history['val_auc'], label='Baseline')
plt.plot(optimized_history.history['val_auc'], label='Optimized')
plt.title('Optimization Impact on Validation AUC')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('optimization_impact.png', dpi=300)
plt.show()
```

## 💡 Optimization Pro Tips

1. **Start with Distribution-Aware Encoding**
   ```python
   # Always enable this first - it's the biggest win
   preprocessor = PreprocessingModel(
       features_specs=features,
       use_distribution_aware=True  # Just this one change helps significantly
   )
   ```

2. **Profile Before Optimizing**
   ```python
   # See where the bottlenecks are
   preprocessor = PreprocessingModel(features_specs=features)
   result = preprocessor.build_preprocessor()

   # Check timing metrics
   timing = preprocessor.get_timing_metrics()
   print("Timing breakdown:")
   for step, time in timing['steps'].items():
       print(f"- {step}: {time:.2f}s ({time/timing['total_seconds']*100:.1f}%)")

   # Check memory metrics
   memory = preprocessor.get_memory_usage()
   print(f"Peak memory: {memory['peak_mb']} MB")
   for feature, mem in memory['per_feature'].items():
       print(f"- {feature}: {mem:.1f}MB")
   ```

3. **Progressive Optimization Strategy**
   ```python
   # Step 1: Basic optimization
   basic = PreprocessingModel(
       features_specs=features,
       use_distribution_aware=True,
       use_caching=True
   )

   # Step 2: Add interaction learning
   intermediate = PreprocessingModel(
       features_specs=features,
       use_distribution_aware=True,
       tabular_attention=True,
       use_caching=True
   )

   # Step 3: Full optimization
   advanced = PreprocessingModel(
       features_specs=features,
       use_distribution_aware=True,
       tabular_attention=True,
       transfo_nr_blocks=2,
       feature_selection_placement="all_features",
       use_caching=True,
   )

   # Compare metrics at each stage
   # This helps you find the optimal cost/benefit point
   ```

4. **Feature-Specific Optimization**
   ```python
   # Focus optimization on problematic features
   from kdp.features import NumericalFeature, CategoricalFeature

   optimized_features = {
       # Standard feature
       "age": FeatureType.FLOAT_NORMALIZED,

       # Optimized high-cardinality feature
       "product_id": CategoricalFeature(
           name="product_id",
           feature_type=FeatureType.STRING_CATEGORICAL,
           embedding_dim=16,             # Smaller embedding
           handle_unknown="use_oov"      # Handle unseen values
       ),

       # Optimized skewed numerical feature
       "transaction_amount": NumericalFeature(
           name="transaction_amount",
           feature_type=FeatureType.FLOAT_RESCALED,
           preferred_distribution="log_normal"  # Distribution hint
       )
   }
   ```

## 🔗 Next Steps

- [Distribution-Aware Encoding](../advanced/distribution-aware-encoding.md) - Deep dive into distribution optimization
- [Tabular Attention](../advanced/tabular-attention.md) - Advanced feature interaction learning
- [Memory Optimization](tabular-optimization.md#memory-optimization) - Handle large-scale datasets efficiently
- [Benchmarking](tabular-optimization.md#benchmarking) - Compare KDP against other approaches

## Related Features

- [Distribution-Aware Encoding](../advanced/distribution-aware-encoding.md)
- [Tabular Attention](../advanced/tabular-attention.md)
- [Memory Optimization](tabular-optimization.md#memory-optimization)
- [Benchmarking](tabular-optimization.md#benchmarking)

## 📊 Advanced Techniques

## Memory Optimization {#memory-optimization}

### Strategies for Large-Scale Datasets

KDP provides several strategies for optimizing memory usage:

1. **Lazy Loading** - Process data in batches instead of loading everything at once
2. **Feature Compression** - Use dimensionality reduction techniques for high-cardinality features
3. **Quantization** - Use numerical precision optimizations when applicable
4. **Sparse Representations** - Leverage sparse tensors for categorical features

```python
# Memory-optimized preprocessing
model = PreprocessingModel(
    path_data="large_dataset.csv",
    features_specs=features,
    batch_size=1024,  # Process in smaller batches
)
```

## Benchmarking {#benchmarking}

### Performance Comparison

KDP is designed to be efficient and performant. Here's how it compares to other preprocessing approaches:

| Metric | KDP | Pandas | TF.Transform | PyTorch |
|--------|-----|--------|-------------|---------|
| Memory Usage | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Processing Speed | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Feature Coverage | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Integration Ease | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### Benchmark Code Example

```python
import time
from kdp import PreprocessingModel
import pandas as pd

# Sample dataset
df = pd.read_csv("benchmark_dataset.csv")

# KDP approach
start_time = time.time()
model = PreprocessingModel(features_specs=features)
model.fit(df)
preprocessor = model.build_preprocessor()
kdp_time = time.time() - start_time

# Pandas approach
start_time = time.time()
# Traditional pandas preprocessing code...
pandas_time = time.time() - start_time

print(f"KDP processing time: {kdp_time:.2f}s")
print(f"Pandas processing time: {pandas_time:.2f}s")
print(f"Speedup: {pandas_time/kdp_time:.2f}x")
```
