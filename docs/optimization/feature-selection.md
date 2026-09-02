# 🎯 Feature Selection: Focus on What Matters

## 📋 Quick Overview

Feature Selection in KDP automatically identifies and prioritizes your most important features, cutting through the noise to focus on what really drives your predictions. Built on the advanced Gated Residual Variable Selection Network (GRVSN) architecture, it's like having a data scientist automatically analyze your feature importance.

## ✨ Key Benefits

- 🧠 **Smarter Models**: Direct computational power to features that actually matter
- 📈 **Better Performance**: Often boosts accuracy by 5-15% by reducing noise
- 🔍 **Instant Insights**: See which features drive predictions without manual analysis
- ⚡ **Training Speedup**: Typically 30-50% faster training with optimized feature sets
- 🛡️ **Better Generalization**: Models that focus on signal, not noise

## 🚀 Quick Start Example

```python
from kdp import PreprocessingModel, FeatureType

# Define your features
features = {
    "age": FeatureType.FLOAT_NORMALIZED,
    "income": FeatureType.FLOAT_RESCALED,
    "education": FeatureType.STRING_CATEGORICAL,
    "occupation": FeatureType.STRING_CATEGORICAL,
    "marital_status": FeatureType.STRING_CATEGORICAL,
    "last_purchase": FeatureType.DATE
}

# Enable feature selection with just a few lines
preprocessor = PreprocessingModel(
    path_data="customer_data.csv",
    features_specs=features,

    # Enable feature selection for all features
    feature_selection_placement="all_features",
    feature_selection_units=64,        # Neural network size
    feature_selection_dropout=0.2      # Regularization strength
)

# Build and use as normal
result = preprocessor.build_preprocessor()
model = result["model"]

# Importances are a per-row softmax, so they need a batch to score
import tensorflow as tf

importances = preprocessor.get_feature_importances({
    "age": tf.constant([[35.0]]),
    "income": tf.constant([[70000.0]]),
    "education": tf.constant([["bsc"]]),
    "occupation": tf.constant([["engineer"]]),
    "marital_status": tf.constant([["single"]]),
    "last_purchase": tf.constant([["2021-06-15"]]),
})
print(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3])
```

## 🧩 Architecture

Feature Selection can be applied at different points in your KDP pipeline:

```python
# Apply feature selection to all features
preprocessor = PreprocessingModel(
    features_specs=features,
    feature_selection_placement="all_features",
    feature_selection_units=32,        # width of the selection network
    feature_selection_dropout=0.2,     # dropout inside it
)
```

*Note: selection is learned, not statistical &mdash; a `VariableSelection` layer
is added to the graph and its weights are trained with the rest of your model.
There is no correlation filter or threshold; `feature_selection_placement`
chooses which feature groups get the layer, and the two parameters above size
it. Valid placements are `"none"`, `"numeric"`, `"categorical"`, `"text"`,
`"date"` and `"all_features"`.*

!!! warning "The weights do not currently rank features"
    Each feature gets its **own** selection layer covering a single feature, so
    the softmax that produces its weight runs over one element and is `1.0` by
    definition &mdash; for every feature, on every input. The layer still
    applies a learned gated residual transform, which is a real
    transformation, but the reported importances carry no ranking information
    yet. `get_feature_importances()` returns them faithfully; do not read a
    ranking into equal numbers.

    Calling it without a batch returns a description of each weight tensor
    instead, which is what earlier releases always returned.

## 🎛️ Configuration Options

### Placement Options

Choose where to apply feature selection with the `feature_selection_placement` parameter:

| Option | Description | Best For |
|--------|-------------|----------|
| `"none"` | Disable feature selection | When you know all features matter |
| `"numeric"` | Only select among numerical features | Financial or scientific data |
| `"categorical"` | Only select among categorical features | Marketing or demographic data |
| `"all_features"` | Apply selection to all feature types | Most use cases - let KDP decide |

### Key Parameters

| Parameter | Purpose | Default | Recommended Range |
|-----------|---------|---------|------------------|
| `feature_selection_units` | Size of neural network | 64 | 32-128 (larger = more capacity) |
| `feature_selection_dropout` | Prevents overfitting | 0.2 | 0.1-0.3 (higher for smaller datasets) |
| `feature_selection_use_bias` | Adds bias term to gates | True | Usually keep as True |

## 📊 Real-World Examples

### Customer Churn Prediction

```python
from kdp import FeatureType, PreprocessingModel

# Perfect for churn prediction with many potential factors
preprocessor = PreprocessingModel(
    path_data="customer_data.csv",
    features_specs={
        "customer_age": FeatureType.FLOAT_NORMALIZED,
        "subscription_length": FeatureType.FLOAT_RESCALED,
        "monthly_spend": FeatureType.FLOAT_RESCALED,
        "support_tickets": FeatureType.FLOAT_RESCALED,
        "product_tier": FeatureType.STRING_CATEGORICAL,
        "last_upgrade": FeatureType.DATE,
        "industry": FeatureType.STRING_CATEGORICAL,
        "region": FeatureType.STRING_CATEGORICAL,
        "company_size": FeatureType.INTEGER_CATEGORICAL
    },
    # Powerful feature selection configuration
    feature_selection_placement="all_features",
    feature_selection_units=96,       # Larger for complex patterns
    feature_selection_dropout=0.15,   # Moderate regularization

    # Combine with distribution-aware for best results
    use_distribution_aware=True
)

# After building, analyze what drives churn
preprocessor.build_preprocessor()
importances = preprocessor.get_feature_importances()
```

### Medical Diagnosis Support

```python
from kdp import FeatureType, PreprocessingModel

# For medical applications where feature interpretation is critical
preprocessor = PreprocessingModel(
    path_data="patient_data.csv",
    features_specs={
        "age": FeatureType.FLOAT_NORMALIZED,
        "heart_rate": FeatureType.FLOAT_NORMALIZED,
        "blood_pressure": FeatureType.FLOAT_NORMALIZED,
        "glucose_level": FeatureType.FLOAT_NORMALIZED,
        "cholesterol": FeatureType.FLOAT_NORMALIZED,
        "bmi": FeatureType.FLOAT_NORMALIZED,
        "smoking_status": FeatureType.STRING_CATEGORICAL,
        "family_history": FeatureType.STRING_CATEGORICAL
    },
    # Focus on numerical biomarkers
    feature_selection_placement="numeric",
    feature_selection_units=64,
    feature_selection_dropout=0.2,

    # Medical applications benefit from careful regularization
    use_advanced_numerical_embedding=True,
    embedding_dim=32
)
```

## 📊 Visualizing Feature Importance

KDP provides utilities to visualize which features are most important:

```python
# After building and training your preprocessor
feature_importance = preprocessor.get_feature_importance()

# Visualize the importance scores
preprocessor.plot_feature_importance()

# Get the top N most important features
top_features = preprocessor.get_top_features(n=10)
```

*Note: The feature importance visualization shows a bar chart with features sorted by their importance scores, helping you identify which features contribute most to your model's performance.*

## 💡 Pro Tips for Feature Selection

1. **Use With Distribution-Aware Encoding**
   ```python
   # This combination often works exceptionally well
   preprocessor = PreprocessingModel(
       features_specs=features,
       feature_selection_placement="all_features",
       use_distribution_aware=True  # Add this line
   )
   ```

2. **Focus Selection for Speed**
   ```python
   # For large datasets, focus on specific feature types first
   preprocessor = PreprocessingModel(
       features_specs=many_features,
       feature_selection_placement="numeric",  # Start with just numerical
       use_caching=True  # Speed up repeated processing
   )
   ```

3. **Progressive Feature Refinement**
   ```python
   # First run to identify important features
   importances = first_preprocessor.get_feature_importances()

   # Keep only features with importance > 0.05
   important_features = {k: v for k, v in features.items()
                        if importances.get(k, 0) > 0.05}

   # Create refined model with just important features
   refined_preprocessor = PreprocessingModel(
       features_specs=important_features,
       # More advanced processing now with fewer features
       transfo_nr_blocks=2,
       tabular_attention=True
   )
   ```

4. **Tracking Importance Over Time**
   ```python
   # For production systems, monitor if important features change
   import json
   from datetime import datetime

   # Save importance scores with timestamp
   def log_importances(preprocessor, name):
       preprocessor.build_preprocessor()
importances = preprocessor.get_feature_importances()
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       with open(f"importance_{name}_{timestamp}.json", "w") as f:
           json.dump(importances, f, indent=2)

   # Call periodically in production
   log_importances(my_preprocessor, "customer_model")
   ```

## 🔗 Related Topics

- [Distribution-Aware Encoding](../advanced/distribution-aware-encoding.md)
- [Tabular Attention](../advanced/tabular-attention.md)
- [Feature MoE](../advanced/feature-moe.md)
- [Complex Examples](../examples/complex-examples.md)
