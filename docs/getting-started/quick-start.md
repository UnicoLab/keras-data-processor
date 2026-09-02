# 🚀 Quick Start Guide: KDP in 5 Minutes

<div class="intro-container">
  <div class="intro-content">
    <h2>Get your tabular data ML-ready in record time!</h2>
    <p>This guide will have you transforming raw data into powerful features before your coffee gets cold.</p>
  </div>
</div>

## 🏁 The KDP Experience in 3 Steps

<div class="steps-container">
  <div class="step-card">
    <div class="step-header">
      <span class="step-number">1</span>
      <h3>Define Your Features</h3>
    </div>
    <div class="code-container">

```python
from kdp import PreprocessingModel, FeatureType

# Quick feature definition - KDP handles the complexity
features = {
    # Numerical features with smart preprocessing
    "age": FeatureType.FLOAT_NORMALIZED,          # zero mean, unit variance
    "income": FeatureType.FLOAT_NORMALIZED,       # standardised too

    # Categorical features with automatic encoding
    "occupation": FeatureType.STRING_CATEGORICAL, # Text categories to embeddings
    "education": FeatureType.INTEGER_CATEGORICAL, # Numeric categories

    # Special types get special treatment
    "feedback": FeatureType.TEXT,                 # Text gets tokenization & embedding
    "signup_date": FeatureType.DATE               # Dates become useful components
}
```

    </div>
  </div>

  <div class="step-card">
    <div class="step-header">
      <span class="step-number">2</span>
      <h3>Build Your Processor</h3>
    </div>
    <div class="code-container">

```python
# Create with smart defaults - one line setup
preprocessor = PreprocessingModel(
    path_data="customer_data.csv",     # Point to your data
    features_specs=features,           # Your feature definitions
    use_distribution_aware=True        # Automatic distribution handling
)

# Build analyzes your data and creates the preprocessing pipeline
result = preprocessor.build_preprocessor()
model = result["model"]                # This is your transformer!
```

    </div>
  </div>

  <div class="step-card">
    <div class="step-header">
      <span class="step-number">3</span>
      <h3>Process Your Data</h3>
    </div>
    <div class="code-container">

```python
# Your data can be a dict, DataFrame, or tensors
new_customer_data = {
    "age": [24, 67, 31],
    "income": [48000, 125000, 52000],
    "occupation": ["developer", "manager", "designer"],
    "education": [4, 5, 3],
    "feedback": ["Great product!", "Could be better", "Love it"],
    "signup_date": ["2023-06-15", "2022-03-22", "2023-10-01"]
}

# Transform into ML-ready features with a single call
processed_features = model(new_customer_data)

# That's it! Your data is now ready for modeling
```

    </div>
  </div>
</div>

## 🔥 Power Features

<div class="feature-showcase">
  <div class="feature-header">
    <h3>Take your preprocessing to the next level with these one-liners:</h3>
  </div>
  <div class="code-container">

```python
# Create a more advanced preprocessor
preprocessor = PreprocessingModel(
    path_data="customer_data.csv",
    features_specs=features,

    # Power features - each adds capability
    use_distribution_aware=True,        # Smart distribution handling
    use_advanced_numerical_embedding=True,       # Neural embeddings for numbers
    tabular_attention=True,                      # Learn feature relationships
    feature_selection_placement="all_features",  # Automatic feature importance

    # Add transformers for state-of-the-art performance
    transfo_nr_blocks=2,                # Two transformer blocks
    transfo_nr_heads=4                  # With four attention heads
)
```

  </div>
</div>

## 💼 Real-World Examples

<div class="examples-container">
  <div class="example-card">
    <div class="example-header">
      <span class="example-icon">👥</span>
      <h3>Customer Churn Prediction</h3>
    </div>
    <div class="code-container">

```python
from kdp import FeatureType, PreprocessingModel

# Perfect setup for churn prediction
preprocessor = PreprocessingModel(
    path_data="customer_data.csv",
    features_specs={
        "days_active": FeatureType.FLOAT_NORMALIZED,
        "monthly_spend": FeatureType.FLOAT_RESCALED,
        "total_purchases": FeatureType.FLOAT_RESCALED,
        "product_category": FeatureType.STRING_CATEGORICAL,
        "last_support_ticket": FeatureType.DATE,
        "support_messages": FeatureType.TEXT
    },
    use_distribution_aware=True,
    feature_selection_placement="all_features",  # Identify churn drivers
    tabular_attention=True                # Model feature interactions
)
```

    </div>
  </div>

  <div class="example-card">
    <div class="example-header">
      <span class="example-icon">📈</span>
      <h3>Financial Time Series</h3>
    </div>
    <div class="code-container">

```python
from kdp import FeatureType, PreprocessingModel

# Setup for financial forecasting
preprocessor = PreprocessingModel(
    path_data="stock_data.csv",
    features_specs={
        "open": FeatureType.FLOAT_RESCALED,
        "high": FeatureType.FLOAT_RESCALED,
        "low": FeatureType.FLOAT_RESCALED,
        "volume": FeatureType.FLOAT_RESCALED,
        "sector": FeatureType.STRING_CATEGORICAL,
        "date": FeatureType.DATE
    },
    use_advanced_numerical_embedding=True,  # Neural embeddings for price data
    embedding_dim=32,                       # Larger embeddings for complex patterns
    tabular_attention_heads=4            # Multiple attention heads
)
```

    </div>
  </div>
</div>

## 📱 Production Integration

<div class="integration-container">
  <div class="code-container">

```python
import tensorflow as tf

# Save your preprocessor after building. This writes model.keras and
# metadata.json into the directory you name.
preprocessor.build_preprocessor()
preprocessor.save_model("customer_churn_preprocessor")

# --- Later in production ---

from kdp import PreprocessingModel

# load_model returns the Keras model AND the metadata it was saved with
loaded_model, metadata = PreprocessingModel.load_model("customer_churn_preprocessor")

# metadata carries features_specs, features_stats, output_mode and use_feature_moe
print(metadata["output_mode"])

# Process new data. Every feature is a column, so each value is a batch of rows.
new_customer = {
    "age": tf.constant([[35.0]]),
    "income": tf.constant([[75000.0]]),
    "city": tf.constant([["paris"]]),
}
features = loaded_model(new_customer)

# Use with your prediction model
prediction = my_model(features)
```

  </div>
</div>

## 💡 Pro Tips

<div class="tips-container">
  <div class="tip-card">
    <div class="tip-header">
      <span class="tip-number">1</span>
      <h3>Start Simple First</h3>
    </div>
    <div class="code-container">

```python
# Begin with basic configuration
basic = PreprocessingModel(features_specs=features)

# Then add advanced features as needed
advanced = PreprocessingModel(
    features_specs=features,
    use_distribution_aware=True,
    tabular_attention=True
)
```

    </div>
  </div>

  <div class="tip-card">
    <div class="tip-header">
      <span class="tip-number">2</span>
      <h3>Handle Big Data Efficiently</h3>
    </div>
    <div class="code-container">

```python
# For large datasets
preprocessor = PreprocessingModel(
    features_specs=features,
    use_caching=True,           # Speed up repeated processing
    batch_size=10000            # Process in manageable chunks
)
```

    </div>
  </div>

  <div class="tip-card">
    <div class="tip-header">
      <span class="tip-number">3</span>
      <h3>Get Feature Importance</h3>
    </div>
    <div class="code-container">

```python
# First enable feature selection when creating the model
preprocessor = PreprocessingModel(
    features_specs=features,
    feature_selection_placement="all_features",  # Required for feature importance
    feature_selection_units=32
)

# Build the preprocessor
preprocessor.build_preprocessor()

# After building, you can get feature importances
importances = preprocessor.get_feature_importances()
print("Most important features:", sorted(
    importances.items(), key=lambda x: x[1], reverse=True
)[:3])
```

    </div>
  </div>
</div>

## 🔗 Where to Next?

<div class="next-steps-container">
  <a href="../features/overview.md" class="next-step-card">
    <span class="next-step-icon">🔍</span>
    <div class="next-step-content">
      <h3>Feature Processing Guide</h3>
      <p>Deep dive into feature types</p>
    </div>
  </a>
  <a href="../advanced/distribution-aware-encoding.md" class="next-step-card">
    <span class="next-step-icon">📊</span>
    <div class="next-step-content">
      <h3>Distribution-Aware Encoding</h3>
      <p>Smart numerical handling</p>
    </div>
  </a>
  <a href="../advanced/numerical-embeddings.md" class="next-step-card">
    <span class="next-step-icon">🧠</span>
    <div class="next-step-content">
      <h3>Advanced Numerical Embeddings</h3>
      <p>Neural representations</p>
    </div>
  </a>
  <a href="../advanced/tabular-attention.md" class="next-step-card">
    <span class="next-step-icon">👁️</span>
    <div class="next-step-content">
      <h3>Tabular Attention</h3>
      <p>Model feature relationships</p>
    </div>
  </a>
  <a href="../examples/complex-examples.md" class="next-step-card">
    <span class="next-step-icon">🛠️</span>
    <div class="next-step-content">
      <h3>Complex Examples</h3>
      <p>Complete real-world scenarios</p>
    </div>
  </a>
</div>

---

<div class="nav-container">
  <a href="installation.md" class="nav-button prev">
    <span class="nav-icon">←</span>
    <span class="nav-text">Installation</span>
  </a>
  <a href="architecture.md" class="nav-button next">
    <span class="nav-text">Architecture Overview</span>
    <span class="nav-icon">→</span>
  </a>
</div>

<style>
/* Base styling */
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.6;
  color: #333;
  margin: 0;
  padding: 0;
}

/* Intro section */
.intro-container {
  background: linear-gradient(135deg, #f0f7ff 0%, #e9ecef 100%);
  border-radius: 10px;
  padding: 30px;
  margin: 30px 0;
  box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

.intro-content h2 {
  margin-top: 0;
  color: #4a86e8;
}

/* Step cards */
.steps-container {
  display: flex;
  flex-direction: column;
  gap: 25px;
  margin: 30px 0;
}

.step-card {
  background-color: #fff;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.step-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.step-header {
  display: flex;
  align-items: center;
  padding: 15px 20px;
  background: linear-gradient(135deg, #f0f7ff 0%, #e9ecef 100%);
  border-bottom: 1px solid #e9ecef;
}

.step-number {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  background-color: #4a86e8;
  color: white;
  border-radius: 50%;
  margin-right: 15px;
  font-weight: bold;
}

.step-header h3 {
  margin: 0;
  color: #333;
}

/* Code containers */
.code-container {
  padding: 0;
  background-color: #f8f9fa;
  border-radius: 0 0 8px 8px;
  overflow: hidden;
}

.code-container pre {
  margin: 0;
  padding: 20px;
}

/* Feature showcase */
.feature-showcase {
  background-color: #fff;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  margin: 30px 0;
}

.feature-header {
  padding: 15px 20px;
  background: linear-gradient(135deg, #f0f7ff 0%, #e9ecef 100%);
  border-bottom: 1px solid #e9ecef;
}

.feature-header h3 {
  margin: 0;
  color: #333;
}

/* Example cards */
.examples-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin: 30px 0;
}

.example-card {
  background-color: #fff;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.example-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.example-header {
  display: flex;
  align-items: center;
  padding: 15px 20px;
  background: linear-gradient(135deg, #f0f7ff 0%, #e9ecef 100%);
  border-bottom: 1px solid #e9ecef;
}

.example-icon {
  font-size: 1.5em;
  margin-right: 15px;
}

.example-header h3 {
  margin: 0;
  color: #333;
}

/* Integration section */
.integration-container {
  background-color: #fff;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  margin: 30px 0;
}

/* Pro tips */
.tips-container {
  display: grid;
  grid-template-columns: 1fr;
  gap: 20px;
  margin: 30px 0;
}

.tip-card {
  background-color: #fff;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.tip-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.tip-header {
  display: flex;
  align-items: center;
  padding: 15px 20px;
  background: linear-gradient(135deg, #f0f7ff 0%, #e9ecef 100%);
  border-bottom: 1px solid #e9ecef;
}

.tip-number {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  background-color: #4CAF50;
  color: white;
  border-radius: 50%;
  margin-right: 15px;
  font-weight: bold;
}

.tip-header h3 {
  margin: 0;
  color: #333;
}

/* Next steps */
.next-steps-container {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.next-step-card {
  display: flex;
  align-items: center;
  background-color: #fff;
  border-radius: 10px;
  padding: 15px 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  text-decoration: none;
  color: #333;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.next-step-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
  background-color: #f0f7ff;
}

.next-step-icon {
  font-size: 1.5em;
  margin-right: 15px;
}

.next-step-content h3 {
  margin: 0 0 5px 0;
  color: #4a86e8;
}

.next-step-content p {
  margin: 0;
  font-size: 14px;
  color: #555;
}

/* Navigation */
.nav-container {
  display: flex;
  justify-content: space-between;
  margin: 40px 0;
}

.nav-button {
  display: flex;
  align-items: center;
  padding: 10px 15px;
  background-color: #f8f9fa;
  border-radius: 8px;
  text-decoration: none;
  color: #333;
  box-shadow: 0 2px 5px rgba(0,0,0,0.1);
  transition: background-color 0.3s ease, transform 0.3s ease;
}

.nav-button:hover {
  background-color: #f0f7ff;
  transform: translateY(-2px);
}

.nav-button.prev {
  padding-left: 10px;
}

.nav-button.next {
  padding-right: 10px;
}

.nav-icon {
  font-size: 1.2em;
  margin: 0 8px;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .examples-container {
    grid-template-columns: 1fr;
  }

  .next-steps-container {
    grid-template-columns: 1fr;
  }
}
</style>
