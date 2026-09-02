# 🔢 Numerical Features

<div class="feature-header-container">
  <div class="feature-header-content">
    <h2>Transform your continuous data like age, income, or prices into powerful feature representations</h2>
  </div>
</div>

## 📋 Quick Overview

<div class="overview-card">
  <p>Numerical features are the backbone of most machine learning models. KDP provides multiple ways to handle them, from simple normalization to advanced neural embeddings.</p>
</div>

## 🎯 Types and Use Cases

<div class="table-container">
  <table class="feature-table">
    <thead>
      <tr>
        <th>Feature Type</th>
        <th>Best For</th>
        <th>Example Values</th>
        <th>When to Use</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>FLOAT_NORMALIZED</code></td>
        <td>Data with clear bounds</td>
        <td>🧓 Age: 18-65, ⭐ Score: 0-100</td>
        <td>When you know your data falls in a specific range</td>
      </tr>
      <tr>
        <td><code>FLOAT_RESCALED</code></td>
        <td>Unbounded, varied data</td>
        <td>💰 Income: $0-$1M+, 📊 Revenue</td>
        <td>When data has outliers or unknown bounds</td>
      </tr>
      <tr>
        <td><code>FLOAT_DISCRETIZED</code></td>
        <td>Values that form groups</td>
        <td>📅 Years: 1-50, ⭐ Ratings: 1-5</td>
        <td>When groups of values have special meaning</td>
      </tr>
      <tr>
        <td><code>FLOAT</code></td>
        <td>Default normalization</td>
        <td>🔢 General numeric values</td>
        <td>When you want standard normalization (identical to FLOAT_NORMALIZED)</td>
      </tr>
    </tbody>
  </table>
</div>

## 🚀 Basic Usage

<div class="code-section">
  <div class="code-description">
    <p>The simplest way to define numerical features is with the <code>FeatureType</code> enum:</p>
  </div>
  <div class="code-container">

```python
from kdp import PreprocessingModel, FeatureType

# ✨ Quick numerical feature definition
features = {
    "age": FeatureType.FLOAT_NORMALIZED,          # 🧓 Age gets 0-1 normalization
    "income": FeatureType.FLOAT_RESCALED,         # 💰 Income gets robust scaling
    "transaction_count": FeatureType.FLOAT,       # 🔢 Default normalization
    "rating": FeatureType.FLOAT_DISCRETIZED       # ⭐ Discretized into bins
}

# 🏗️ Create your preprocessor
preprocessor = PreprocessingModel(
    path_data="customer_data.csv",
    features_specs=features
)
```

  </div>
</div>

## 🧠 Advanced Configuration

<div class="code-section">
  <div class="code-description">
    <p>For more control, use the <code>NumericalFeature</code> class:</p>
  </div>
  <div class="code-container">

```python
from kdp.features import NumericalFeature

features = {
    # 🧓 Simple example with enhanced configuration
    "age": NumericalFeature(
        name="age",
        feature_type=FeatureType.FLOAT_NORMALIZED,
        use_embedding=True,                 # 🔄 Create neural embeddings
        embedding_dim=16,                   # 📏 Size of embedding
        preferred_distribution="normal"      # 📊 Hint about distribution
    ),

    # 💰 Financial data example
    "transaction_amount": NumericalFeature(
        name="transaction_amount",
        feature_type=FeatureType.FLOAT_RESCALED,
        use_embedding=True,
        embedding_dim=32,
        preferred_distribution="heavy_tailed"
    ),

    # ⏳ Custom binning example
    "years_experience": NumericalFeature(
        name="years_experience",
        feature_type=FeatureType.FLOAT_DISCRETIZED,
        num_bins=5                          # 📏 Number of bins
    )
}
```

  </div>
</div>

## ⚙️ Key Configuration Options

<div class="table-container">
  <table class="config-table">
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Description</th>
        <th>Default</th>
        <th>Suggested Range</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>feature_type</code></td>
        <td>🏷️ Base feature type</td>
        <td><code>FLOAT_NORMALIZED</code></td>
        <td>Choose from 4 types</td>
      </tr>
      <tr>
        <td><code>use_embedding</code></td>
        <td>🧠 Enable neural embeddings</td>
        <td><code>False</code></td>
        <td><code>True</code>/<code>False</code></td>
      </tr>
      <tr>
        <td><code>embedding_dim</code></td>
        <td>📏 Dimensionality of embedding</td>
        <td>8</td>
        <td>4-64</td>
      </tr>
      <tr>
        <td><code>preferred_distribution</code></td>
        <td>📊 Hint about data distribution</td>
        <td><code>None</code></td>
        <td>"normal", "log_normal", etc.</td>
      </tr>
      <tr>
        <td><code>num_bins</code></td>
        <td>🔢 Bins for discretization</td>
        <td>10</td>
        <td>5-100</td>
      </tr>
    </tbody>
  </table>
</div>

## 🔥 Power Features

<div class="feature-cards">
  <div class="feature-card">
    <div class="feature-card-header">
      <span class="feature-icon">📊</span>
      <h3>Distribution-Aware Processing</h3>
    </div>
    <div class="feature-card-content">
      <p>Let KDP automatically detect and handle distributions:</p>
      <div class="code-container">

```python
# ✨ Enable distribution-aware processing for all numerical features
preprocessor = PreprocessingModel(
    features_specs=features,
    use_distribution_aware=True      # 🔍 Enable distribution detection
)
```

      </div>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-header">
      <span class="feature-icon">🧠</span>
      <h3>Advanced Numerical Embeddings</h3>
    </div>
    <div class="feature-card-content">
      <p>Using advanced numerical embeddings:</p>
      <div class="code-container">

```python
from kdp import FeatureType, NumericalFeature, PreprocessingModel

# Configure numerical embeddings
preprocessor = PreprocessingModel(
    features_specs={
        "income": NumericalFeature(
            name="income",
            feature_type=FeatureType.FLOAT_RESCALED,
            use_embedding=True,
            embedding_dim=32,
            preferred_distribution="log_normal"
        )
    }
)
```

      </div>
    </div>
  </div>
</div>

## 💼 Real-World Examples

<div class="example-cards">
  <div class="example-card">
    <div class="example-header">
      <span class="example-icon">💰</span>
      <h3>Financial Analysis</h3>
    </div>
    <div class="code-container">

```python
from kdp import FeatureType, NumericalFeature, PreprocessingModel

# 📈 Financial metrics with appropriate processing
preprocessor = PreprocessingModel(
    features_specs={
        "income": NumericalFeature(
            name="income",
            feature_type=FeatureType.FLOAT_RESCALED,
            preferred_distribution="log_normal"   # 📉 Log-normal distribution
        ),
        "credit_score": NumericalFeature(
            name="credit_score",
            feature_type=FeatureType.FLOAT_NORMALIZED,
            use_embedding=True,
            embedding_dim=16
        ),
        "debt_ratio": NumericalFeature(
            name="debt_ratio",
            feature_type=FeatureType.FLOAT_NORMALIZED,
            preferred_distribution="bounded"      # 📊 Bounded between 0 and 1
        )
    },
    use_distribution_aware=True                   # 🧠 Smart distribution handling
)
```

    </div>
  </div>

  <div class="example-card">
    <div class="example-header">
      <span class="example-icon">🔌</span>
      <h3>Sensor Data</h3>
    </div>
    <div class="code-container">

```python
from kdp import FeatureType, NumericalFeature, PreprocessingModel

# 📡 Processing sensor readings
preprocessor = PreprocessingModel(
    features_specs={
        "temperature": NumericalFeature(
            name="temperature",
            feature_type=FeatureType.FLOAT_RESCALED,
            use_embedding=True,
            embedding_dim=16
        ),
        "humidity": NumericalFeature(
            name="humidity",
            feature_type=FeatureType.FLOAT_NORMALIZED,
            preferred_distribution="bounded"      # 💧 Bounded between 0 and 100
        ),
        "pressure": NumericalFeature(
            name="pressure",
            feature_type=FeatureType.FLOAT_RESCALED,
            use_embedding=True,
            embedding_dim=16
        )
    }
)
```

    </div>
  </div>
</div>

## 💡 Pro Tips

<div class="tips-container">
  <div class="tip-card">
    <div class="tip-header">
      <span class="tip-icon">📊</span>
      <h3>Understand Your Data Distribution</h3>
    </div>
    <div class="tip-content">
      <ul>
        <li>Use <code>FLOAT_NORMALIZED</code> when your data has clear bounds (e.g., 0-100%)</li>
        <li>Use <code>FLOAT_RESCALED</code> when your data has outliers (e.g., income, prices)</li>
        <li>Use <code>FLOAT_DISCRETIZED</code> when your values naturally form groups (e.g., age groups)</li>
      </ul>
    </div>
  </div>

  <div class="tip-card">
    <div class="tip-header">
      <span class="tip-icon">🧠</span>
      <h3>Consider Neural Embeddings for Complex Relationships</h3>
    </div>
    <div class="tip-content">
      <ul>
        <li>Enable when a simple scaling doesn't capture the pattern</li>
        <li>Increase embedding dimensions for more complex patterns (16→32→64)</li>
      </ul>
    </div>
  </div>

  <div class="tip-card">
    <div class="tip-header">
      <span class="tip-icon">🔍</span>
      <h3>Let KDP Handle Distribution Detection</h3>
    </div>
    <div class="tip-content">
      <ul>
        <li>Enable <code>use_distribution_aware=True</code> and let KDP automatically choose</li>
        <li>This is especially important for skewed or multi-modal distributions</li>
      </ul>
    </div>
  </div>

  <div class="tip-card">
    <div class="tip-header">
      <span class="tip-icon">📏</span>
      <h3>Custom Bin Boundaries</h3>
    </div>
    <div class="tip-content">
      <ul>
        <li>Use <code>num_bins</code> parameter to control discretization granularity</li>
        <li>More bins = finer granularity but more parameters to learn</li>
      </ul>
    </div>
  </div>
</div>

## 🔗 Related Topics

<div class="related-topics">
  <a href="../advanced/distribution-aware-encoding.md" class="related-topic-card">
    <span class="related-topic-icon">📊</span>
    <div class="related-topic-content">
      <h3>Distribution-Aware Encoding</h3>
      <p>Smart numerical handling</p>
    </div>
  </a>
  <a href="../advanced/numerical-embeddings.md" class="related-topic-card">
    <span class="related-topic-icon">🧠</span>
    <div class="related-topic-content">
      <h3>Advanced Numerical Embeddings</h3>
      <p>Neural representations</p>
    </div>
  </a>
  <a href="../advanced/feature-selection.md" class="related-topic-card">
    <span class="related-topic-icon">🎯</span>
    <div class="related-topic-content">
      <h3>Feature Selection</h3>
      <p>Finding important features</p>
    </div>
  </a>
</div>

## 🧮 Types of Numerical Features

<div class="types-container">
  <p>KDP supports different types of numerical features, each with specialized processing:</p>

  <div class="types-grid">
    <div class="type-card">
      <span class="type-icon">🔄</span>
      <h3>FLOAT</h3>
      <p>Basic floating-point features with default normalization</p>
    </div>

    <div class="type-card">
      <span class="type-icon">📏</span>
      <h3>FLOAT_NORMALIZED</h3>
      <p>Values normalized to the [0,1] range using min-max scaling</p>
    </div>

    <div class="type-card">
      <span class="type-icon">⚖️</span>
      <h3>FLOAT_RESCALED</h3>
      <p>Values rescaled using standardization (mean=0, std=1)</p>
    </div>

    <div class="type-card">
      <span class="type-icon">📊</span>
      <h3>FLOAT_DISCRETIZED</h3>
      <p>Continuous values binned into discrete buckets</p>
    </div>
  </div>
</div>

## 📊 Architecture Diagrams

<div class="diagram-section">
  <div class="diagram-card">
    <h3>📏 Normalized Numerical Feature</h3>
    <p>Below is a visualization of a model with a normalized numerical feature:</p>
    <div class="diagram-container">
      <img src="imgs/models/basic_numeric_normalized.png" alt="Normalized Numerical Feature" class="diagram-image"/>
    </div>
  </div>

  <div class="diagram-card">
    <h3>⚖️ Rescaled Numerical Feature</h3>
    <p>Below is a visualization of a model with a rescaled numerical feature:</p>
    <div class="diagram-container">
      <img src="imgs/models/basic_numeric_rescaled.png" alt="Rescaled Numerical Feature" class="diagram-image"/>
    </div>
  </div>

  <div class="diagram-card">
    <h3>📊 Discretized Numerical Feature</h3>
    <p>Below is a visualization of a model with a discretized numerical feature:</p>
    <div class="diagram-container">
      <img src="imgs/models/basic_numeric_discretized.png" alt="Discretized Numerical Feature" class="diagram-image"/>
    </div>
  </div>

  <div class="diagram-card">
    <h3>🧠 Advanced Numerical Embeddings</h3>
    <p>When using advanced numerical embeddings, the model architecture looks like this:</p>
    <div class="diagram-container">
      <img src="imgs/models/advanced_numerical_embedding.png" alt="Advanced Numerical Embeddings" class="diagram-image"/>
    </div>
  </div>
</div>

---

<div class="nav-container">
  <a href="overview.md" class="nav-button prev">
    <span class="nav-icon">←</span>
    <span class="nav-text">Feature Overview</span>
  </a>
  <a href="categorical-features.md" class="nav-button next">
    <span class="nav-text">Categorical Features</span>
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

/* Feature header */
.feature-header-container {
  background: linear-gradient(135deg, #f0f7ff 0%, #e9ecef 100%);
  border-radius: 10px;
  padding: 30px;
  margin: 30px 0;
  box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

.feature-header-content h2 {
  margin-top: 0;
  color: #4a86e8;
}

/* Overview card */
.overview-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px 25px;
  margin: 20px 0;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  border-left: 4px solid #4a86e8;
}

/* Tables */
.table-container {
  margin: 25px 0;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.feature-table, .config-table {
  width: 100%;
  border-collapse: collapse;
}

.feature-table th, .config-table th {
  background-color: #f0f7ff;
  padding: 15px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #4a86e8;
}

.feature-table td, .config-table td {
  padding: 12px 15px;
  border-bottom: 1px solid #eaecef;
}

.feature-table tr:nth-child(even), .config-table tr:nth-child(even) {
  background-color: #f8f9fa;
}

.feature-table tr:hover, .config-table tr:hover {
  background-color: #f0f7ff;
}

/* Code sections */
.code-section {
  margin: 25px 0;
}

.code-description {
  margin-bottom: 15px;
}

.code-container {
  background-color: #f8f9fa;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.code-container pre {
  margin: 0;
  padding: 20px;
}

/* Feature cards */
.feature-cards {
  display: grid;
  grid-template-columns: 1fr;
  gap: 20px;
  margin: 30px 0;
}

.feature-card {
  background-color: #fff;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.feature-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.feature-card-header {
  display: flex;
  align-items: center;
  padding: 15px 20px;
  background: linear-gradient(135deg, #f0f7ff 0%, #e9ecef 100%);
  border-bottom: 1px solid #e9ecef;
}

.feature-icon {
  font-size: 1.5em;
  margin-right: 15px;
}

.feature-card-header h3 {
  margin: 0;
  color: #333;
}

.feature-card-content {
  padding: 20px;
}

/* Example cards */
.example-cards {
  display: grid;
  grid-template-columns: 1fr;
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

/* Tips */
.tips-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
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

.tip-icon {
  font-size: 1.5em;
  margin-right: 15px;
}

.tip-header h3 {
  margin: 0;
  color: #333;
}

.tip-content {
  padding: 15px 20px;
}

.tip-content ul {
  margin: 0;
  padding-left: 20px;
}

.tip-content li {
  margin-bottom: 8px;
}

/* Related topics */
.related-topics {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.related-topic-card {
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

.related-topic-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
  background-color: #f0f7ff;
}

.related-topic-icon {
  font-size: 1.5em;
  margin-right: 15px;
}

.related-topic-content h3 {
  margin: 0 0 5px 0;
  color: #4a86e8;
}

.related-topic-content p {
  margin: 0;
  font-size: 14px;
  color: #555;
}

/* Types section */
.types-container {
  margin: 30px 0;
}

.types-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 20px;
  margin-top: 20px;
}

.type-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  text-align: center;
}

.type-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.type-icon {
  font-size: 2em;
  display: block;
  margin-bottom: 10px;
}

.type-card h3 {
  margin: 10px 0;
  color: #4a86e8;
}

.type-card p {
  margin: 0;
  font-size: 15px;
}

/* Diagram section */
.diagram-section {
  display: grid;
  grid-template-columns: 1fr;
  gap: 30px;
  margin: 30px 0;
}

.diagram-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.diagram-card h3 {
  margin-top: 0;
  color: #4a86e8;
  border-bottom: 2px solid #eaecef;
  padding-bottom: 10px;
}

.diagram-container {
  text-align: center;
  margin-top: 20px;
}

.diagram-image {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  box-shadow: 0 2px 5px rgba(0,0,0,0.1);
  transition: transform 0.3s ease;
}

.diagram-image:hover {
  transform: scale(1.02);
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
@media (max-width: 1024px) {
  .tips-container {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .tips-container,
  .example-cards,
  .related-topics,
  .types-grid {
    grid-template-columns: 1fr;
  }
}
</style>
