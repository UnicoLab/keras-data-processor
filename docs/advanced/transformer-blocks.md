# ⚡ Transformer Blocks

<div class="feature-header">
  <div class="feature-title">
    <h2>Transformer Blocks</h2>
    <p>Powerful self-attention mechanisms for tabular data</p>
  </div>
</div>

## 📋 Overview

<div class="overview-card">
  <p>Transformer Blocks in KDP bring the power of self-attention mechanisms to tabular data processing. These blocks enable your models to capture complex feature interactions and dependencies through sophisticated attention mechanisms, leading to better model performance on structured data.</p>
</div>

<div class="key-benefits">
  <div class="benefit-card">
    <span class="benefit-icon">🧠</span>
    <h3>Self-Attention</h3>
    <p>Capture complex feature interactions</p>
  </div>
  <div class="benefit-card">
    <span class="benefit-icon">🔄</span>
    <h3>Multi-Head Processing</h3>
    <p>Learn diverse feature relationships</p>
  </div>
  <div class="benefit-card">
    <span class="benefit-icon">⚡</span>
    <h3>Efficient Computation</h3>
    <p>Optimized for tabular data</p>
  </div>
  <div class="benefit-card">
    <span class="benefit-icon">🎯</span>
    <h3>Feature Importance</h3>
    <p>Learn which features matter most</p>
  </div>
</div>

## 🚀 Getting Started

<div class="code-container">

```python
from kdp import PreprocessingModel, FeatureType

# Define features
features_specs = {
    "age": FeatureType.FLOAT_NORMALIZED,
    "income": FeatureType.FLOAT_RESCALED,
    "occupation": FeatureType.STRING_CATEGORICAL,
    "education": FeatureType.INTEGER_CATEGORICAL
}

# Initialize model with transformer blocks
preprocessor = PreprocessingModel(
    path_data="data/my_data.csv",
    features_specs=features_specs,
    transfo_nr_blocks=3,            # Number of transformer blocks
    transfo_nr_heads=4,             # Number of attention heads
    transfo_ff_units=64,            # Feed-forward hidden dimension
)
```

</div>

## 🧠 How It Works

<div class="architecture-diagram">
  <img src="imgs/transformer_blocks.png" alt="Transformer Blocks Architecture" class="architecture-image">
  <div class="diagram-caption">
    <p>KDP's transformer blocks process tabular data through multiple layers of self-attention and feed-forward networks, enabling the model to learn complex feature interactions and dependencies.</p>
  </div>
</div>

## ⚙️ Configuration Options

<div class="table-container">
  <table class="config-table">
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Type</th>
        <th>Default</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>transfo_nr_blocks</code></td>
        <td>int | None</td>
        <td>None</td>
        <td>Number of transformer blocks. <strong>Setting this is what enables them</strong> &mdash; leave it <code>None</code> and no blocks are added.</td>
      </tr>
      <tr>
        <td><code>transfo_nr_heads</code></td>
        <td>int</td>
        <td>4</td>
        <td>Number of attention heads</td>
      </tr>
      <tr>
        <td><code>transfo_ff_units</code></td>
        <td>int</td>
        <td>64</td>
        <td>Hidden dimension</td>
      </tr>
      <tr>
        <td><code>transfo_dropout_rate</code></td>
        <td>float</td>
        <td>0.1</td>
        <td>Dropout rate</td>
      </tr>
    </tbody>
  </table>
</div>

## 💡 Pro Tips

<div class="pro-tips-grid">
  <div class="pro-tip-card">
    <h3>Block Configuration</h3>
    <p>Start with 2-3 blocks and increase based on feature complexity. More blocks can capture deeper interactions but may lead to overfitting.</p>
  </div>

  <div class="pro-tip-card">
    <h3>Head Selection</h3>
    <p>Use 4-8 heads for most tasks. More heads can capture diverse relationships but increase computational cost.</p>
  </div>

  <div class="pro-tip-card">
    <h3>Dimension Tuning</h3>
    <p>Choose dimensions divisible by number of heads. Larger dimensions capture more complex patterns but require more computation.</p>
  </div>
</div>

## 🔍 Examples

<div class="examples-container">
  <div class="example-card">
    <h3>Customer Analytics</h3>
    <div class="code-container">

```python
from kdp import FeatureType, PreprocessingModel

features_specs = {
    "age": FeatureType.FLOAT_NORMALIZED,
    "income": FeatureType.FLOAT_RESCALED,
    "tenure": FeatureType.FLOAT_NORMALIZED,
    "purchases": FeatureType.FLOAT_RESCALED,
    "customer_type": FeatureType.STRING_CATEGORICAL,
    "region": FeatureType.STRING_CATEGORICAL
}

preprocessor = PreprocessingModel(
    path_data="data/customer_data.csv",
    features_specs=features_specs,
    transfo_nr_blocks=4,            # More blocks for complex customer patterns
    transfo_nr_heads=8,             # More heads for diverse relationships
    transfo_ff_units=128,                 # Larger dimension for rich representations
    transfo_dropout_rate=0.2              # Higher dropout for regularization
)
```

    </div>
  </div>

  <div class="example-card">
    <h3>Product Recommendations</h3>
    <div class="code-container">

```python
from kdp import FeatureType, PreprocessingModel

features_specs = {
    "user_id": FeatureType.INTEGER_CATEGORICAL,
    "item_id": FeatureType.INTEGER_CATEGORICAL,
    "category": FeatureType.STRING_CATEGORICAL,
    "price": FeatureType.FLOAT_NORMALIZED,
    "rating": FeatureType.FLOAT_NORMALIZED,
    "timestamp": FeatureType.DATE
}

preprocessor = PreprocessingModel(
    path_data="data/recommendation_data.csv",
    features_specs=features_specs,
    transfo_nr_blocks=3,            # Standard configuration
    transfo_nr_heads=4,             # Balanced number of heads
    transfo_ff_units=64,                  # Moderate dimension
    transfo_dropout_rate=0.1              # Standard dropout
)
```

    </div>
  </div>
</div>

## 🔗 Related Topics

<div class="related-topics">
  <a href="tabular-attention.md" class="topic-link">
    <span class="topic-icon">👁️</span>
    <span class="topic-text">Tabular Attention</span>
  </a>
  <a href="feature-moe.md" class="topic-link">
    <span class="topic-icon">🧩</span>
    <span class="topic-text">Feature-wise MoE</span>
  </a>
  <a href="numerical-embeddings.md" class="topic-link">
    <span class="topic-icon">🧮</span>
    <span class="topic-text">Advanced Numerical Embeddings</span>
  </a>
  <a href="../features/cross-features.md" class="topic-link">
    <span class="topic-icon">🔗</span>
    <span class="topic-text">Cross Features</span>
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
.feature-header {
  background: linear-gradient(135deg, #673ab7 0%, #9c27b0 100%);
  border-radius: 10px;
  padding: 30px;
  margin: 30px 0;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  color: white;
}

.feature-title h2 {
  margin-top: 0;
  font-size: 28px;
}

.feature-title p {
  font-size: 18px;
  margin-bottom: 0;
  opacity: 0.9;
}

/* Overview card */
.overview-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px 25px;
  margin: 20px 0;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  border-left: 4px solid #673ab7;
}

.overview-card p {
  margin: 0;
  font-size: 16px;
}

/* Key benefits */
.key-benefits {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.benefit-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.benefit-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.benefit-icon {
  font-size: 2.5em;
  margin-bottom: 15px;
}

.benefit-card h3 {
  margin: 0 0 10px 0;
  color: #673ab7;
}

.benefit-card p {
  margin: 0;
}

/* Code containers */
.code-container {
  background-color: #f8f9fa;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 5px rgba(0,0,0,0.1);
  margin: 20px 0;
}

.code-container pre {
  margin: 0;
  padding: 20px;
}

/* Architecture diagram */
.architecture-diagram {
  background-color: white;
  border-radius: 10px;
  padding: 20px;
  margin: 30px 0;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  text-align: center;
}

.architecture-image {
  max-width: 100%;
  border-radius: 5px;
}

.diagram-caption {
  margin-top: 20px;
  text-align: center;
  font-style: italic;
}

/* Tables */
.table-container {
  margin: 30px 0;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}

.config-table {
  width: 100%;
  border-collapse: collapse;
}

.config-table th {
  background-color: #ede7f6;
  padding: 15px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #673ab7;
}

.config-table td {
  padding: 12px 15px;
  border-bottom: 1px solid #eaecef;
}

.config-table tr:nth-child(even) {
  background-color: #f8f9fa;
}

.config-table tr:hover {
  background-color: #ede7f6;
}

/* Pro tips */
.pro-tips-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.pro-tip-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.pro-tip-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.pro-tip-card h3 {
  margin-top: 0;
  color: #673ab7;
}

/* Examples */
.examples-container {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.example-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.example-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.example-card h3 {
  margin-top: 0;
  color: #673ab7;
}

/* Related topics */
.related-topics {
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
  margin: 30px 0;
}

.topic-link {
  display: flex;
  align-items: center;
  padding: 10px 15px;
  background-color: #ede7f6;
  border-radius: 8px;
  text-decoration: none;
  color: #333;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  transition: background-color 0.3s ease, transform 0.3s ease;
}

.topic-link:hover {
  background-color: #d1c4e9;
  transform: translateY(-2px);
}

.topic-icon {
  font-size: 1.2em;
  margin-right: 10px;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .key-benefits,
  .pro-tips-grid,
  .examples-container {
    grid-template-columns: 1fr;
  }

  .related-topics {
    flex-direction: column;
  }
}
</style>
