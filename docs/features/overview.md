# 🛠️ Feature Types Overview

<div class="feature-overview-header">
  <div class="overview-title">
    <h2>Making Data ML-Ready</h2>
    <p>KDP makes feature processing intuitive and powerful by transforming your raw data into the optimal format for machine learning.</p>
  </div>
</div>

## 💪 Feature Types at a Glance

<div class="features-overview-container">
  <div class="table-container">
    <table class="features-overview-table">
      <thead>
        <tr>
          <th>Feature Type</th>
          <th>What It's For</th>
          <th>Processing Magic</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>🔢 <strong>Numerical</strong></td>
          <td>Continuous values like age, income, scores</td>
          <td>Normalization, scaling, embeddings, distribution analysis</td>
        </tr>
        <tr>
          <td>🏷️ <strong>Categorical</strong></td>
          <td>Discrete values like occupation, product type</td>
          <td>Embeddings, one-hot encoding, vocabulary management</td>
        </tr>
        <tr>
          <td>📝 <strong>Text</strong></td>
          <td>Free-form text like reviews, descriptions</td>
          <td>Tokenization, embeddings, sequence handling</td>
        </tr>
        <tr>
          <td>📅 <strong>Date</strong></td>
          <td>Temporal data like signup dates, transactions</td>
          <td>Component extraction, cyclical encoding, seasonality</td>
        </tr>
        <tr>
          <td>➕ <strong>Cross Features</strong></td>
          <td>Feature interactions</td>
          <td>Combined embeddings, interaction modeling</td>
        </tr>
        <tr>
          <td>🔍 <strong>Passthrough</strong></td>
          <td>IDs, metadata, pre-processed data</td>
          <td>Input signatures without processing (v1.11.1+: separate or included)</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

## 🚀 Getting Started

<div class="getting-started-container">
  <p>The simplest way to define features is with the <code>FeatureType</code> enum:</p>

  <div class="code-container">

```python
from kdp import PreprocessingModel, FeatureType

# ✨ Quick and easy feature definition
features = {
    # 🔢 Numerical features - different processing strategies
    "age": FeatureType.FLOAT_NORMALIZED,        # 📊 [0,1] range normalization
    "income": FeatureType.FLOAT_RESCALED,       # 📈 Standard scaling
    "transaction_count": FeatureType.FLOAT,     # 🧮 Default normalization (same as FLOAT_NORMALIZED)

    # 🏷️ Categorical features - automatic encoding
    "occupation": FeatureType.STRING_CATEGORICAL,      # 👔 Job titles, roles
    "education_level": FeatureType.INTEGER_CATEGORICAL, # 🎓 Education codes

    # 📝 Text and dates - specialized processing
    "product_review": FeatureType.TEXT,         # 💬 Customer feedback
    "signup_date": FeatureType.DATE,            # 📆 User registration date

    # 🔍 Passthrough feature - use without any processing
    "embedding_vector": FeatureType.PASSTHROUGH # 🔄 Pre-processed data passes directly to output
}

# 🏗️ Create your preprocessor
preprocessor = PreprocessingModel(
    path_data="customer_data.csv",
    features_specs=features
)
```

  </div>
</div>

## ⭐ Why Strong Feature Types Matter

<div class="benefits-grid">
  <div class="benefit-card">
    <span class="benefit-icon">🎯</span>
    <h3>Optimized Processing</h3>
    <p>Each feature type gets specialized handling for better ML performance</p>
  </div>

  <div class="benefit-card">
    <span class="benefit-icon">🐛</span>
    <h3>Reduced Errors</h3>
    <p>Catch type mismatches early in development, not during training</p>
  </div>

  <div class="benefit-card">
    <span class="benefit-icon">📝</span>
    <h3>Clearer Code</h3>
    <p>Self-documenting feature definitions make your code more maintainable</p>
  </div>

  <div class="benefit-card">
    <span class="benefit-icon">⚡</span>
    <h3>Enhanced Performance</h3>
    <p>Type-specific optimizations improve preprocessing speed</p>
  </div>
</div>

## 📚 Feature Type Documentation

<div class="feature-types-grid">
  <a href="numerical-features.md" class="feature-type-card">
    <span class="feature-type-icon">🔢</span>
    <div class="feature-type-content">
      <h3>Numerical Features</h3>
      <p>Handle continuous values with advanced normalization and distribution-aware processing</p>
    </div>
  </a>

  <a href="categorical-features.md" class="feature-type-card">
    <span class="feature-type-icon">🏷️</span>
    <div class="feature-type-content">
      <h3>Categorical Features</h3>
      <p>Process discrete categories with smart embedding techniques and vocabulary management</p>
    </div>
  </a>

  <a href="text-features.md" class="feature-type-card">
    <span class="feature-type-icon">📝</span>
    <div class="feature-type-content">
      <h3>Text Features</h3>
      <p>Work with free-form text using tokenization, embeddings, and sequence handling</p>
    </div>
  </a>

  <a href="date-features.md" class="feature-type-card">
    <span class="feature-type-icon">📅</span>
    <div class="feature-type-content">
      <h3>Date Features</h3>
      <p>Extract temporal patterns from dates with component extraction and cyclical encoding</p>
    </div>
  </a>

  <a href="cross-features.md" class="feature-type-card">
    <span class="feature-type-icon">➕</span>
    <div class="feature-type-content">
      <h3>Cross Features</h3>
      <p>Model feature interactions with combined embeddings and interaction modeling</p>
    </div>
  </a>

  <a href="passthrough-features.md" class="feature-type-card">
    <span class="feature-type-icon">🔍</span>
    <div class="feature-type-content">
      <h3>Passthrough Features</h3>
      <p>Include unmodified data or pre-computed features directly in your model</p>
    </div>
  </a>
</div>

## 👨‍💻 Advanced Feature Configuration

<div class="advanced-config-container">
  <p>For more control, use specialized feature classes:</p>

  <div class="code-container">

```python
from kdp.features import NumericalFeature, CategoricalFeature, TextFeature, DateFeature, PassthroughFeature
import tensorflow as tf

# 🔧 Advanced feature configuration
features = {
    # 💰 Numerical with advanced embedding
    "income": NumericalFeature(
        name="income",
        feature_type=FeatureType.FLOAT_RESCALED,
        embedding_size=32
    ),

    # 🏪 Categorical with hashing
    "product_id": CategoricalFeature(
        name="product_id",
        feature_type=FeatureType.STRING_CATEGORICAL,
        max_tokens=10000,
        category_encoding="hashing"
    ),

    # 📋 Text with custom tokenization
    "description": TextFeature(
        name="description",
        max_tokens=5000,
        embedding_size=64,
        sequence_length=128,
        ngrams=2
    ),

    # 🗓️ Date with cyclical encoding
    "purchase_date": DateFeature(
        name="purchase_date"
    ),

    # 🧠 Passthrough feature
    "embedding": PassthroughFeature(
        name="embedding",
        dtype=tf.float32
    )
}
```

  </div>
</div>

## 💡 Pro Tips for Feature Definition

<div class="pro-tips-grid">
  <div class="pro-tip-card">
    <span class="pro-tip-number">1</span>
    <div class="pro-tip-content">
      <h3>Start Simple</h3>
      <p>Begin with basic <code>FeatureType</code> definitions</p>
    </div>
  </div>

  <div class="pro-tip-card">
    <span class="pro-tip-number">2</span>
    <div class="pro-tip-content">
      <h3>Add Complexity Gradually</h3>
      <p>Refactor to specialized feature classes when needed</p>
    </div>
  </div>

  <div class="pro-tip-card">
    <span class="pro-tip-number">3</span>
    <div class="pro-tip-content">
      <h3>Combine Approaches</h3>
      <p>Mix distribution-aware, attention, embeddings for best results</p>
    </div>
  </div>

  <div class="pro-tip-card">
    <span class="pro-tip-number">4</span>
    <div class="pro-tip-content">
      <h3>Check Distributions</h3>
      <p>Review your data distribution before choosing feature types</p>
    </div>
  </div>

  <div class="pro-tip-card">
    <span class="pro-tip-number">5</span>
    <div class="pro-tip-content">
      <h3>Experiment with Types</h3>
      <p>Sometimes a different encoding provides better results</p>
    </div>
  </div>

  <div class="pro-tip-card">
    <span class="pro-tip-number">6</span>
    <div class="pro-tip-content">
      <h3>Consider Passthrough</h3>
      <p>Use passthrough features for pre-processed data or custom vectors</p>
    </div>
  </div>
</div>

## 📊 Model Architecture Diagrams

<div class="architecture-container">
  <p>KDP creates optimized preprocessing architectures based on your feature definitions. Here are examples of different model configurations:</p>

  <div class="architecture-section">
    <h3>🔄 Basic Feature Combinations</h3>
    <p>When combining numerical and categorical features:</p>
    <div class="architecture-image-container">
      <img src="imgs/models/numeric_and_categorical.png" alt="Numeric and Categorical Features" class="architecture-image"/>
    </div>
  </div>

  <div class="architecture-section">
    <h3>🌟 All Feature Types Combined</h3>
    <p>KDP can handle all feature types in a single model:</p>
    <div class="architecture-image-container">
      <img src="imgs/models/all_basic_types.png" alt="All Feature Types Combined" class="architecture-image"/>
    </div>
  </div>

  <div class="architecture-section">
    <h3>🔋 Advanced Configurations</h3>

    <div class="advanced-architectures">
      <div class="advanced-architecture-card">
        <h4>✨ Tabular Attention</h4>
        <p>Enhance feature interactions with tabular attention:</p>
        <div class="architecture-image-container">
          <img src="imgs/models/tabular_attention.png" alt="Tabular Attention" class="architecture-image"/>
        </div>
      </div>

      <div class="advanced-architecture-card">
        <h4>🔄 Transformer Blocks</h4>
        <p>Process categorical features with transformer blocks:</p>
        <div class="architecture-image-container">
          <img src="imgs/models/transformer_blocks.png" alt="Transformer Blocks" class="architecture-image"/>
        </div>
      </div>

      <div class="advanced-architecture-card">
        <h4>🧠 Feature MoE (Mixture of Experts)</h4>
        <p>Specialized feature processing with Mixture of Experts:</p>
        <div class="architecture-image-container">
          <img src="imgs/models/feature_moe.png" alt="Feature MoE" class="architecture-image"/>
        </div>
      </div>
    </div>
  </div>

  <div class="architecture-section">
    <h3>📤 Output Modes</h3>
    <p>KDP supports different output modes for your preprocessed features:</p>

    <div class="output-modes">
      <div class="output-mode-card">
        <h4>🔗 Concatenated Output</h4>
        <div class="architecture-image-container">
          <img src="imgs/models/output_mode_concat.png" alt="Concat Output Mode" class="architecture-image"/>
        </div>
      </div>

      <div class="output-mode-card">
        <h4>📦 Dictionary Output</h4>
        <div class="architecture-image-container">
          <img src="imgs/models/output_mode_dict.png" alt="Dict Output Mode" class="architecture-image"/>
        </div>
      </div>
    </div>
  </div>
</div>

---

<div class="nav-container">
  <a href="../getting-started/architecture.md" class="nav-button prev">
    <span class="nav-icon">←</span>
    <span class="nav-text">Architecture</span>
  </a>
  <a href="numerical-features.md" class="nav-button next">
    <span class="nav-text">Numerical Features</span>
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

/* Feature overview header */
.feature-overview-header {
  background: linear-gradient(135deg, #4a86e8 0%, #7dabf5 100%);
  border-radius: 10px;
  padding: 30px;
  margin: 30px 0;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  color: white;
}

.overview-title h2 {
  margin-top: 0;
  font-size: 28px;
}

.overview-title p {
  font-size: 18px;
  margin-bottom: 0;
  opacity: 0.9;
}

/* Features overview container */
.features-overview-container {
  margin: 30px 0;
}

/* Table styling */
.table-container {
  margin: 20px 0;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}

.features-overview-table {
  width: 100%;
  border-collapse: collapse;
}

.features-overview-table th {
  background-color: #f0f7ff;
  padding: 15px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #4a86e8;
}

.features-overview-table td {
  padding: 12px 15px;
  border-bottom: 1px solid #eaecef;
}

.features-overview-table tr:nth-child(even) {
  background-color: #f8f9fa;
}

.features-overview-table tr:hover {
  background-color: #f0f7ff;
}

/* Getting started container */
.getting-started-container {
  background-color: #f8f9fa;
  border-radius: 10px;
  padding: 20px;
  margin: 30px 0;
  border-left: 4px solid #4a86e8;
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

/* Benefits grid */
.benefits-grid {
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
  color: #4a86e8;
}

.benefit-card p {
  margin: 0;
}

/* Feature types grid */
.feature-types-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.feature-type-card {
  display: flex;
  align-items: flex-start;
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  text-decoration: none;
  color: #333;
}

.feature-type-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.feature-type-icon {
  font-size: 2.5em;
  margin-right: 15px;
}

.feature-type-content {
  flex: 1;
}

.feature-type-content h3 {
  margin: 0 0 10px 0;
  color: #4a86e8;
}

.feature-type-content p {
  margin: 0;
  font-size: 14px;
}

/* Advanced config container */
.advanced-config-container {
  background-color: #f8f9fa;
  border-radius: 10px;
  padding: 20px;
  margin: 30px 0;
  border-left: 4px solid #4a86e8;
}

/* Pro tips */
.pro-tips-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.pro-tip-card {
  display: flex;
  align-items: flex-start;
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

.pro-tip-number {
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

.pro-tip-content {
  flex: 1;
}

.pro-tip-content h3 {
  margin: 0 0 5px 0;
  color: #4a86e8;
}

.pro-tip-content p {
  margin: 0;
}

/* Architecture container */
.architecture-container {
  margin: 30px 0;
}

.architecture-section {
  margin: 30px 0;
}

.architecture-section h3 {
  color: #4a86e8;
  border-bottom: 1px solid #eaecef;
  padding-bottom: 10px;
}

.architecture-image-container {
  text-align: center;
  margin: 20px 0;
}

.architecture-image {
  max-width: 100%;
  height: auto;
  border-radius: 10px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  transition: transform 0.3s ease;
}

.architecture-image:hover {
  transform: scale(1.02);
}

.advanced-architectures, .output-modes {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 20px;
  margin: 20px 0;
}

.advanced-architecture-card, .output-mode-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}

.advanced-architecture-card h4, .output-mode-card h4 {
  color: #4a86e8;
  margin-top: 0;
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
  .benefits-grid,
  .feature-types-grid,
  .pro-tips-grid,
  .advanced-architectures,
  .output-modes {
    grid-template-columns: 1fr;
  }

  .feature-type-card {
    flex-direction: column;
    align-items: center;
    text-align: center;
  }

  .feature-type-icon {
    margin-right: 0;
    margin-bottom: 15px;
  }

  .pro-tip-card {
    flex-direction: column;
    align-items: center;
    text-align: center;
  }

  .pro-tip-number {
    margin-right: 0;
    margin-bottom: 15px;
  }
}
</style>
