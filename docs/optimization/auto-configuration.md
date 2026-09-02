# 🧙‍♂️ Auto-Configuration: Analytics and Recommendations

<style>
.feature-header {
  background: linear-gradient(135deg, #FF5722 0%, #FF7043 100%);
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

.intro-container {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  border-radius: 10px;
  padding: 30px;
  margin: 30px 0;
  box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

.intro-content h2 {
  color: #4a86e8;
  margin-top: 0;
}

.step-card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  margin: 20px 0;
  overflow: hidden;
}

.step-header {
  background: #f8f9fa;
  padding: 15px 20px;
  border-bottom: 1px solid #e9ecef;
}

.step-header h3 {
  margin: 0;
  color: #333;
}

.code-container {
  background: #1e1e1e;
  border-radius: 8px;
  padding: 20px;
  margin: 20px 0;
  overflow-x: auto;
}

.code-container pre {
  margin: 0;
  color: #d4d4d4;
}

.grid-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.grid-item {
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.grid-item:hover {
  transform: translateY(-5px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.feature-icon {
  font-size: 24px;
  margin-bottom: 10px;
  display: block;
}

.table-container {
  margin: 30px 0;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.feature-table {
  width: 100%;
  border-collapse: collapse;
}

.feature-table th {
  background: #f8f9fa;
  padding: 15px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #e9ecef;
}

.feature-table td {
  padding: 12px 15px;
  border-bottom: 1px solid #e9ecef;
}

.feature-table tr:nth-child(even) {
  background: #f8f9fa;
}

.feature-table tr:hover {
  background: #f0f7ff;
}

.examples-container {
  margin: 30px 0;
}

.example-card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  margin-bottom: 20px;
  overflow: hidden;
}

.example-header {
  background: #f8f9fa;
  padding: 15px 20px;
  border-bottom: 1px solid #e9ecef;
  display: flex;
  align-items: center;
}

.example-icon {
  font-size: 20px;
  margin-right: 10px;
}

.example-header h3 {
  margin: 0;
  color: #333;
}

@media (max-width: 768px) {
  .grid-container {
    grid-template-columns: 1fr;
  }

  .feature-header {
    padding: 20px;
  }

  .intro-container {
    padding: 20px;
  }
}
</style>

<div class="feature-header">
  <div class="feature-title">
    <h2>Auto-Configuration: Analytics and Recommendations</h2>
    <p>Let KDP analyze your data and suggest the optimal preprocessing</p>
  </div>
</div>

<div class="intro-container">
  <div class="intro-content">
    <h2>Intelligent Data Analysis</h2>
    <p>Auto-Configuration examines your dataset and provides intelligent recommendations for feature processing, helping you build better models faster.</p>
  </div>
</div>

!!! note "How the feature types are chosen"
    Without `features_specs`, KDP reads a sample of the file and infers a type
    per column:

    | Column looks like | Inferred type |
    |---|---|
    | Floats, or whole numbers with many distinct values | `FLOAT_NORMALIZED` |
    | Whole numbers with at most 20 distinct values | `INTEGER_CATEGORICAL` |
    | Strings that parse as dates (`YYYY-MM-DD` or `YYYY/MM/DD`) | `DATE` |
    | Strings of at most three words | `STRING_CATEGORICAL` |
    | Longer strings | `TEXT` |

    A high-cardinality single-token column such as an id stays categorical, so
    hashing can keep it bounded rather than building a vocabulary the size of
    the dataset. Pass `features_specs` for any column you want decided
    differently &mdash; explicit specs are used as given.

## 🚀 Getting Started

<div class="step-card">
  <div class="step-header">
    <h3>Basic Usage</h3>
  </div>
  <div class="code-container">

```python
from kdp import auto_configure, PreprocessingModel

# Analyze data and get recommendations
config = auto_configure("customer_data.csv")

# Review the recommendations
recommendations = config["recommendations"]
code_snippet = config["code_snippet"]

# Create your preprocessor using the code snippet as a guide
# Note: You'll need to manually implement the suggestions
```

  </div>
</div>

## ✨ What Auto-Configuration Provides

<div class="grid-container">
  <div class="grid-item">
    <span class="feature-icon">🔍</span>
    <h3>Distribution Analysis</h3>
    <p>Identifies patterns in your numeric data to suggest optimal transformations</p>
  </div>
  <div class="grid-item">
    <span class="feature-icon">📊</span>
    <h3>Feature Statistics</h3>
    <p>Calculates important statistics about your features to guide preprocessing</p>
  </div>
  <div class="grid-item">
    <span class="feature-icon">💡</span>
    <h3>Preprocessing Recommendations</h3>
    <p>Suggests appropriate feature types and transformations based on data analysis</p>
  </div>
  <div class="grid-item">
    <span class="feature-icon">📝</span>
    <h3>Example Code</h3>
    <p>Generates ready-to-use code snippets based on the analysis</p>
  </div>
</div>

## 🔍 What It Analyzes

<div class="table-container">
  <table class="feature-table">
    <thead>
      <tr>
        <th>Data Characteristic</th>
        <th>Example</th>
        <th>What It Detects</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Distribution Types</strong></td>
        <td>Log-normal income, bimodal age</td>
        <td>Statistical distribution patterns</td>
      </tr>
      <tr>
        <td><strong>Feature Statistics</strong></td>
        <td>Mean, variance, skewness</td>
        <td>Basic statistical properties</td>
      </tr>
      <tr>
        <td><strong>Data Ranges</strong></td>
        <td>Min/max values, outliers</td>
        <td>Value boundaries and extremes</td>
      </tr>
      <tr>
        <td><strong>Value Patterns</strong></td>
        <td>Discrete vs continuous</td>
        <td>How values are distributed</td>
      </tr>
    </tbody>
  </table>
</div>

## 💼 Examples

<div class="examples-container">
  <div class="example-card">
    <div class="example-header">
      <span class="example-icon">🔎</span>
      <h3>Basic Analysis</h3>
    </div>
    <div class="code-container">

```python
# Basic auto-configuration analysis
config = auto_configure(
    "customer_data.csv",  # Your dataset
    batch_size=50000,     # Process in batches of this size
    save_stats=True       # Save computed statistics
)

# Review the recommendations
for feature_name, recommendation in config["recommendations"].items():
    print(f"Feature: {feature_name}")
    print(f"  Type: {recommendation['feature_type']}")
    print(f"  Preprocessing: {recommendation['preprocessing']}")

# Get the suggested code snippet
print(config["code_snippet"])
```

    </div>
  </div>
</div>

## 📊 Understanding the Results

<div class="example-card">
  <div class="example-header">
    <span class="example-icon">📊</span>
    <h3>Results Structure</h3>
  </div>
  <div class="code-container">

```python
# Example results structure
config = {
    "recommendations": {
        "income": {
            "feature_type": "NumericalFeature",
            "preprocessing": ["NORMALIZATION"],
            "detected_distribution": "log_normal",
            "config": {
                # Specific configuration recommendations
            }
        },
        # More features...
    },
    "code_snippet": "# Python code with recommended configuration",
    "statistics": {
        # If save_stats=True, contains computed statistics
    }
}
```

  </div>
</div>

## 🛠️ Available Options

<div class="example-card">
  <div class="example-header">
    <span class="example-icon">⚙️</span>
    <h3>Configuration Options</h3>
  </div>
  <div class="code-container">

```python
# Auto-configuration with options
config = auto_configure(
    data_path="customer_data.csv",      # Path to your dataset
    features_specs=None,                # Optional: provide existing features specs
    batch_size=50000,                   # Batch size for processing
    save_stats=True,                    # Whether to include statistics in results
    stats_path="features_stats.json",   # Where to save/load statistics
    overwrite_stats=False               # Whether to recalculate existing stats
)
```

  </div>
</div>

## 💡 Pro Tips

<div class="grid-container">
  <div class="grid-item">
    <span class="feature-icon">👀</span>
    <h3>Review Before Implementing</h3>
    <p>Always review the recommendations before blindly applying them</p>
    <div class="code-container">

```python
# Inspect the recommendations first
config = auto_configure("data.csv")

# Review before implementing
for feature, recommendation in config["recommendations"].items():
    print(f"{feature}: {recommendation['detected_distribution']}")
```

    </div>
  </div>
  <div class="grid-item">
    <span class="feature-icon">🧠</span>
    <h3>Combine with Domain Knowledge</h3>
    <p>Use the recommendations alongside your domain expertise</p>
    <div class="code-container">

```python
# Get recommendations
config = auto_configure("data.csv")

# Create your features dictionary, informed by recommendations
features = {
    "income": FeatureType.FLOAT_RESCALED,  # Based on recommendation
    "age": FeatureType.FLOAT_NORMALIZED,   # Based on domain knowledge
}
```

    </div>
  </div>
  <div class="grid-item">
    <span class="feature-icon">🔄</span>
    <h3>Update When Data Changes</h3>
    <p>Rerun when your data distribution changes</p>
    <div class="code-container">

```python
# Update statistics with new data
new_config = auto_configure(
    "updated_data.csv",
    overwrite_stats=True  # Force recalculation with new data
)
```

    </div>
  </div>
</div>

## 🔗 Related Topics

<div class="grid-container">
  <div class="grid-item">
    <span class="feature-icon">📊</span>
    <h3>Distribution-Aware Encoding</h3>
    <p>Apply recommendations for numerical features</p>
    <a href="../advanced/distribution-aware-encoding.md">Learn more →</a>
  </div>
  <div class="grid-item">
    <span class="feature-icon">🎯</span>
    <h3>Feature Selection</h3>
    <p>Improve model performance</p>
    <a href="feature-selection.md">Learn more →</a>
  </div>
  <div class="grid-item">
    <span class="feature-icon">📚</span>
    <h3>Feature Types Overview</h3>
    <p>Learn about all available feature types</p>
    <a href="../features/overview.md">Learn more →</a>
  </div>
</div>
