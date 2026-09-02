# 🧩 Feature-wise Mixture of Experts

<div class="feature-header">
  <div class="feature-title">
    <h2>Feature-wise Mixture of Experts (MoE)</h2>
    <p>Specialized processing for heterogeneous tabular features</p>
  </div>
</div>

## 📋 Overview

<div class="overview-card">
  <p>Feature-wise Mixture of Experts (MoE) is a powerful technique that applies different processing strategies to different features based on their characteristics. This approach allows for more specialized handling of each feature, improving model performance on complex, heterogeneous datasets.</p>
</div>

## 🚀 Basic Usage

<div class="code-container">

```python
from kdp import PreprocessingModel, FeatureType

# Define features
features = {
    "age": FeatureType.FLOAT_NORMALIZED,
    "income": FeatureType.FLOAT_RESCALED,
    "occupation": FeatureType.STRING_CATEGORICAL,
    "purchase_history": FeatureType.FLOAT_NORMALIZED,
}

# Create preprocessor with Feature MoE
preprocessor = PreprocessingModel(
    path_data="data.csv",
    features_specs=features,
    use_feature_moe=True,               # Turn on the magic
    feature_moe_num_experts=4,          # Four specialized experts
    feature_moe_expert_dim=64           # Size of expert representations
)

# Build and use
result = preprocessor.build_preprocessor()
model = result["model"]
```

</div>

## 🧩 How Feature MoE Works

<div class="architecture-diagram">
  <img src="imgs/feature_moe.png" alt="Feature MoE Architecture" class="architecture-image">
  <div class="diagram-caption">
    <p>KDP's Feature MoE uses a "divide and conquer" approach with smart routing: each expert is a specialized neural network, a router determines which experts should process each feature, features can use multiple experts with different weights, and residual connections preserve original feature information.</p>
  </div>
</div>

## ⚙️ Configuration Options

<div class="table-container">
  <table class="config-table">
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Description</th>
        <th>Default</th>
        <th>Recommended Range</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>feature_moe_num_experts</code></td>
        <td>Number of specialists</td>
        <td>4</td>
        <td>3-5 for most tasks, 6-8 for very complex data</td>
      </tr>
      <tr>
        <td><code>feature_moe_expert_dim</code></td>
        <td>Size of expert output</td>
        <td>64</td>
        <td>Larger (96-128) for complex patterns</td>
      </tr>
      <tr>
        <td><code>feature_moe_routing</code></td>
        <td>How to assign experts</td>
        <td>"learned"</td>
        <td>"learned" for automatic, "predefined" for control</td>
      </tr>
      <tr>
        <td><code>feature_moe_sparsity</code></td>
        <td>How many experts each feature may use. Setting it to <code>feature_moe_num_experts</code> routes densely.</td>
        <td>2</td>
        <td>1-3 (lower = faster, higher = more accurate)</td>
      </tr>
      <tr>
        <td><code>feature_moe_hidden_dims</code></td>
        <td>Expert network size</td>
        <td>[64, 32]</td>
        <td>Deeper for complex relationships</td>
      </tr>
      <tr>
        <td><code>feature_moe_assignments</code></td>
        <td>Feature &rarr; expert index map, required by <code>"predefined"</code> routing</td>
        <td>None</td>
        <td>Group related features onto the same expert index</td>
      </tr>
      <tr>
        <td><code>feature_moe_dropout</code></td>
        <td>Dropout applied inside every expert network</td>
        <td>0.1</td>
        <td>0.0-0.3 (raise it when experts overfit)</td>
      </tr>
      <tr>
        <td><code>feature_moe_freeze_experts</code></td>
        <td>Freeze expert weights so only the router trains</td>
        <td>False</td>
        <td>True when reusing pretrained experts</td>
      </tr>
      <tr>
        <td><code>feature_moe_use_residual</code></td>
        <td>Add the original feature back onto its expert output</td>
        <td>True</td>
        <td>Keep True unless you want experts to fully replace the input</td>
      </tr>
    </tbody>
  </table>
</div>

## 🎛️ Steering the Experts

The router is not the only control you have. Four extra parameters decide *how*
experts are assigned, trained and combined.

### Hand-picked routing

Set `feature_moe_routing="predefined"` and hand KDP a `feature_moe_assignments`
map to place each feature on a specific expert yourself. Features that share an
expert index are processed by the same specialist network.

<div class="code-container">

```python
from kdp import PreprocessingModel, FeatureType

features = {
    "age": FeatureType.FLOAT_NORMALIZED,
    "income": FeatureType.FLOAT_RESCALED,
    "occupation": FeatureType.STRING_CATEGORICAL,
    "education": FeatureType.STRING_CATEGORICAL,
}

preprocessor = PreprocessingModel(
    path_data="data.csv",
    features_specs=features,
    use_feature_moe=True,
    feature_moe_num_experts=2,
    feature_moe_routing="predefined",
    feature_moe_assignments={
        "age": 0,          # expert 0 gets the demographic signals
        "education": 0,
        "income": 1,       # expert 1 gets the financial ones
        "occupation": 1,
    },
)

result = preprocessor.build_preprocessor()
```

</div>

The map has to be complete. KDP raises a `ValueError` naming the gaps if a
routed feature has no expert, because the assignment matrix doubles as the
router's weights &mdash; an unassigned feature would be multiplied by zero and
vanish from the model. Expert indices are range-checked too, and an index may
be a weight map (`{0: 0.7, 1: 0.3}`) to split one feature across experts.

In `concat` output mode only numeric and categorical features reach the
mixture, so those are the names the map may contain; the error lists the exact
set if you name something else.

### Regularising, freezing and residuals

<div class="code-container">

```python
from kdp import PreprocessingModel, FeatureType

features = {
    "age": FeatureType.FLOAT_NORMALIZED,
    "income": FeatureType.FLOAT_RESCALED,
    "occupation": FeatureType.STRING_CATEGORICAL,
}

preprocessor = PreprocessingModel(
    path_data="data.csv",
    features_specs=features,
    use_feature_moe=True,
    feature_moe_num_experts=3,
    feature_moe_dropout=0.2,        # dropout inside every expert network
    feature_moe_freeze_experts=False,  # True keeps expert weights fixed
    feature_moe_use_residual=True,  # add the input back onto the expert output
)

result = preprocessor.build_preprocessor()
```

</div>

<div class="table-container">
  <table class="config-table">
    <thead>
      <tr>
        <th>Parameter</th>
        <th>What it changes</th>
        <th>Reach for it when</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>feature_moe_dropout</code></td>
        <td>Inserts a <code>Dropout</code> layer after every hidden layer of every expert. <code>0.0</code> removes the layers entirely.</td>
        <td>Experts memorise the training set, or you have many experts and little data.</td>
      </tr>
      <tr>
        <td><code>feature_moe_freeze_experts</code></td>
        <td>Marks every expert non-trainable, so gradients only reach the router and the surrounding layers.</td>
        <td>You loaded pretrained experts, or you want the router to settle before fine-tuning.</td>
      </tr>
      <tr>
        <td><code>feature_moe_use_residual</code></td>
        <td>Adds the untouched feature representation onto the expert output. Applied per feature, only where the two widths already match.</td>
        <td>Almost always &mdash; it keeps the original signal reachable. Turn it off when experts should fully replace their input.</td>
      </tr>
    </tbody>
  </table>
</div>

!!! note "Residuals need matching widths"
    The residual is a plain `Add`, so it only fires for features whose
    preprocessed width already equals `feature_moe_expert_dim`. Features of a
    different width pass through the expert output alone &mdash; no error, no
    silent reshape.

### Reading the routing back

`get_expert_assignments()` on the mixture layer reports, per feature, how much
of it each expert handles. Predefined routing answers from the map you gave it;
learned routing decides from the data, so hand it a batch.

<div class="code-container">

```python
moe = preprocessor.model.get_layer("feature_moe_concat")

# Predefined routing: the map you supplied, normalised to weights.
print(moe.get_expert_assignments())
# {"age": {0: 1.0}, "income": {1: 1.0}, ...}
```

</div>

With learned routing, pass the stacked features the layer sees. Each row keeps
only the experts with a non-zero share, so a run with `feature_moe_sparsity=2`
lists two experts per feature and their weights sum to one.

!!! warning "Save and reload with `output_mode="concat"`"
    A Feature MoE model built with `output_mode="dict"` does not survive a
    save/load round trip: the per-feature projection layers are written in one
    order and read back in another, so two features come back holding each
    other's weights. The model is correct in memory, and `concat` mode round
    trips exactly -- use it if the model has to be saved and reloaded. KDP logs
    a warning when you build the affected combination.

## 💡 Pro Tips for Feature MoE

<div class="pro-tips-grid">
  <div class="pro-tip-card">
    <h3>Group Similar Features</h3>
    <p>Assign related features to the same expert for consistent processing, like grouping demographic, financial, product, and temporal features to different experts.</p>
  </div>

  <div class="pro-tip-card">
    <h3>Visualize Expert Assignments</h3>
    <p>Read the routing off the layer with <code>get_expert_assignments()</code> and plot it as a heatmap. See the section below for the call.</p>
  </div>

  <div class="pro-tip-card">
    <h3>Progressive Training</h3>
    <p>Start with frozen experts, then fine-tune to allow the model to learn basic patterns before specializing.</p>
  </div>
</div>

## 🔍 When to Use Feature MoE

<div class="use-cases-container">
  <div class="use-case-card">
    <h3>Heterogeneous Features</h3>
    <p>When your features have very different statistical properties (categorical, text, numerical, temporal).</p>
  </div>

  <div class="use-case-card">
    <h3>Complex Multi-Modal Data</h3>
    <p>When features come from different sources or modalities (user features, item features, interaction features).</p>
  </div>

  <div class="use-case-card">
    <h3>Transfer Learning</h3>
    <p>When adapting a model to new features with domain-specific experts for different feature groups.</p>
  </div>
</div>

## 🔗 Related Topics

<div class="related-topics">
  <a href="distribution-aware-encoding.md" class="topic-link">
    <span class="topic-icon">🔄</span>
    <span class="topic-text">Distribution-Aware Encoding</span>
  </a>
  <a href="numerical-embeddings.md" class="topic-link">
    <span class="topic-icon">🧮</span>
    <span class="topic-text">Advanced Numerical Embeddings</span>
  </a>
  <a href="tabular-attention.md" class="topic-link">
    <span class="topic-icon">👁️</span>
    <span class="topic-text">Tabular Attention</span>
  </a>
  <a href="../optimization/feature-selection.md" class="topic-link">
    <span class="topic-icon">🎯</span>
    <span class="topic-text">Feature Selection</span>
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
  background: linear-gradient(135deg, #ff9800 0%, #ffca28 100%);
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
  border-left: 4px solid #ff9800;
}

.overview-card p {
  margin: 0;
  font-size: 16px;
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
  background-color: #fff3e0;
  padding: 15px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #ff9800;
}

.config-table td {
  padding: 12px 15px;
  border-bottom: 1px solid #eaecef;
}

.config-table tr:nth-child(even) {
  background-color: #f8f9fa;
}

.config-table tr:hover {
  background-color: #fff3e0;
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
  color: #ff9800;
}

/* Use cases */
.use-cases-container {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.use-case-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.use-case-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.use-case-card h3 {
  margin-top: 0;
  color: #ff9800;
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
  background-color: #fff3e0;
  border-radius: 8px;
  text-decoration: none;
  color: #333;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  transition: background-color 0.3s ease, transform 0.3s ease;
}

.topic-link:hover {
  background-color: #ffe0b2;
  transform: translateY(-2px);
}

.topic-icon {
  font-size: 1.2em;
  margin-right: 10px;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .pro-tips-grid,
  .use-cases-container {
    grid-template-columns: 1fr;
  }

  .related-topics {
    flex-direction: column;
  }
}
</style>
