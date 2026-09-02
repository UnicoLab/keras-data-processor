# 🏷️ Categorical Features

<div class="feature-header">
  <div class="feature-title">
    <h2>Categorical Features in KDP</h2>
    <p>Turn labels and IDs into dense vectors, one-hot columns, or hashed buckets.</p>
  </div>
</div>

## 📋 Overview

<div class="overview-card">
  <p>KDP learns the vocabulary of a categorical column during the statistics pass, then encodes it one of three ways: a learned <strong>embedding</strong>, a <strong>one-hot</strong> vector, or a <strong>hash</strong> into a fixed number of buckets. Hashing is the only one that needs no vocabulary, so it is also the only one that works without a data pass.</p>
</div>

## 🚀 The Two Categorical Feature Types

<div class="table-container">
  <table>
    <thead>
      <tr>
        <th>Feature type</th>
        <th>Column dtype</th>
        <th>Use for</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>FeatureType.STRING_CATEGORICAL</code></td>
        <td>string</td>
        <td>City names, product codes, channels &mdash; anything written as text.</td>
      </tr>
      <tr>
        <td><code>FeatureType.INTEGER_CATEGORICAL</code></td>
        <td>integer</td>
        <td>IDs and codes that are numbers but have no ordering.</td>
      </tr>
    </tbody>
  </table>
</div>

!!! warning "`MULTI_CATEGORICAL` and `STRING_HASHED` do not exist"
    Earlier documentation listed these as feature types, along with a
    multi-value workflow using `separator` and `multi_hot`. `FeatureType` has
    exactly eleven members and neither of these is among them, so
    `FeatureType.MULTI_CATEGORICAL` raises `AttributeError`. Hashing is not a
    separate feature type either &mdash; it is the `category_encoding` option
    below. To encode a multi-value column, split it into columns yourself or
    use a [custom pipeline](../advanced/custom-preprocessing.md).

## 📝 Basic Usage

<div class="code-container">

```python
from kdp import PreprocessingModel, FeatureType

preprocessor = PreprocessingModel(
    path_data="data.csv",
    features_specs={
        "city": FeatureType.STRING_CATEGORICAL,
        "store_id": FeatureType.INTEGER_CATEGORICAL,
    },
)
preprocessor.build_preprocessor()
```

</div>

## ⚙️ Configuration Parameters

<div class="table-container">
  <table>
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
        <td><code>category_encoding</code></td>
        <td>str</td>
        <td><code>"EMBEDDING"</code></td>
        <td><code>"EMBEDDING"</code>, <code>"ONE_HOT_ENCODING"</code> or <code>"HASHING"</code>.</td>
      </tr>
      <tr>
        <td><code>embedding_size</code></td>
        <td>int</td>
        <td>derived from vocabulary size</td>
        <td>Width of the learned embedding. Also used when hashing with an embedding on top (default 8 there).</td>
      </tr>
      <tr>
        <td><code>hash_bucket_size</code></td>
        <td>int</td>
        <td>derived from vocabulary size</td>
        <td>Number of hash buckets. <strong>Hashing only.</strong> Setting it explicitly removes the need for a statistics pass.</td>
      </tr>
      <tr>
        <td><code>salt</code></td>
        <td>int</td>
        <td><code>None</code></td>
        <td>Seed for the hash, so two columns hash differently. <strong>Hashing only.</strong></td>
      </tr>
      <tr>
        <td><code>hash_with_embedding</code></td>
        <td>bool</td>
        <td><code>False</code></td>
        <td>Put a learned embedding on top of the hash instead of a multi-hot vector. <strong>Hashing only.</strong></td>
      </tr>
    </tbody>
  </table>
</div>

!!! warning "Options that are silently ignored"
    `embedding_dim` (the real name is `embedding_size`), `vocabulary_size`,
    `max_vocabulary_size`, `use_embedding`, `unknown_token`, `oov_buckets`,
    `multi_hot`, `separator`, `pretrained_embeddings`, `multi_hash` and
    `num_hash_functions` are **not read by KDP**. `CategoricalFeature` accepts
    any keyword argument, so passing them looks like it works and changes
    nothing. Out-of-vocabulary values are always routed to a single reserved
    slot; that is not configurable.

## 🎛️ The Three Encodings

### Embedding (default)

<div class="code-container">

```python
from kdp.features import CategoricalFeature, CategoryEncodingOptions, FeatureType

CategoricalFeature(
    name="city",
    feature_type=FeatureType.STRING_CATEGORICAL,
    category_encoding=CategoryEncodingOptions.EMBEDDING,
    embedding_size=16,
)
```

</div>

Leave `embedding_size` unset and KDP derives it from the vocabulary size.

### One-hot

<div class="code-container">

```python
CategoricalFeature(
    name="status",
    feature_type=FeatureType.STRING_CATEGORICAL,
    category_encoding=CategoryEncodingOptions.ONE_HOT_ENCODING,
)
```

</div>

Output width is the vocabulary size. Good for a handful of categories, wasteful
beyond a few dozen.

### Hashing

<div class="code-container">

```python
CategoricalFeature(
    name="user_id",
    feature_type=FeatureType.STRING_CATEGORICAL,
    category_encoding=CategoryEncodingOptions.HASHING,
    hash_bucket_size=1024,
    salt=42,                    # optional, decorrelates two hashed columns
    hash_with_embedding=True,   # embedding instead of multi-hot
    embedding_size=16,          # width of that embedding
)
```

</div>

Hashing maps values into a fixed bucket count, so it handles unbounded
cardinality and unseen values without growing. Collisions are the trade-off.

!!! tip "Hashing can skip the statistics pass entirely"
    When **every** feature is a hashing categorical with an explicit
    `hash_bucket_size`, nothing has to be learned from your data, so
    `path_data` is not required and `build_preprocessor()` runs immediately.
    Omit `hash_bucket_size` and the bucket count is derived from the
    vocabulary, which does require a data pass.

<div class="code-container">

```python
# Builds with no dataset at all
preprocessor = PreprocessingModel(
    features_specs={
        "user_id": CategoricalFeature(
            name="user_id",
            feature_type=FeatureType.STRING_CATEGORICAL,
            category_encoding=CategoryEncodingOptions.HASHING,
            hash_bucket_size=32,
        ),
    },
)
preprocessor.build_preprocessor()
```

</div>

## 📐 Output Widths

<div class="table-container">
  <table>
    <thead>
      <tr>
        <th>Encoding</th>
        <th>Output width</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Embedding</td>
        <td><code>embedding_size</code> (derived if unset)</td>
      </tr>
      <tr>
        <td>One-hot</td>
        <td>vocabulary size</td>
      </tr>
      <tr>
        <td>Hashing, multi-hot (default)</td>
        <td><code>hash_bucket_size</code></td>
      </tr>
      <tr>
        <td>Hashing with embedding</td>
        <td><code>embedding_size</code> (default 8)</td>
      </tr>
    </tbody>
  </table>
</div>

## 🔗 Combining With Other Features

### Crossing two categoricals

<div class="code-container">

```python
preprocessor = PreprocessingModel(
    path_data="data.csv",
    features_specs={
        "city": FeatureType.STRING_CATEGORICAL,
        "channel": FeatureType.STRING_CATEGORICAL,
    },
    feature_crosses=[("city", "channel", 10)],
)
```

</div>

### Feature selection and attention

<div class="code-container">

```python
preprocessor = PreprocessingModel(
    path_data="data.csv",
    features_specs={"city": FeatureType.STRING_CATEGORICAL},
    feature_selection_placement="categorical",   # or "all_features"
    tabular_attention=True,
    tabular_attention_placement="categorical",
    transfo_nr_blocks=2,                         # transformer over categoricals
    transfo_placement="categorical",
)
```

</div>

## 💎 Practical Notes

<div class="pro-tips-grid">
  <div class="pro-tip-card">
    <h4>Reach for hashing on high cardinality</h4>
    <p>User and session IDs blow up a vocabulary. A fixed bucket count keeps the model the same size no matter how many values appear.</p>
  </div>
  <div class="pro-tip-card">
    <h4>One-hot only for small vocabularies</h4>
    <p>Width equals vocabulary size, so it grows linearly with the number of categories.</p>
  </div>
  <div class="pro-tip-card">
    <h4>Salt when you hash two columns</h4>
    <p>Without different salts, the same value in two columns lands in the same bucket and the model cannot tell them apart.</p>
  </div>
  <div class="pro-tip-card">
    <h4>Integer IDs are not numbers</h4>
    <p>Use <code>INTEGER_CATEGORICAL</code>, not a float type &mdash; otherwise the model reads store 7 as greater than store 3.</p>
  </div>
</div>

## 🔗 Related Topics

<div class="related-topics">
  <a href="../examples/categorical-hashing-example.md" class="topic-link">🧮 Categorical Hashing</a>
  <a href="cross-features.md" class="topic-link">➕ Cross Features</a>
  <a href="text-features.md" class="topic-link">📝 Text Features</a>
  <a href="overview.md" class="topic-link">🛠️ Features Overview</a>
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
  background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
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
  border-left: 4px solid #6a11cb;
}

.overview-card p {
  margin: 0;
  font-size: 16px;
}

/* Tables */
.table-container {
  margin: 30px 0;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}

.features-table, .config-table {
  width: 100%;
  border-collapse: collapse;
}

.features-table th, .config-table th {
  background-color: #f0f0ff;
  padding: 15px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #6a11cb;
}

.features-table td, .config-table td {
  padding: 12px 15px;
  border-bottom: 1px solid #eaecef;
}

.features-table tr:nth-child(even), .config-table tr:nth-child(even) {
  background-color: #f8f9fa;
}

.features-table tr:hover, .config-table tr:hover {
  background-color: #f0f0ff;
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

/* Advanced section */
.advanced-section {
  background-color: #f8f9fa;
  border-radius: 10px;
  padding: 20px;
  margin: 30px 0;
  border-left: 4px solid #6a11cb;
}

.advanced-section p {
  margin-top: 0;
}

/* Power features */
.power-features {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.power-feature-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.power-feature-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.power-feature-card h3 {
  margin-top: 0;
  color: #6a11cb;
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
  color: #6a11cb;
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
  color: #6a11cb;
}

.pro-tip-card p {
  margin-bottom: 10px;
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

.diagram-caption {
  margin-top: 20px;
  text-align: center;
  font-style: italic;
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
  background-color: #f0f0ff;
  border-radius: 8px;
  text-decoration: none;
  color: #333;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  transition: background-color 0.3s ease, transform 0.3s ease;
}

.topic-link:hover {
  background-color: #e0e0ff;
  transform: translateY(-2px);
}

.topic-icon {
  font-size: 1.2em;
  margin-right: 10px;
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
  background-color: #f0f0ff;
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
  .power-features,
  .examples-container,
  .pro-tips-grid {
    grid-template-columns: 1fr;
  }

  .related-topics {
    flex-direction: column;
  }
}
</style>
