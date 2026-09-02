# 📝 Text Features

<div class="feature-header">
  <div class="feature-title">
    <h2>Text Features in KDP</h2>
    <p>Vocabulary-based text vectorization, learned from your own data.</p>
  </div>
</div>

## 📋 Overview

<div class="overview-card">
  <p>KDP builds a vocabulary from the text column during the statistics pass, then encodes each row with Keras <code>TextVectorization</code> against that vocabulary. Everything is learned from <em>your</em> corpus &mdash; there are no downloaded embeddings and no external model weights.</p>
</div>

## 📝 Basic Usage

<div class="code-container">

```python
from kdp import PreprocessingModel, FeatureType

preprocessor = PreprocessingModel(
    path_data="reviews.csv",
    features_specs={
        "review_text": FeatureType.TEXT,
    },
)
preprocessor.build_preprocessor()
```

</div>

By default a text column becomes a **35-token integer sequence**, padded with
zeros.

## ⚙️ Configuration Parameters

`TextFeature` forwards its keyword arguments to Keras `TextVectorization`,
apart from `stop_words`, which KDP applies itself beforehand.

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
        <td><code>stop_words</code></td>
        <td>list[str]</td>
        <td><code>[]</code></td>
        <td>Words stripped before vectorization, by KDP's own text preprocessing layer.</td>
      </tr>
      <tr>
        <td><code>output_sequence_length</code></td>
        <td>int</td>
        <td><code>35</code></td>
        <td>Token count per row, and therefore the output width. Applies to <code>output_mode="int"</code> only.</td>
      </tr>
      <tr>
        <td><code>output_mode</code></td>
        <td>str</td>
        <td><code>"int"</code></td>
        <td><code>"int"</code>, <code>"multi_hot"</code> or <code>"count"</code>. See the table below.</td>
      </tr>
      <tr>
        <td><code>max_tokens</code></td>
        <td>int</td>
        <td>&mdash;</td>
        <td>Caps the vocabulary. Must be at least as large as the vocabulary found in your data, or Keras raises.</td>
      </tr>
      <tr>
        <td><code>ngrams</code></td>
        <td>int | tuple</td>
        <td><code>None</code></td>
        <td>Generate n-grams in addition to single tokens.</td>
      </tr>
      <tr>
        <td><code>split</code>, <code>standardize</code></td>
        <td>str | callable</td>
        <td>Keras defaults</td>
        <td>Passed straight through to <code>TextVectorization</code>.</td>
      </tr>
    </tbody>
  </table>
</div>

!!! warning "Pretrained embeddings and attention are not implemented"
    Earlier documentation advertised `use_pretrained`, `pretrained_name`
    (GloVe, word2vec, BERT), `tokenizer`, `use_attention`, `attention_heads`,
    `attention_dropout`, `max_sequence_length`, `embedding_dim` and
    `sequence_length`. **None of these exist.** `TextFeature` accepts any
    keyword without complaint, so they appear to work while changing nothing
    &mdash; verified by comparing model output with and without each one.
    KDP learns its vocabulary from your data; it does not download or load
    pretrained language models. To use one, wrap it yourself with
    [custom preprocessing](../advanced/custom-preprocessing.md).

## 🔤 Output Modes

<div class="table-container">
  <table>
    <thead>
      <tr>
        <th>Mode</th>
        <th>Output width</th>
        <th>What each value means</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>"int"</code> (default)</td>
        <td><code>output_sequence_length</code></td>
        <td>Token index at that position; order is preserved.</td>
      </tr>
      <tr>
        <td><code>"multi_hot"</code></td>
        <td>vocabulary size</td>
        <td>1 if the token appears anywhere in the row, else 0. Order is discarded.</td>
      </tr>
      <tr>
        <td><code>"count"</code></td>
        <td>vocabulary size</td>
        <td>How many times the token appears in the row.</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="code-container">

```python
from kdp import PreprocessingModel
from kdp.features import FeatureType, TextFeature

# Bag-of-words instead of a padded sequence
preprocessor = PreprocessingModel(
    path_data="reviews.csv",
    features_specs={
        "review_text": TextFeature(
            name="review_text",
            feature_type=FeatureType.TEXT,
            output_mode="multi_hot",
        ),
    },
)
preprocessor.build_preprocessor()
```

</div>

!!! note "tf_idf needs weights KDP does not compute"
    `output_mode="tf_idf"` requires an IDF weight array alongside the
    vocabulary. KDP's statistics pass records the vocabulary only, so this mode
    raises a clear Keras error rather than working. Use `"count"` and apply
    your own weighting downstream if you need it.

## 🧹 Stop Words

<div class="code-container">

```python
TextFeature(
    name="review_text",
    feature_type=FeatureType.TEXT,
    stop_words=["the", "a", "an", "and", "or"],
    output_sequence_length=64,
)
```

</div>

Stop words are removed **before** vectorization, so they never enter the
vocabulary and never occupy a token slot.

## 🔗 Combining With Other Features

### N-grams for short text

<div class="code-container">

```python
TextFeature(
    name="product_title",
    feature_type=FeatureType.TEXT,
    ngrams=2,                       # unigrams and bigrams
    output_sequence_length=24,
)
```

</div>

### Feature selection over text

<div class="code-container">

```python
preprocessor = PreprocessingModel(
    path_data="reviews.csv",
    features_specs={
        "review_text": FeatureType.TEXT,
        "rating": FeatureType.FLOAT_NORMALIZED,
    },
    feature_selection_placement="text",   # or "all_features"
)
```

</div>

## 💎 Practical Notes

<div class="pro-tips-grid">
  <div class="pro-tip-card">
    <h4>Sequence length drives width</h4>
    <p>In <code>"int"</code> mode the output is exactly <code>output_sequence_length</code> columns wide. Long default sequences on short text are mostly padding.</p>
  </div>
  <div class="pro-tip-card">
    <h4>Text needs a statistics pass</h4>
    <p>The vocabulary comes from your data, so <code>path_data</code> is required and the column is read end to end.</p>
  </div>
  <div class="pro-tip-card">
    <h4>multi_hot for keyword signals</h4>
    <p>When only presence matters &mdash; tags, short titles &mdash; <code>"multi_hot"</code> is smaller and easier to learn from than a padded sequence.</p>
  </div>
  <div class="pro-tip-card">
    <h4>max_tokens must fit the data</h4>
    <p>Keras raises if the cap is below the vocabulary actually found. Set it generously or leave it unset.</p>
  </div>
</div>

## 🔗 Related Topics

<div class="related-topics">
  <a href="categorical-features.md" class="topic-link">🏷️ Categorical Features</a>
  <a href="../advanced/custom-preprocessing.md" class="topic-link">🛠️ Custom Preprocessing</a>
  <a href="../optimization/feature-selection.md" class="topic-link">🎯 Feature Selection</a>
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
  background: linear-gradient(135deg, #00897b 0%, #4db6ac 100%);
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
  border-left: 4px solid #00897b;
}

.overview-card p {
  margin: 0;
  font-size: 16px;
}

/* Approaches */
.approaches-container {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.approach-card {
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

.approach-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.approach-icon {
  font-size: 2.5em;
  margin-bottom: 15px;
}

.approach-card h3 {
  margin: 0 0 10px 0;
  color: #00897b;
}

.approach-card p {
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

/* Advanced section */
.advanced-section {
  background-color: #f8f9fa;
  border-radius: 10px;
  padding: 20px;
  margin: 30px 0;
  border-left: 4px solid #00897b;
}

.advanced-section p {
  margin-top: 0;
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
  background-color: #e0f2f1;
  padding: 15px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #00897b;
}

.config-table td {
  padding: 12px 15px;
  border-bottom: 1px solid #eaecef;
}

.config-table tr:nth-child(even) {
  background-color: #f8f9fa;
}

.config-table tr:hover {
  background-color: #e0f2f1;
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
  color: #00897b;
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
  color: #00897b;
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
  color: #00897b;
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
  background-color: #e0f2f1;
  border-radius: 8px;
  text-decoration: none;
  color: #333;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  transition: background-color 0.3s ease, transform 0.3s ease;
}

.topic-link:hover {
  background-color: #b2dfdb;
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
  background-color: #e0f2f1;
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
  .approaches-container,
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
