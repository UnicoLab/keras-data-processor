# 📅 Date Features

<div class="feature-header">
  <div class="feature-title">
    <h2>Date Features in KDP</h2>
    <p>Turn date strings into cyclical encodings your model can actually learn from.</p>
  </div>
</div>

## 📋 Overview

<div class="overview-card">
  <p>A date is a string until you encode it. KDP parses the column, splits it into year, month, day of month and day of week, and encodes each one <strong>cyclically</strong> &mdash; as a sine/cosine pair &mdash; so December and January sit next to each other rather than at opposite ends of a number line. Optionally it adds a one-hot season.</p>
</div>

## 📝 Basic Usage

The shorthand is enough for most columns:

<div class="code-container">

```python
from kdp import PreprocessingModel, FeatureType

preprocessor = PreprocessingModel(
    path_data="data.csv",
    features_specs={
        "signup_date": FeatureType.DATE,
    },
)
preprocessor.build_preprocessor()
```

</div>

Use the class when you need to set an option:

<div class="code-container">

```python
from kdp import PreprocessingModel
from kdp.features import DateFeature, FeatureType

preprocessor = PreprocessingModel(
    path_data="data.csv",
    features_specs={
        "signup_date": DateFeature(
            name="signup_date",
            feature_type=FeatureType.DATE,
            format="YYYY-MM-DD",   # or "YYYY/MM/DD", each with an optional time
            add_season=True,       # append a 4-dim one-hot season
        ),
    },
)
preprocessor.build_preprocessor()
```

</div>

## ⚙️ Configuration Parameters

`DateFeature` takes exactly two options. Anything else you pass is accepted
and not used, and says so in a warning.

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
        <td><code>format</code></td>
        <td>str</td>
        <td><code>"YYYY-MM-DD"</code></td>
        <td>Layout of the date string. Dates are read as year, then month, then day, separated by <code>-</code> or <code>/</code>, and may be followed by a time -- <code>"%Y-%m-%d"</code>, <code>"%Y/%m/%d"</code>, <code>"%Y-%m-%d %H:%M:%S"</code> and <code>"YYYY-MM-DD"</code> all describe a column this reads. A day-first or month-first format is refused where you write it. <code>date_format</code> is accepted as a synonym.</td>
      </tr>
      <tr>
        <td><code>add_season</code></td>
        <td>bool</td>
        <td><code>False</code></td>
        <td>Append a 4-dimensional one-hot season vector to the encoding.</td>
      </tr>
    </tbody>
  </table>
</div>

!!! warning "Other date options do not exist"
    Earlier documentation listed options such as `add_year`, `add_month`,
    `add_day_of_week`, `add_hour`, `add_is_weekend`, `add_quarter`,
    `cyclical_encoding`, `add_time_since_reference`, `reference_date` and
    `time_since_unit`. None of them are read by KDP. `DateFeature` accepts
    arbitrary keyword arguments, and only `output_format` and `extract` are
    called out in a warning; the rest pass without a word and change nothing.
    Year, month, day of month and day
    of week are **always** extracted and **always** cyclically encoded; that
    is not configurable. For anything beyond that, use a
    [custom preprocessing pipeline](../advanced/custom-preprocessing.md).

## 📐 What You Actually Get

Each date column expands to a fixed-width block of floats:

<div class="table-container">
  <table>
    <thead>
      <tr>
        <th>Configuration</th>
        <th>Output width</th>
        <th>Components</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Default</td>
        <td><strong>8</strong></td>
        <td>year, month, day of month, day of week &mdash; each as a <code>(sin, cos)</code> pair</td>
      </tr>
      <tr>
        <td><code>add_season=True</code></td>
        <td><strong>12</strong></td>
        <td>the 8 above, plus a 4-dim one-hot season</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="code-container">

```python
import tensorflow as tf

# With add_season=True, "2021-06-15" encodes to 12 values:
#   [sin_year, cos_year, sin_month, cos_month,
#    sin_day, cos_day, sin_dow, cos_dow,
#    season_0, season_1, season_2, season_3]
output = preprocessor.model({"signup_date": tf.constant([["2021-06-15"]])})
print(output.shape)   # (1, 12)
```

</div>

### Why cyclical encoding

Month 12 and month 1 are one step apart in reality but eleven apart as
integers. Encoding each component as `(sin, cos)` places them adjacent on a
circle, so a model can learn "end of year rolls into start of year" without
having to memorise the discontinuity.

## 🔗 Combining With Other Features

### Feature selection

Date features participate in learned feature selection:

<div class="code-container">

```python
from kdp import FeatureType, PreprocessingModel

preprocessor = PreprocessingModel(
    path_data="data.csv",
    features_specs={
        "signup_date": FeatureType.DATE,
        "amount": FeatureType.FLOAT_NORMALIZED,
    },
    feature_selection_placement="date",   # or "all_features"
    feature_selection_units=32,
    feature_selection_dropout=0.2,
)
```

</div>

Valid `feature_selection_placement` values are `"none"`, `"numeric"`,
`"categorical"`, `"text"`, `"date"` and `"all_features"`.

### Crossing a date with a categorical

<div class="code-container">

```python
from kdp import FeatureType, PreprocessingModel

preprocessor = PreprocessingModel(
    path_data="data.csv",
    features_specs={
        "signup_date": FeatureType.DATE,
        "channel": FeatureType.STRING_CATEGORICAL,
    },
    feature_crosses=[("signup_date", "channel", 10)],
)
```

</div>

## 🛠️ Going Beyond the Built-in Encoding

Need hour-of-day, a weekend flag, or days since a reference date? Those are
not built in. Supply your own layers with `preprocessors`, which receives the
raw string column:

<div class="code-container">

```python
import keras
from kdp.features import DateFeature, FeatureType

DateFeature(
    name="signup_date",
    feature_type=FeatureType.DATE,
    preprocessors=[MyDateParsingLayer, keras.layers.Dense],
    units=16,           # forwarded to Dense
)
```

</div>

See [Custom Preprocessing Pipelines](../advanced/custom-preprocessing.md) for
how `preprocessors` and forwarded keyword arguments work.

## ⏱️ Dates vs. Time Series

A `DATE` feature encodes **one timestamp per row**, independently. If you need
lags, rolling statistics or differencing across ordered rows, that is a
[Time Series Feature](time_series_features.md) &mdash; where a date column
serves as the `sort_by` key rather than as a feature itself.

## 💡 Practical Notes

<div class="pro-tips-grid">
  <div class="pro-tip-card">
    <h4>Keep dates as strings in your CSV</h4>
    <p>KDP parses the string itself. Pre-converting to epoch integers turns the column numeric and skips date handling entirely.</p>
  </div>
  <div class="pro-tip-card">
    <h4>Match the format exactly</h4>
    <p>Only <code>YYYY-MM-DD</code> and <code>YYYY/MM/DD</code> parse. Normalise other layouts before writing the CSV.</p>
  </div>
  <div class="pro-tip-card">
    <h4>add_season is cheap</h4>
    <p>Four extra dimensions, no statistics needed. Worth enabling when seasonality plausibly matters.</p>
  </div>
  <div class="pro-tip-card">
    <h4>Dates need no statistics pass</h4>
    <p>The encoding is deterministic, so nothing is learned from your data for this column.</p>
  </div>
</div>

## 🔗 Related Topics

<div class="related-topics">
  <a href="time_series_features.md" class="topic-link">📊 Time Series Features</a>
  <a href="cross-features.md" class="topic-link">➕ Cross Features</a>
  <a href="../advanced/custom-preprocessing.md" class="topic-link">🛠️ Custom Preprocessing</a>
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
  background: linear-gradient(135deg, #1976d2 0%, #64b5f6 100%);
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
  border-left: 4px solid #1976d2;
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
  color: #1976d2;
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
  border-left: 4px solid #1976d2;
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
  background-color: #e3f2fd;
  padding: 15px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #1976d2;
}

.config-table td {
  padding: 12px 15px;
  border-bottom: 1px solid #eaecef;
}

.config-table tr:nth-child(even) {
  background-color: #f8f9fa;
}

.config-table tr:hover {
  background-color: #e3f2fd;
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
  color: #1976d2;
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
  color: #1976d2;
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
  color: #1976d2;
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
  background-color: #e3f2fd;
  border-radius: 8px;
  text-decoration: none;
  color: #333;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  transition: background-color 0.3s ease, transform 0.3s ease;
}

.topic-link:hover {
  background-color: #bbdefb;
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
  background-color: #e3f2fd;
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
