# ⬆️ Upgrading from 1.11.x

<div class="feature-header">
  <div class="feature-title">
    <h2>Upgrade Guide</h2>
    <p>What changes when you move off 1.11.x, and what you need to do about it.</p>
  </div>
</div>

## 📋 Overview

<div class="overview-card">
  <p>This release repairs defects that produced <strong>wrong numbers rather than errors</strong>. Pipelines that ran without complaint on 1.11.x will keep running, but several of them will now emit different values &mdash; correct ones. Re-train any downstream model that was fitted on the old output.</p>
</div>

## ⚠️ Output values change

These transforms returned incorrect values before. Nothing about your
configuration changes, but the numbers coming out of the preprocessor do.

| Area | Before | Now |
|------|--------|-----|
| Time series layers | `DifferencingLayer`, `MovingAverageLayer`, `RollingStatsLayer`, `LagFeatureLayer` and `AutoLagSelectionLayer` returned fixed constants for certain shapes instead of computing over the input | Values are computed from the data in every case |
| `DistributionTransformLayer` — `robust-scale` | Median and IQR were taken across the whole tensor | Computed per feature, as the transform is defined |
| `DistributionTransformLayer` — `quantile` | Ranking was global | Ranked per feature |
| Distribution-aware encoding | `preferred_distribution` was accepted and then ignored, so the layer always auto-detected | The requested distribution is honoured |
| Output width | `get_output_dim()` combined dimensions additively for some feature configurations | Combined multiplicatively, matching the tensor the model actually produces |
| Time series + other features | A time series feature could not be combined with any other feature: the warm-up rows `drop_na` removes left that column shorter and `Concatenate` raised | Transforms preserve rows by default, so the documented combinations build |

!!! warning "Re-train downstream models"
    If you trained a model on top of KDP output from 1.11.x and any of the
    rows above apply to your feature set, the preprocessor now feeds it
    different numbers. Re-fit before you compare metrics.

## 🗂️ Existing `features_stats.json` files

Statistics files written by 1.11.2 and earlier stored categorical
vocabularies as byte reprs — the literal string `"b'paris'"` rather than
`paris`. Such a file loaded without complaint and produced a model in which
**every category missed the vocabulary and encoded identically to an unseen
value**.

KDP now detects that format and recomputes from the data instead of trusting
it, logging a warning that names the file. You do not have to do anything,
but be aware that:

- the first build after upgrading re-reads your dataset, so it takes as long
  as a fresh statistics pass;
- `path_data` must still point at the data for that recomputation to happen.

To recompute deliberately instead, pass `overwrite_stats=True` once.

## 🐍 Requirements

<div class="code-container">

```
Python  >= 3.10        # 1.11.x declared 3.9, but never ran on it
keras   >= 3.5, < 4.0  # now an explicit dependency
```

</div>

The package previously reached Keras through the `tf.keras` compatibility
shim. It targets the Keras 3 API directly, which means it is no longer tied
to whichever Keras version TensorFlow happens to pin, and it is tested
against the newest Keras 3 release on every pull request. The `tf-keras`
dependency is gone.

The 3.9 requirement in 1.11.x was a metadata error rather than a supported
configuration — the source already used `X | None` type syntax, which 3.9
cannot parse — so no working 3.9 installation exists to break.

## ✅ Things that now work that did not before

None of these need a change on your side; they simply stop failing.

<div class="code-container">

```python
# Saving and reloading a preprocessor. Custom layers are registered as
# serializable, so a saved model round-trips.
preprocessor.model.save("model.keras")
reloaded = keras.models.load_model("model.keras")

# predict() with plain Python or NumPy input, including string columns.
preprocessor.predict({
    "age": [[35.0]],
    "city": [["paris"]],
})

# Custom pipelines by layer name -- the spelling the docs lead with.
Feature(
    name="value",
    feature_type=FeatureType.FLOAT_NORMALIZED,
    preprocessors=["Rescaling", "Dense"],
    scale=2.0,
    units=4,
)

# auto_configure() returns real per-feature recommendations rather than an
# empty set, and DatasetStatistics works from features_specs alone.
auto_configure("data.csv", features_specs=specs)
```

</div>

Building a model whose features learn nothing from the data — hashing
categoricals with an explicit `hash_bucket_size` — no longer requires
`path_data`, because no statistics pass runs. When a feature *does* need
statistics and `path_data` is missing, the error now names the parameter
instead of surfacing as a `TypeError` from inside `tf.data`.

## 🔀 Moved symbols

`FeatureSpaceConverter` now lives in `kdp.features`, next to the feature
classes it converts. It is re-exported from `kdp.processor`, so existing
imports keep working:

<div class="code-container">

```python
from kdp.features import FeatureSpaceConverter   # preferred
from kdp.processor import FeatureSpaceConverter  # still works
```

</div>
