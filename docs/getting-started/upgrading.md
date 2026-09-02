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
| Feature MoE in `concat` mode | Only `feature_moe_num_experts`, `feature_moe_expert_dim` and `feature_moe_routing` reached the layer; `feature_moe_hidden_dims`, `feature_moe_sparsity`, `feature_moe_freeze_experts` and `feature_moe_dropout` were dropped on the floor | Every option reaches the mixture, so a model that set them now has the network it asked for |
| `feature_moe_use_residual` | Stored on the model and read nowhere, so the residual connection it names never existed | The original feature is added back onto its expert output wherever the widths match |
| `feature_moe_freeze_experts` | Passed `training=False` to the experts, which only changes dropout and batch norm; the weights still received gradients | The experts are marked non-trainable, which is what the option says |
| `feature_moe_sparsity` | The top-k mask that enforces it sat behind `routing_activation != "softmax"`, which `PreprocessingModel` never exposes, so every feature was routed densely to every expert | Each feature reaches at most `feature_moe_sparsity` experts, as documented. With the defaults (4 experts, sparsity 2) a learned-routing model produces different values than it did |

!!! warning "Re-train downstream models"
    If you trained a model on top of KDP output from 1.11.x and any of the
    rows above apply to your feature set, the preprocessor now feeds it
    different numbers. Re-fit before you compare metrics.

## 🚫 Configurations that are now rejected

Two configurations used to be accepted and then silently do the wrong thing.
They raise now, with a message naming what to fix.

<div class="code-container">

```python
from kdp import PreprocessingModel, FeatureType

features = {
    "age": FeatureType.FLOAT_NORMALIZED,
    "income": FeatureType.FLOAT_RESCALED,
}

# Predefined MoE routing with an incomplete map. "income" had no expert, so
# its routing row was all zeros and the feature was multiplied away -- the
# model built, ran, and silently dropped the column.
PreprocessingModel(
    path_data="data.csv",
    features_specs=features,
    use_feature_moe=True,
    feature_moe_routing="predefined",
    feature_moe_assignments={"age": 0, "income": 1},  # every feature, or ValueError
)
```

</div>

An expert index outside the mixture, or a name the mixture never sees, is
rejected the same way.

## 🔤 `TextVectorizerOutputOptions`

`kdp.features` and `kdp.processor` each defined a class of this name.  The
`kdp.processor` one held the strings Keras accepts; the `kdp.features` one held
`auto()` integers, so `TextFeature(output_mode=TextVectorizerOutputOptions.TF_IDF)`
produced `1` and matched nothing. There is one class now, exported from `kdp`,
whose members are the strings:

<div class="code-container">

```python
from kdp import TextVectorizerOutputOptions

TextVectorizerOutputOptions.TF_IDF == "tf_idf"      # True, was False
```

</div>

If you compared against `.value` and got `1`, `2` or `3`, that comparison now
sees `"tf_idf"`, `"int"` and `"multi_hot"`.

## 📐 `DistributionType`

Like `TextVectorizerOutputOptions`, this enum existed twice. The `kdp.features`
copy carried a `WEIBULL` member the encoder has never known, so
`preferred_distribution=DistributionType.WEIBULL` was warned about and replaced
with `"normal"`. There is one class now -- the encoder's own -- and every
member it exposes is a distribution the encoder accepts. `WEIBULL` is gone
rather than silently ignored.

Note also that `BOUNDED`, `ORDINAL` and `POISSON` can be requested explicitly
but are never returned by automatic detection: the detector scores no evidence
for them.

## ✍️ `preferred_distribution`, spelled with two "r"s

`kdp.layers` and the model advisor spell this option `prefered_distribution`,
with one "r", while `NumericalFeature` takes `preferred_distribution`. The
feature swallows `**kwargs`, so the misspelling was discarded and the feature
stayed on automatic detection while looking configured. The advisor's own
recommendation went the same way, and the code snippet it generates never
carried the distribution at all -- so following `auto_configure()` end to end
gave you none of its distribution advice.

Three changes:

- `NumericalFeature` accepts `prefered_distribution` as a deprecated alias and
  warns. Existing code keeps working.
- The advisor's recommendation now uses the key `preferred_distribution`. If
  you read `recommendation["config"]["prefered_distribution"]`, read the
  correctly spelled key instead.
- The generated snippet writes `preferred_distribution=...` onto the feature.

<div class="code-container">

```python
from kdp.features import FeatureType, NumericalFeature

NumericalFeature(
    name="revenue",
    feature_type=FeatureType.FLOAT_RESCALED,
    preferred_distribution="log_normal",   # two "r"s
)
```

</div>

## 🕳️ `MissingValueHandlerLayer` and NaN

Missing values were found with `inputs == mask_value`, and NaN compares equal
to nothing -- not even itself -- so a series carrying the marker pandas and
numpy actually use passed through untouched, and those NaNs then poisoned
every statistic computed from it. No value of `mask_value` could select them.
Setting `mask_value` to NaN now does:

<div class="code-container">

```python
from kdp.layers.time_series.missing_value_handler_layer import (
    MissingValueHandlerLayer,
)

MissingValueHandlerLayer(mask_value=float("nan"), strategy="linear_interpolation")
```

</div>

A sentinel such as `0.0` behaves exactly as before.

## 📊 `FeatureMoE.get_expert_assignments()`

It returned an empty dictionary for learned routing -- the default -- so the
documented way to see which expert handles which feature reported nothing. It
answers for both routing modes now, as `{feature: {expert_index: weight}}`.
Learned routing decides from the data, so pass it a batch:

<div class="code-container">

```python
moe = preprocessor.model.get_layer("feature_moe_concat")

moe.get_expert_assignments()            # predefined routing
moe.get_expert_assignments(batch)       # learned routing
```

</div>

Predefined routing used to return the map you passed in, unchanged. It now
returns the same information as weights: `{"age": 0}` reads back as
`{"age": {0: 1.0}}`.

## 🆕 Methods the documentation promised

`get_timing_metrics()`, `get_memory_usage()` and `plot_model()` appeared in the
docs and did not exist. They do now:

<div class="code-container">

```python
preprocessor.build_preprocessor()

print(preprocessor.get_timing_metrics()["total_seconds"])
print(preprocessor.get_memory_usage()["peak_mb"])
preprocessor.plot_model("architecture.png")   # needs pydot and Graphviz
```

</div>

`get_feature_importance()`, `plot_feature_importance()`, `get_top_features()`,
`update_statistics()`, `transform()` and `fit()` were documented too and were
never real. The pages that used them now show the working equivalents:
`get_feature_importances()`, an `overwrite_stats=True` rebuild, and calling the
built Keras model on a batch.

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
