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
| `AutoLagSelectionLayer` on multi-channel input | Lags were chosen from the autocorrelation of channel 0 alone, then applied to every channel, so reordering the columns of the same data changed the result | The autocorrelation is averaged over the channels as well as the batch, so every channel contributes and the choice does not depend on column order |
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

## 🧩 Feature MoE routed slices, not features

Three defects compounded here, and together they meant the mixture was rarely
routing what it claimed to.

`processed_features_dims` was written as a flat `{name: dim}` mapping and read
back as a nested one keyed by `"numeric"`/`"categorical"`, so the lookup always
missed. The code then fell back to cutting the concatenated tensor into equal
parts. With features of unequal width -- a normalised float is one column, a
discretised one is ten -- that split lands mid-feature: given widths 1, 1 and
10 it cut 12 columns as `[4, 4, 4]`, so every "feature" the router saw was a
slice spanning several real ones, and `feature_moe_assignments` named the wrong
thing. Nothing raised; the model built, trained and produced numbers.

In `dict` output mode the same mismatch raised instead, on the first batch
rather than at build time: `StackFeaturesLayer` reported the *first* feature's
width whatever the others were, so Keras built a graph the layer could not
execute.

Both are fixed by taking the widths and names from the tensors that were
actually concatenated, and padding features up to a common width before
stacking. Padding is parameter-free, so a model whose features are all the same
width is unchanged; a model with mixed widths was previously routing scrambled
input and now routes real features.

!!! warning "Retrain any model that used Feature MoE with mixed-width features"
    Its inputs were being sliced at the wrong offsets. The numbers coming out
    of the mixture are different now, and correct.

`use_feature_moe` combined with `use_global_numerical_embedding` raises rather
than producing nonsense: the global embedding merges every numeric feature into
one vector, leaving no per-feature slices for the mixture to route.

## 🎲 Feature MoE routing no longer depends on the batch

Learned routing fed a Dense layer the *batch mean* of the features, so the
routing weights changed with whatever rows happened to share a batch. Changing
one row moved every row's output, and the same record scored alone did not
match itself scored in a batch: training and single-record serving disagreed
for reasons nothing in the configuration explained.

Feature-level routing is a property of the feature, not of the data passing
through, so the logits now live in a trainable weight with one row per feature.
Routing is identical for every row and every batch, and still learned.

<div class="code-container">

```python
moe = preprocessor.model.get_layer("feature_moe_concat")

moe.routing_logits          # (num_features, num_experts), trainable
```

</div>

!!! warning "Retrain models that used learned Feature MoE routing"
    The old `router` Dense layer is gone, so its weights cannot be loaded, and
    the routing a trained model learned was a function of its batches.

## 🔢 Advanced numerical embedding on discretised features

`NumericalEmbedding` embeds each column it receives, so a feature arriving one
column wide came back rank 2 and a discretised one -- ten one-hot columns --
came back rank 3. Mixing the two failed in `Concatenate`, and a model built only
from discretised features silently produced a three-dimensional output. The
embedding output is flattened now, so every numeric feature is rank 2:
a discretised feature with `embedding_dim=8` contributes 80 columns rather
than a `(10, 8)` block.

### Dict output mode plus Feature MoE round-trips again

That combination used to come back from a save/load with two features holding
each other's projection weights. `.keras` stores a layer's weights under a name
derived from its class and the order it was built -- `dense`, `dense_1` -- not
the name you gave it, and Keras reorders sibling layers when it rebuilds a
functional graph from config. Dict mode used one `Dense` per feature, so four
siblings were enough to cross them.

The per-feature projections are one `PerFeatureDense` layer now, holding a
kernel and bias per feature. The arithmetic is identical and there is no
sibling to be confused with, so the round trip is exact.

## 🔤 `output_mode="tf_idf"` works

TF-IDF weights each token by how rare it is across documents, and Keras refuses
a vocabulary in that mode without matching `idf_weights` -- which KDP's
statistics never collected. The documented, exported `TF_IDF` mode therefore
could not build at all. The vectorizer is adapted on the column now, which
computes the vocabulary and the weights together from the same data the
statistics came from.

## 🧮 `NumericalFeature(use_embedding=True)` is honoured

The flag was stored on the feature and read nowhere, so a feature asking for its
own embedding came through at its original width. It now turns the embedding on
for that feature alone, the way `use_advanced_numerical_embedding` does for all
of them:

<div class="code-container">

```python
from kdp.features import FeatureType, NumericalFeature

NumericalFeature(
    name="revenue",
    feature_type=FeatureType.FLOAT_NORMALIZED,
    use_embedding=True,
    embedding_dim=6,      # this feature is six columns wide now, not one
)
```

</div>

## 📅 Calendar features read dates, not floats

`calendar_feature_config` on a `TimeSeriesFeature` failed with "Cast string to
float is not supported": the pipeline declared the column float and cast it
before the calendar layer ever saw it. Such a feature declares a string column
now and skips the numeric front of the pipeline. Because one column cannot be
both a date and a number, combining it with `lag_config`,
`rolling_stats_config`, `differencing_config`, `moving_average_config`,
`wavelet_transform_config` or `tsfresh_feature_config` raises -- declare those
as separate features.

## 〰️ `wavelet_transform_config` needs a window

The wavelet layer computed `min(window, time_steps // 2)`, which is `0` for the
one-column input a time series feature hands it, and the empty coefficients
failed to broadcast. It also wrote the window sizes back onto itself, so the
first batch pinned them for good. Both are fixed, and a wavelet with only one
step to work on now raises instead of emitting a constant column of zeros:
combine it with `lag_config`, `rolling_stats_config` or `moving_average_config`
so there is a window to decompose.

## 🎯 Feature importances rank features now

`get_feature_importances()` returned `1.0` for every feature, on every input.
Selection wrapped each feature in its own `VariableSelection` with
`nr_features=1`, and a softmax over one element is `1.0` by definition: the
gating was real, the numbers ranked nothing, and the documentation had to say
so.

A single softmax now scores every selected feature against the others. The
scores sum to one across features and scale each feature's output, so the
importances are shares you can sort. The output width is unchanged -- each
selected feature is still `feature_selection_units` wide.

<div class="code-container">

```python
importances = preprocessor.get_feature_importances(batch)
# {"age": 0.39, "income": 0.35, "city": 0.18, "signup_date": 0.08}
```

</div>

!!! warning "Feature selection changes values"
    Each selected feature is scaled by its score, where before it was
    multiplied by `1.0`. A model trained on the old output should be retrained.

## 🚫 Configurations that are now rejected

These configurations used to be accepted and then silently do the wrong thing.
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

A categorical feature whose column holds no values at all is the other case. Its
vocabulary came back empty and was replaced with `["<UNK>"]`, which for a string
feature encoded every real value to the out-of-vocabulary slot — a constant
column, produced without a word — and for an integer feature failed inside Keras
with `invalid literal for int() with base 10`, naming neither the feature nor
the cause. Both now raise a `ValueError` that names the feature and points at
`CategoryEncodingOptions.HASHING`, which needs no vocabulary.

This check is deliberately narrow. A column of empty strings has the vocabulary
`[""]` and a column with one repeated value has a vocabulary of length one;
neither is empty, and both still build.

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
