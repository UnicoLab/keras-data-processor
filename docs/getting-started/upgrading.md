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
| `TimeSeriesFeature.get_output_dim()` on wavelet, tsfresh or calendar configs | Counted those transforms as columns added to the series. They replace it: the wavelet returns coefficients, tsfresh returns statistics, the calendar returns date components. The declared width matched no configuration at all | Each layer is asked for its own width, so the number matches the tensor the stack produces |
| `DateFeature(add_season=True)` | The season layer ran after the cyclic encoding and read its column 1 — the cosine of the year, `1.0` for every row — as the month, so every date came out winter and the four season columns were constant | The season is read from the month before the encoding replaces it, so the four columns carry the season. The output is the same width as before |
| Any numeric column far from zero | The variance was accumulated in float32 by subtracting the running mean from each raw value, so the arithmetic happened at the magnitude of the values. A column around `1e8` came back with a variance of **minus** 1.6e6 against a true 1.0e6, and `Normalization` then flattened it to a constant | Accumulated in float64 and pooled batch by batch, so IDs, epoch timestamps and amounts in cents standardise to mean 0 and variance 1 like any other column |
| `path_data` pointing at a file | The filename was discarded and every `*.csv` in that directory was read, so `path_data="data/train.csv"` computed its statistics over `test.csv` as well | The named file is the only one read. A directory still reads every CSV in it, and a glob is passed through |
| Grouped time series statistics | Each group was replaced by copies of its own mean before combining, leaving only the variance *between* the group means — 0.73 against a true 643 for two groups of spread 25 | The groups are pooled whole, so the reported variance is the variance of the data |
| `DistributionTransformLayer` — `min-max` | `min_value` and `max_value` were read as the range the data was assumed to be *in* rather than the range to scale it *onto*, so with their defaults of 0 and 1 the transform returned its input untouched. `clip_values` chose between that and the data's own range instead of clipping anything | Each feature's own range is scaled onto `[min_value, max_value]`, and `clip_values` clips the result into it. `min-max` is also one of the candidates `transform_type="auto"` can pick, so an auto-selected column may change too |
| `DistributionTransformLayer` — `log` | `log(x + epsilon)` with no guard on the domain, so any negative value returned **NaN**, which then travelled through every layer and every gradient downstream without a message | Clamped at zero first, the way the neighbouring `sqrt` and `box-cox` already were, so a negative value lands on the floor of the transform rather than outside it |
| `DistributionTransformLayer` — `logit` | Clipped into `[epsilon, 1 - epsilon]`, but the default epsilon of `1e-10` is far below the float32 gap at 1.0, so the upper bound rounded back to exactly 1.0 and every value at or above 1 came out as **+inf** | The margin is at least one float32 step wide, so both ends of the clip bite |
| `DistributionTransformLayer` — `arcsinh` | Spelled out as `log(x + sqrt(x*x + 1))`, which loses the whole negative tail: at `x = -1e6` the square is `1e12`, float32 cannot hold the `+1`, the root is exactly `1e6`, the sum is exactly zero and the logarithm is **-inf** | Computed with `asinh`, which is the same function without the cancellation |
| Text vocabularies | The statistics lowercased and split on whitespace but kept punctuation, while `TextVectorization` strips punctuation before splitting. The vocabulary held `product,` and `it!` while the layer looked up `product` and `it`, so on ordinary prose roughly **half of every sentence fell into the out-of-vocabulary slot** — with the right output width and correct-looking counts | The statistics standardize exactly as the layer does, so the vocabulary collected is the one that gets looked up |
| `DistributionTransformLayer` — `robust-scale` | Median and IQR were taken across the whole tensor | Computed per feature, as the transform is defined |
| `DistributionTransformLayer` — `quantile` | Ranking was global | Ranked per feature |
| Distribution-aware encoding | `preferred_distribution` was accepted and then ignored, so the layer always auto-detected | The requested distribution is honoured |
| Output width | `get_output_dim()` combined dimensions additively for some feature configurations | Combined multiplicatively, matching the tensor the model actually produces |
| Time series + other features | A time series feature could not be combined with any other feature: the warm-up rows `drop_na` removes left that column shorter and `Concatenate` raised | Transforms preserve rows by default, so the documented combinations build |
| Feature MoE in `concat` mode | Only `feature_moe_num_experts`, `feature_moe_expert_dim` and `feature_moe_routing` reached the layer; `feature_moe_hidden_dims`, `feature_moe_sparsity`, `feature_moe_freeze_experts` and `feature_moe_dropout` were dropped on the floor | Every option reaches the mixture, so a model that set them now has the network it asked for |
| `feature_moe_use_residual` | Stored on the model and read nowhere, so the residual connection it names never existed | The original feature is added back onto its expert output wherever the widths match |
| `feature_moe_freeze_experts` | Passed `training=False` to the experts, which only changes dropout and batch norm; the weights still received gradients | The experts are marked non-trainable, which is what the option says |
| `feature_moe_sparsity` | The top-k mask that enforces it sat behind `routing_activation != "softmax"`, which `PreprocessingModel` never exposes, so every feature was routed densely to every expert | Each feature reaches at most `feature_moe_sparsity` experts, as documented. With the defaults (4 experts, sparsity 2) a learned-routing model produces different values than it did |

!!! danger "Statistics change for two common shapes of data"
    If any numeric column sits far from zero — an ID, a UNIX timestamp, an
    amount in cents — its statistics were wrong and the column reached your
    model as a constant. It now carries its signal, which changes that
    column's output completely.

    Separately, if you passed `path_data` a **file** in a directory holding
    other CSVs, the statistics were computed over all of them. If that
    directory held your test set, it was folded into the training statistics.
    Recompute with `overwrite_stats=True`; any `features_stats.json` written
    before this release may be built from the wrong rows.

!!! warning "Re-train downstream models"
    If you trained a model on top of KDP output from 1.11.x and any of the
    rows above apply to your feature set, the preprocessor now feeds it
    different numbers. Re-fit before you compare metrics.

## ➕ `feature_crosses` reaches the output

Every cross was built and then dropped. `_group_features_by_type` looks each
processed feature up in `features_specs`, and a cross is configured with
`feature_crosses` rather than by declaring a column, so the lookup found
nothing and skipped it. In the default `concat` output mode the option did
nothing at all: the model built, ran, and had exactly the width it would have
had with no crosses.

Each cross now appends one column — the bin the `(feature_a, feature_b)` pair
hashes into — alongside the categorical features. **A model that set
`feature_crosses` is wider than it was**, so re-train anything fitted on the
old output.

Two configurations that used to fail later now fail where you wrote them:

- Crossing a float column. `HashedCrossing` hashes raw values and accepts only
  integers and strings; a float built a model that looked complete and raised
  on the first batch it was given. Bucket the column into a categorical one
  first.
- Naming a feature that is not in `features_specs`, which surfaced as a bare
  `KeyError`.

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

## 〰️ `tsfresh_feature_config` needs a window too

The same shape as the wavelet, one section up. A time series feature hands one
column per row over, `window_size` is clamped to the number of steps available,
and every statistic over a window of one is degenerate: the mean, the minimum
and the maximum are all the value itself and the standard deviation is zero. A
`tsfresh_feature_config` asking for four statistics therefore returned the input
column three times plus a column of zeros -- four engineered features carrying
nothing, produced without a word. It raises now, naming the configs that widen
the feature; paired with any of them the statistics match `pandas.rolling`
exactly.

## 🔢 The column order no longer moves between builds

Features are preprocessed in parallel batches, and the concatenated output was
assembled by walking a dict keyed in whichever order the workers happened to
finish. Building the same configuration twice laid the columns out differently
— five distinct layouts in six builds of the same six features. Nothing raised,
and each build's numbers were right for the layout it chose.

The damage lands on anyone who trains a model on one build of the preprocessor
and serves it from another: every feature is read as a different one. Reloading
a saved `.keras` file was never affected, because the order is baked into its
graph.

Columns now follow the order features are declared in — numeric first, then
categorical, then any crosses — and `processed_features_dims` reports the same
order. **If you recorded a column-to-feature mapping from an earlier build,
take it again.**

## 📆 Calendar time series features can be built

The documented way to ask for calendar components is a `TimeSeriesFeature`
whose `calendar_feature_config` names them. Its column holds dates, and
everything it produces is derived from that string by `CalendarFeatureLayer` --
there is no mean or variance to take. The statistics pass fed the column to the
numeric accumulator anyway and died inside a `tf.function` with `Cast string to
double is not supported`, so the whole configuration could not be built. The
layer had tests; the path from `PreprocessingModel` to it did not.

Calendar columns are skipped in the numeric statistics now. With
`normalize=False` every component matches what pandas reads off the same
column; with the default `normalize=True` they are scaled into `[0, 1]`.

## 💾 A `tf_idf` text model can be reloaded

Saving worked; loading raised `object of type 'bool' has no len()`. Keras writes
a `tf_idf` vectorizer's IDF weights as layer variables and, on load, assigns
them to a layer that has no vocabulary yet and therefore no such variable. It
happens with `TextVectorization` alone, with no KDP in the picture. The
vocabulary and the weights are now passed to the constructor, where the config
carries them and loading reads them; the layer computes the same numbers either
way.

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

## 📐 The output is always a table

A transformer block adds a sequence axis to a 2-D input and hands it back. The
`categorical` placement removed it again; `transfo_placement="all_features"`
did not, so that one configuration returned `(rows, 1, width)` while every
other returned `(rows, width)` — enough to break a `Dense` head bolted onto the
preprocessor, decided by a flag that says nothing about shape.

Every configuration now returns `(rows, width)`. If you added a `Reshape` or a
`Flatten` after the preprocessor to work around this, remove it.

## 🧾 `predict()` takes the shapes it documents

Its docstring offers "pandas DataFrame, dict, or TensorFlow dataset". A
DataFrame went straight to Keras, which reads a frame as a block of floats and
raised `could not convert string to float` on the first categorical column. A
dict of flat lists — the natural spelling, and exactly what
`InferenceFormatter` produces — converted to shape `(N,)` where every input is
declared `(N, 1)`, and failed with `as_list() is not defined on an unknown
TensorShape`. Both carried the right values; only the container was wrong.

<div class="code-container">

```python
# All four of these now give the same tensor.
preprocessor.predict(frame)
preprocessor.predict({"age": frame["age"].tolist(), "city": frame["city"].tolist()})
preprocessor.predict({"age": [[35.0]], "city": [["paris"]]})
preprocessor.predict(tf.data.Dataset.from_tensor_slices(dict(frame)).batch(32))
```

</div>

## 🔤 `max_tokens` and `ngrams` on a text feature

Both were accepted and neither could work, because the vocabulary handed to
`TextVectorization` came from the statistics pass, which collects single
standardized words:

- `max_tokens` below the vocabulary the data holds was refused by Keras with
  "Attempted to set a vocabulary larger than the maximum vocab size", so the
  only reason to set the option — asking for a smaller vocabulary — could not
  build. A larger one worked and did nothing.
- `ngrams` asked for word pairs the statistics never recorded, so every n-gram
  landed in the single out-of-vocabulary slot. The output width did not move.

Either option, and a custom `standardize` or `split`, now reads the column
again and lets the layer build the vocabulary it will actually use — the same
thing `output_mode="tf_idf"` already did. **A model that set `ngrams` is wider
than it was.**

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

A date format the parser cannot read is the next. `DateFeature` never read the
format string it was given, so `format="%d-%m-%Y"` was accepted, ignored, and
met again at the first batch as an assertion failure inside a TensorFlow graph
error, far from the line that caused it. Dates are read as year, then month,
then day; anything else raises where you write it. `date_format` is accepted as
a synonym for `format`, and `output_format` and `extract` — which nothing reads
— now say so in a warning.

The same parser gained a capability rather than losing one: a time after the
date is dropped instead of rejected, so `"2021-06-15 13:45:00"` and its ISO
`T` spelling are read as the date they carry. A timestamp column previously had
no way through the layer at all.

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

## 〰️ Time series options that did not do what they said

Two of these raised on every input they were given. The rest were accepted and
changed nothing.

`WaveletTransformLayer(flatten_output=False)` returned an array of zeros —
every coefficient it had just computed, discarded — and then declared a rank-2
shape for the rank-3 tensor it returned, so `set_shape` rejected it before the
zeros could reach anyone. It now returns the same coefficients as
`flatten_output=True`, split out per input channel, and inside a
`TimeSeriesFeature` they are folded back into the feature's columns, so the
width is the same either way.

`TSFreshFeatureLayer(window_size=...)` computed the window count from the raw
`window_size` while the extraction clamped it to the length of the series.
A window longer than the series therefore declared a negative number of
windows and raised `Dimension -1 must be >= 0`. A window longer than the
series now covers all of it, which is one window.

`CalendarFeatureLayer(cyclic_encoding=...)` was accepted, stored and
serialized, and read nowhere: the output is identical whichever value you
pass. The sin/cos components are requested by name, and always were —
`features=["month_sin", "month_cos", "day_of_week_sin", "day_of_week_cos"]`.
The flag is deprecated, warns when you pass it, and still loads.

!!! warning "`month` is not cyclic"
    If you asked for `month` and set `cyclic_encoding=True`, you were getting a
    plain normalised month all along, where December and January sit at
    opposite ends of the range. Add `month_sin` and `month_cos` to `features`
    to get the encoding the flag promised. This widens your output, so re-fit
    anything trained on it.

### A `NaN` at the edge of a series reached the model

No imputation strategy can fill a gap at the very start or end of a series: a
gap at the start has nothing before it to carry forward, a gap at the end
nothing after it to carry back. `MissingValueHandlerLayer` has an `extrapolate`
option, documented and defaulting to `True`, that names exactly this case — and
it was read nowhere. With the default `forward_fill` strategy and
`mask_value=float("nan")`, a leading `NaN` came back untouched and went
straight into the model; `backward_fill` and `linear_interpolation` did the
same at the end.

Those gaps now take the nearest value that is not itself a gap, so no strategy
leaves a missing value behind. A series missing everywhere has nothing to reach
for and becomes zeros. Setting `extrapolate=False` keeps the old behaviour, and
now genuinely means it.

### Options that are accepted and do nothing

Each of these was stored, round-tripped through `get_config`, and read nowhere.
None of them raises — they are documented as inert so nobody plans around
behaviour that is not there:

| Option | Why it does nothing |
|--------|---------------------|
| `WaveletTransformLayer(drop_na=...)` | The transform zero-fills, so no row carries a `NaN` for it to drop |
| `MovingAverageLayer(pad_value=...)` | With `drop_na=False` the leading rows keep their original values instead of being padded |
| `CalendarFeatureLayer(onehot_categorical=...)` | Every requested feature comes back as one numeric column |
| `CalendarFeatureLayer(cyclic_encoding=...)` | See above — the components are requested by name |
| `DistributionAwareEncoder(handle_sparsity=...)` | Sparse data is detected and handled as one of the recognised distributions either way |
| `DateParsingLayer(date_format=...)` | Both `YYYY-MM-DD` and `YYYY/MM/DD` are parsed regardless; the separator is normalised first |
| `FeatureMoE(routing_activation=...)` | Routing is always a softmax. For fewer experts per feature, use `sparsity` |

A test now walks every class in `kdp` and fails on a constructor argument that
is stored and never read unless its docstring says so, so this list cannot grow
without someone deciding it should.


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

Files written before this release are also missing the `min`, `max`,
`skewness` and `kurtosis` entries the model advisor reads; a stats file without
them still builds a preprocessor, but the advisor falls back to its neutral
defaults. Pass `overwrite_stats=True` once to collect them.

KDP now detects the byte-repr format and recomputes from the data instead of
trusting it, logging a warning that names the file. You do not have to do anything,
but be aware that:

- the first build after upgrading re-reads your dataset, so it takes as long
  as a fresh statistics pass;
- `path_data` must still point at the data for that recomputation to happen.

To recompute deliberately instead, pass `overwrite_stats=True` once.

## 🕳️ Missing values reaching `predict()`

A frame with a gap in it raised `Invalid dtype: object` — pandas gives a column
mixing `NaN` with strings the object dtype, and Keras refuses that outright.
`predict()` now takes a DataFrame apart the way `InferenceFormatter` does,
deciding each column's type from the values that are actually there, so the two
paths agree.

What each kind of gap becomes:

| Column | A missing value becomes |
|--------|-------------------------|
| Categorical, text | the empty string, which every vocabulary layer maps to its out-of-vocabulary slot |
| Numeric | `NaN`, which normalization carries through to that one column of that one row |
| Date | refused — see below |

The `NaN` is deliberate: the statistics know the column's mean, but filling it
in would hide a hole in your data behind a plausible number. It stays inside
the feature it came from — the rest of the row, and the rest of the batch, are
unaffected — so you can find it with `np.isfinite` on the output. Fill the
column yourself before preprocessing if you want something else.

A date is the exception: there is no neutral date, so a value the parser cannot
read stops the batch rather than being invented. It stopped it before too, as a
TensorFlow graph error naming an internal assertion node; the message now says
what was expected, that an empty value counts as invalid, and prints the value
that failed. Fill or drop those rows before preprocessing.

## 📈 The advisor can tell distributions apart

`ModelAdvisor` reads `skewness`, `kurtosis`, `min` and `max` off each numeric
feature, and the statistics carried none of them. Every column arrived with the
neutral defaults — skew 0, kurtosis 3 — and came back as "Normal distribution
detected, standard normalization recommended" whatever shape it had, while the
rescaling factor derived from `min` and `max` always worked out to exactly 1.

The statistics now collect all four, pooled batch by batch in float64 like the
mean and variance, so the advisor's existing logic reaches its heavy-tailed,
log-normal and uniform branches for the first time. The generated code snippet
also carries a `path_data` line: without it, pasting the advice raised, because
every configuration the advisor recommends needs the data to build.

## ⚠️ Columns float32 cannot hold

Everything after the CSV reader is float32, which carries about seven
significant digits. A column whose values sit far from zero and vary only
slightly loses that variation on the way in: Unix timestamps in seconds are 128
apart in float32 near `1.6e9`, so a column spread over half a minute arrives as
one or two distinct numbers. Normalization then works perfectly on what is
left, reports a standard deviation of 1.0, and hands the model a constant.

The loss happens before any statistic is computed and cannot be undone there,
so this is a warning naming the column, not a change in behaviour. Subtract a
reference point before training, or declare the column as a `DATE` feature.

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
