# 🛠️ Custom Preprocessing Pipelines

<div class="intro-container">
  <div class="intro-content">
    <h2>Create specialized preprocessing flows with complete control</h2>
    <p>Design custom transformations for your features when standard preprocessing doesn't meet your specific needs.</p>
  </div>
</div>

## 📋 Overview

<div class="overview-card">
  <p>KDP allows you to define custom preprocessing pipelines for your features, giving you complete control over how each feature is processed before being fed into your model. This is particularly useful when the standard preprocessing options don't meet your specific needs.</p>
</div>

## ✨ Key Benefits

<div class="grid-container">
  <div class="grid-item">
    <span class="feature-icon">🔍</span>
    <h3>Specific Transformations</h3>
    <p>Define custom preprocessing steps not covered by built-in options</p>
  </div>
  <div class="grid-item">
    <span class="feature-icon">🔄</span>
    <h3>Combined Techniques</h3>
    <p>Combine multiple preprocessing techniques in a single pipeline</p>
  </div>
  <div class="grid-item">
    <span class="feature-icon">🧪</span>
    <h3>Domain-Specific</h3>
    <p>Handle specialized data with custom preprocessing logic</p>
  </div>
  <div class="grid-item">
    <span class="feature-icon">🔬</span>
    <h3>Novel Approaches</h3>
    <p>Experiment with new preprocessing methods</p>
  </div>
  <div class="grid-item">
    <span class="feature-icon">🧩</span>
    <h3>Legacy Integration</h3>
    <p>Incorporate existing preprocessing logic</p>
  </div>
</div>

## 🚀 Getting Started

<div class="step-card">
  <div class="step-header">
    <span class="step-number">1</span>
    <h3>Basic Example</h3>
  </div>
  <div class="code-container">

```python
from kdp.features import NumericalFeature, FeatureType
from tensorflow.keras.layers import Normalization, Dense, Activation

# Create a feature with custom preprocessing steps
log_transform_feature = NumericalFeature(
    name="revenue",
    feature_type=FeatureType.FLOAT_NORMALIZED,
    preprocessors=[
        "Lambda",  # Using a standard Keras layer by name
        "Dense",   # Another standard layer
        "ReLU"     # Activation function
    ],
    # Parameters for the layers
    function=lambda x: tf.math.log1p(x),  # For Lambda layer
    units=16,  # For Dense layer
)
```

</div>

## 🤔 When to Use Custom Preprocessing

Consider using custom preprocessing pipelines when:

- 🔍 You need specific transformations not covered by the built-in options
- 🔄 You want to combine multiple preprocessing techniques
- 🧪 You're working with domain-specific data that requires specialized handling
- 🔬 You want to experiment with novel preprocessing approaches
- 🧩 You have legacy preprocessing logic that you want to incorporate

## 📦 Custom Preprocessors with PreprocessingModel

### 🏁 Basic Approach

The simplest way to define custom preprocessing is by specifying a list of preprocessors when creating a feature:

```python
from kdp.features import NumericalFeature, FeatureType
from tensorflow.keras.layers import Normalization, Dense, Activation

# Create a feature with custom preprocessing steps
log_transform_feature = NumericalFeature(
    name="revenue",
    feature_type=FeatureType.FLOAT_NORMALIZED,
    preprocessors=[
        "Lambda",  # Using a standard Keras layer by name
        "Dense",   # Another standard layer
        "ReLU"     # Activation function
    ],
    # Parameters for the layers
    function=lambda x: tf.math.log1p(x),  # For Lambda layer
    units=16,  # For Dense layer
)
```

### 🚀 Advanced Usage with Layer Parameters

For more control, you can provide specific parameters to each preprocessing layer:

```python
from kdp.features import CategoricalFeature, FeatureType
from kdp.layers_factory import PreprocessorLayerFactory

# Advanced categorical feature with custom preprocessing
advanced_categorical = CategoricalFeature(
    name="product_category",
    feature_type=FeatureType.STRING_CATEGORICAL,
    preprocessors=[
        "StringLookup",
        "Embedding",
        "Dropout"
    ],
    # Parameters for layers
    num_oov_indices=2,  # For StringLookup
    input_dim=100,      # For Embedding
    output_dim=32,      # For Embedding
    rate=0.2            # For Dropout
)
```

### 🏭 Using the PreprocessorLayerFactory

For more complex scenarios, you can use the `PreprocessorLayerFactory` directly:

```python
from kdp.features import TextFeature, FeatureType
from kdp.layers_factory import PreprocessorLayerFactory

# Create a text feature with custom preprocessing using the factory
text_feature = TextFeature(
    name="review_text",
    feature_type=FeatureType.TEXT,
    preprocessors=[
        PreprocessorLayerFactory.text_preprocessing_layer,
        "TextVectorization",
        "Embedding"
    ],
    # Parameters
    stop_words=["the", "and", "is"],  # For TextPreprocessingLayer
    max_tokens=10000,                 # For TextVectorization
    output_sequence_length=50,        # For TextVectorization
    output_dim=64                     # For Embedding
)
```

### 🧩 Mixing Built-in and Custom Layers

You can combine KDP's specialized layers with standard Keras layers:

```python
from kdp.features import NumericalFeature, FeatureType
from kdp.layers_factory import PreprocessorLayerFactory

# Mix custom and specialized layers
numeric_feature = NumericalFeature(
    name="transaction_amount",
    feature_type=FeatureType.FLOAT,
    preprocessors=[
        PreprocessorLayerFactory.cast_to_float32_layer,
        "Lambda",
        PreprocessorLayerFactory.distribution_aware_encoder,
        "Dense"
    ],
    # Parameters
    function=lambda x: tf.clip_by_value(x, 0, 1000),  # For Lambda
    num_bins=100,                                     # For DistributionAwareEncoder
    units=32                                          # For Dense
)
```

### 🌟 Advanced KDP Layer Examples

KDP provides several specialized preprocessing layers that you can use for advanced feature processing:

#### 💹 Distribution Transformation Layer

```python
from kdp.features import NumericalFeature, FeatureType
from kdp.layers_factory import PreprocessorLayerFactory

# Create a feature that applies distribution transformations
skewed_feature = NumericalFeature(
    name="highly_skewed_metric",
    feature_type=FeatureType.FLOAT,
    preprocessors=[
        PreprocessorLayerFactory.cast_to_float32_layer,
        PreprocessorLayerFactory.distribution_transform_layer,
    ],
    # Parameters for DistributionTransformLayer
    transform_type="box-cox",  # Apply Box-Cox transformation
    lambda_param=0.5,         # Parameter for Box-Cox
    epsilon=1e-6              # Prevent numerical issues
)

# Automatic transformation selection
auto_transform_feature = NumericalFeature(
    name="unknown_distribution",
    feature_type=FeatureType.FLOAT,
    preprocessors=[
        PreprocessorLayerFactory.cast_to_float32_layer,
        PreprocessorLayerFactory.distribution_transform_layer,
    ],
    # Let the layer choose the best transformation
    transform_type="auto",
    auto_candidates=["log", "sqrt", "box-cox", "yeo-johnson"]
)
```

#### 🧮 Numerical Embeddings

```python
from kdp.features import NumericalFeature, FeatureType
from kdp.layers_factory import PreprocessorLayerFactory

# Create a feature with numerical embedding
embedded_numeric = NumericalFeature(
    name="user_age",
    feature_type=FeatureType.FLOAT,
    preprocessors=[
        PreprocessorLayerFactory.cast_to_float32_layer,
        PreprocessorLayerFactory.numerical_embedding_layer,
    ],
    # Parameters for NumericalEmbedding
    embedding_dim=16,       # Output dimension
    mlp_hidden_units=32,    # MLP hidden units
    num_bins=20,            # Number of bins for discretization
    init_min=18,            # Minimum value for initialization
    init_max=100,           # Maximum value for initialization
    dropout_rate=0.2,       # Dropout rate
    use_batch_norm=True     # Apply batch normalization
)
```

#### 🌐 Global Numerical Embedding

```python
from kdp.features import NumericalFeature, FeatureType
from kdp.layers_factory import PreprocessorLayerFactory

# Process multiple numeric features as a group with global pooling
global_numerics = NumericalFeature(
    name="numeric_group",
    feature_type=FeatureType.FLOAT,
    preprocessors=[
        PreprocessorLayerFactory.cast_to_float32_layer,
        PreprocessorLayerFactory.global_numerical_embedding_layer,
    ],
    # Parameters for GlobalNumericalEmbedding
    global_embedding_dim=32,        # Final embedding dimension
    global_mlp_hidden_units=64,     # MLP hidden units
    global_num_bins=15,             # Number of bins
    global_dropout_rate=0.1,        # Dropout rate
    global_use_batch_norm=True,     # Apply batch normalization
    global_pooling="average"        # Pooling method ("average" or "max")
)
```

#### 🔀 Gated Linear Unit

```python
from kdp.features import NumericalFeature, FeatureType
from kdp.layers_factory import PreprocessorLayerFactory

# Apply a gated linear unit to a feature
gated_feature = NumericalFeature(
    name="sales_volume",
    feature_type=FeatureType.FLOAT,
    preprocessors=[
        PreprocessorLayerFactory.cast_to_float32_layer,
        "Normalization",
        PreprocessorLayerFactory.gated_linear_unit_layer,
    ],
    # Parameters for GatedLinearUnit
    units=32  # Output dimension
)
```

#### 🔄 Gated Residual Network

```python
from kdp.features import NumericalFeature, FeatureType
from kdp.layers_factory import PreprocessorLayerFactory

# Apply a gated residual network to a feature
grn_feature = NumericalFeature(
    name="complex_metric",
    feature_type=FeatureType.FLOAT,
    preprocessors=[
        PreprocessorLayerFactory.cast_to_float32_layer,
        PreprocessorLayerFactory.gated_residual_network_layer,
    ],
    # Parameters for GatedResidualNetwork
    units=64,            # Output dimension
    dropout_rate=0.3     # Dropout rate
)
```

### ➕ Adding Preprocessors Dynamically

You can also add preprocessors dynamically after feature creation:

```python
from kdp.features import NumericalFeature, FeatureType
from tensorflow.keras.layers import Dense

# Create a feature
feature = NumericalFeature(
    name="age",
    feature_type=FeatureType.FLOAT_NORMALIZED
)

# Add preprocessors later
feature.add_preprocessor("Normalization")
feature.add_preprocessor(Dense, units=16, activation="relu")
feature.add_preprocessor(PreprocessorLayerFactory.distribution_aware_encoder, num_bins=50)
```

## 🚄 DynamicPreprocessingPipeline

For even more flexibility and direct control over your preprocessing workflow, KDP offers the `DynamicPreprocessingPipeline`. This powerful tool allows you to create custom preprocessing pipelines with explicit dependency tracking between layers and seamless integration with TensorFlow's data pipelines.

### ✨ Key Benefits

- 🔄 **Layer Dependencies**: Automatically tracks dependencies between layers
- 🎯 **Selective Computation**: Only computes what's needed based on dependencies
- 💾 **Memory Efficiency**: Doesn't keep unnecessary intermediate tensors
- 🔌 **TF Data Integration**: Works directly with `tf.data.Dataset` objects
- 🛠️ **Customizability**: Use any Keras layer in your preprocessing pipeline
- 📝 **Simplicity**: Cleaner approach for complex preprocessing compared to feature-based methods

### 🏁 Basic Usage

```python
import tensorflow as tf
from kdp.dynamic_pipeline import DynamicPreprocessingPipeline

# Create custom layers
class ScalingLayer(tf.keras.layers.Layer):
    def __init__(self, scaling_factor=2.0, **kwargs):
        super().__init__(**kwargs)
        self.scaling_factor = scaling_factor

    def call(self, inputs):
        return inputs * self.scaling_factor

    def get_config(self):
        config = super().get_config()
        config.update({"scaling_factor": self.scaling_factor})
        return config

class NormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=0)
        std = tf.math.reduce_std(inputs, axis=0)
        return (inputs - mean) / (std + 1e-5)

    def get_config(self):
        return super().get_config()

# Create the pipeline with custom layers
scaling_layer = ScalingLayer(scaling_factor=3.0, name='scaling')
normalization_layer = NormalizationLayer(name='normalization')
pipeline = DynamicPreprocessingPipeline([scaling_layer, normalization_layer])

# Create sample data with keys matching layer names
data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
dataset = tf.data.Dataset.from_tensor_slices({
    'scaling': data,
    'normalization': data
})

# Process the data
processed_dataset = pipeline.process(dataset)

# Use the processed data
for element in processed_dataset:
    print("Scaled data:", element['scaling'].numpy())
    print("Normalized data:", element['normalization'].numpy())
```

### 🔗 Pipeline with Dependencies

The real power of `DynamicPreprocessingPipeline` comes from its ability to automatically handle dependencies between layers:

```python
import tensorflow as tf
from kdp.dynamic_pipeline import DynamicPreprocessingPipeline

# Create a pipeline with a sequence of layers
scaling_layer = ScalingLayer(scaling_factor=2.0, name='scaling')
log_layer = LogTransformLayer(name='log_transform')
norm_layer = NormalizationLayer(name='normalization')

# Create pipeline with dependency order - each layer processes the output of the previous
pipeline = DynamicPreprocessingPipeline([scaling_layer, log_layer, norm_layer])

# Only need to provide the initial input - the rest is handled automatically
data = np.array([[1.0], [5.0], [10.0], [50.0], [100.0]], dtype=np.float32)
dataset = tf.data.Dataset.from_tensor_slices({
    'scaling': data,  # Only provide the input for the first layer
})

# Process the data
processed_dataset = pipeline.process(dataset)

# Access all intermediate and final outputs
for element in processed_dataset:
    print("Scaled data:", element['scaling'].numpy())
    print("Log-transformed data:", element['log_transform'].numpy())
    print("Normalized data:", element['normalization'].numpy())
```

### 🧩 Processing Multiple Feature Types

You can create separate pipelines for different feature types:

```python
import tensorflow as tf
import numpy as np
from kdp.dynamic_pipeline import DynamicPreprocessingPipeline

# Custom encoding layer for categorical features
class EncodingLayer(tf.keras.layers.Layer):
    def __init__(self, vocabulary=None, **kwargs):
        super().__init__(**kwargs)
        self.vocabulary = vocabulary or []

    def build(self, input_shape):
        self.lookup_table = tf.keras.layers.StringLookup(
            vocabulary=self.vocabulary,
            mask_token=None,
            num_oov_indices=1
        )
        super().build(input_shape)

    def call(self, inputs):
        indices = self.lookup_table(inputs)
        return tf.one_hot(indices, depth=len(self.vocabulary) + 1)

# Create pipelines for different feature types
numeric_scaling = ScalingLayer(scaling_factor=2.0, name='numeric_scaling')
numeric_pipeline = DynamicPreprocessingPipeline([numeric_scaling])

categorical_encoding = EncodingLayer(
    vocabulary=['A', 'B', 'C'],
    name='categorical_encoding'
)
categorical_pipeline = DynamicPreprocessingPipeline([categorical_encoding])

# Process different types of data
numeric_data = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
categorical_data = np.array([['A'], ['B'], ['C'], ['D']], dtype=np.object_)

numeric_dataset = tf.data.Dataset.from_tensor_slices({
    'numeric_scaling': numeric_data
})

categorical_dataset = tf.data.Dataset.from_tensor_slices({
    'categorical_encoding': categorical_data
})

# Process each dataset
processed_numeric = numeric_pipeline.process(numeric_dataset)
processed_categorical = categorical_pipeline.process(categorical_dataset)
```

### 🔄 Integration with Keras Models

The `DynamicPreprocessingPipeline` can be easily integrated with Keras models:

```python
import tensorflow as tf
from kdp.dynamic_pipeline import DynamicPreprocessingPipeline

# Create preprocessing pipeline
scaling_layer = ScalingLayer(scaling_factor=2.0, name='scaling')
normalization_layer = NormalizationLayer(name='normalization')
preprocess_pipeline = DynamicPreprocessingPipeline([scaling_layer, normalization_layer])

# Create a simple Keras model
inputs = tf.keras.Input(shape=(1,), name='model_input')
dense1 = tf.keras.layers.Dense(10, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(1)(dense1)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

# Prepare data
data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
targets = np.array([[2.0], [4.0], [6.0], [8.0], [10.0]], dtype=np.float32)

# Create dataset and preprocess
dataset = tf.data.Dataset.from_tensor_slices({
    'scaling': data,
    'normalization': data,
    'y': targets
}).batch(2)

processed_dataset = preprocess_pipeline.process(dataset)

# Create training data generator
def data_generator():
    for batch in processed_dataset:
        # Use the normalized data as model input
        x = batch['normalization']
        y = batch['y']
        yield x, y

# Create a dataset from the generator and train the model
train_dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    )
)

model.fit(train_dataset, epochs=5)
```

## 🛠️ Creating Custom Preprocessing Layers

For even more flexibility, you can create your own custom preprocessing layer:

```python
import tensorflow as tf

class CustomScalingLayer(tf.keras.layers.Layer):
    def __init__(self, scaling_factor=10.0, **kwargs):
        super().__init__(**kwargs)
        self.scaling_factor = scaling_factor

    def call(self, inputs):
        return inputs * self.scaling_factor

    def get_config(self):
        config = super().get_config()
        config.update({"scaling_factor": self.scaling_factor})
        return config

# Use your custom layer in a feature
from kdp.features import NumericalFeature, FeatureType

feature = NumericalFeature(
    name="custom_scaled",
    feature_type=FeatureType.FLOAT,
    preprocessors=[
        CustomScalingLayer,
        "Dense"
    ],
    units=16             # For Dense
)
```

## 🤔 When to Use DynamicPreprocessingPipeline vs. PreprocessingModel

Both approaches have their strengths:

| 🚄 **Use DynamicPreprocessingPipeline when:** | 📦 **Use PreprocessingModel when:** |
|---|---|
| 🔍 You need fine-grained control over the preprocessing flow | 🔄 You want the full KDP feature mapping system |
| 🔗 You want explicit dependency tracking between layers | 🧩 You need integration with other KDP features (feature selection, etc.) |
| 📊 You're working with `tf.data.Dataset` and want efficient streaming | 📝 You prefer a declarative approach with feature specifications |
| 🧪 You prefer a more procedural approach to preprocessing | 💾 You want to save the entire preprocessing model as one unit |
| ⚡ You want to avoid the overhead of the feature-based system | |

## 📋 Complete Example with DynamicPreprocessingPipeline

Here's a comprehensive example showing how to use `DynamicPreprocessingPipeline` with various custom layer types:

```python
import tensorflow as tf
import numpy as np
from kdp.dynamic_pipeline import DynamicPreprocessingPipeline

# Define custom layers
class ScalingLayer(tf.keras.layers.Layer):
    def __init__(self, scaling_factor=2.0, **kwargs):
        super().__init__(**kwargs)
        self.scaling_factor = scaling_factor

    def call(self, inputs):
        return inputs * self.scaling_factor

class LogTransformLayer(tf.keras.layers.Layer):
    def __init__(self, offset=1.0, **kwargs):
        super().__init__(**kwargs)
        self.offset = offset

    def call(self, inputs):
        return tf.math.log(inputs + self.offset)

class NormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=0)
        std = tf.math.reduce_std(inputs, axis=0)
        return (inputs - mean) / (std + 1e-5)

class EncodingLayer(tf.keras.layers.Layer):
    def __init__(self, vocabulary=None, **kwargs):
        super().__init__(**kwargs)
        self.vocabulary = vocabulary or []

    def build(self, input_shape):
        self.lookup_table = tf.keras.layers.StringLookup(
            vocabulary=self.vocabulary,
            mask_token=None,
            num_oov_indices=1
        )
        super().build(input_shape)

    def call(self, inputs):
        indices = self.lookup_table(inputs)
        return tf.one_hot(indices, depth=len(self.vocabulary) + 1)

# Create a multi-step pipeline
scaling = ScalingLayer(scaling_factor=2.0, name='scaling')
log_transform = LogTransformLayer(name='log_transform')
normalization = NormalizationLayer(name='normalization')
pipeline = DynamicPreprocessingPipeline([scaling, log_transform, normalization])

# Create sample data
numeric_data = np.array([[1.0], [5.0], [10.0], [50.0], [100.0]], dtype=np.float32)
dataset = tf.data.Dataset.from_tensor_slices({
    'scaling': numeric_data  # Initial input
}).batch(2)

# Process the data
processed_dataset = pipeline.process(dataset)

# Create a model
inputs = tf.keras.Input(shape=(1,))
x = tf.keras.layers.Dense(10, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

# Use the preprocessed data for training
def data_generator():
    for batch in processed_dataset:
        # Use the fully processed data
        x = batch['normalization']
        y = x * 2  # Synthetic targets
        yield x, y

train_dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    )
)

# Train the model
model.fit(train_dataset, epochs=5)
```

## 📝 Summary

KDP offers multiple approaches to custom preprocessing, from simple layer addition to sophisticated dynamic pipelines. The `DynamicPreprocessingPipeline` provides a powerful and flexible way to create custom preprocessing workflows with explicit dependency tracking, while the feature-based approach with `PreprocessingModel` offers integration with KDP's broader feature handling ecosystem. Choose the approach that best fits your specific needs and workflow preferences.

## 💡 Best Practices

1. 🏁 **Start Simple**: Begin with the simplest preprocessing pipeline that meets your needs
2. 🧪 **Test Incrementally**: Add preprocessing steps one at a time and test their impact
3. ⚡ **Consider Performance**: Complex preprocessing can impact training and inference speed
4. 💾 **Monitor Memory Usage**: Custom preprocessing can increase memory requirements
5. 📝 **Document Your Approach**: Document why custom preprocessing was necessary
6. 🔁 **Ensure Reproducibility**: Make sure custom preprocessing is deterministic

## 🤖 Auto-Configuration Script

KDP provides an auto-configuration script that analyzes your dataset and recommends optimal preprocessing configurations. This tool can help you get started quickly by automatically detecting feature types and suggesting appropriate preprocessing steps.

### 🚀 Basic Usage

```python
from kdp import auto_configure

# Analyze your dataset and get recommendations
config = auto_configure(
    data_path="your_data.csv",
    batch_size=50000,
    save_stats=True
)

# Review the recommendations
print(config["recommendations"])  # Feature-specific recommendations
print(config["code_snippet"])     # Ready-to-use code
```

### 📊 What It Analyzes

The auto-configuration script examines:

- 🔍 **Data Distributions**: Identifies patterns in numerical data
- 📈 **Feature Statistics**: Calculates mean, variance, skewness, etc.
- 🎯 **Value Ranges**: Detects min/max values and outliers
- 🔄 **Value Patterns**: Distinguishes between discrete and continuous values

### 🛠️ Command Line Interface

You can also use the script from the command line:

```bash
python -m kdp.scripts.analyze_dataset \
    --data your_data.csv \
    --output recommendations.json \
    --stats features_stats.json \
    --batch-size 50000
```

### 📝 Example Output

The script generates a comprehensive report including:

```python
{
    "recommendations": {
        "income": {
            "feature_type": "NumericalFeature",
            "preprocessing": ["NORMALIZATION"],
            "detected_distribution": "log_normal",
            "config": {
                "embedding_dim": 16,
                "num_bins": 20
            }
        },
        "age": {
            "feature_type": "NumericalFeature",
            "preprocessing": ["NORMALIZATION"],
            "detected_distribution": "normal",
            "config": {
                "embedding_dim": 8,
                "num_bins": 10
            }
        }
    },
    "code_snippet": "# Generated code implementing the recommendations",
    "statistics": {
        # Detailed feature statistics
    }
}
```

### 💡 Pro Tips for Auto-Configuration

1. **Review Before Implementing**: Always review the recommendations before applying them
2. **Combine with Domain Knowledge**: Use the recommendations alongside your expertise
3. **Update When Data Changes**: Rerun the analysis when your data distribution changes
4. **Customize as Needed**: Modify the generated code to match your specific requirements

## ⚠️ Limitations and Considerations

- 💾 Custom preprocessing layers must be compatible with TensorFlow's serialization
- 🔄 All layers must implement `get_config()` and `from_config()` for proper saving/loading
- ⏱️ Complex custom preprocessing may impact performance
- 🚀 Consider using `tf.function` for performance-critical custom operations
- 🧪 Ensure custom preprocessing works in both eager and graph execution modes

## 🔍 Advanced Topics

### 🔄 Handling Stateful Preprocessing

For preprocessing that requires state (like normalization), ensure proper initialization:

```python
# Stateful preprocessing example
from kdp.features import NumericalFeature, FeatureType
import tensorflow as tf

feature = NumericalFeature(
    name="height",
    feature_type=FeatureType.FLOAT,
    preprocessors=[
        "Normalization"
    ]
)

# The normalization layer is adapted while the preprocessor is built
model = PreprocessingModel(
    path_data="data.csv",
    features_specs={"height": feature},
)
model.build_preprocessor()
```

### 🚀 GPU-Accelerated Custom Preprocessing

Ensure your custom layers leverage GPU acceleration when available:

```python
class GPUAwareCustomLayer(tf.keras.layers.Layer):
    @tf.function  # Enable graph execution for better GPU performance
    def call(self, inputs):
        # Use TensorFlow operations that support GPU execution
        return tf.nn.relu(inputs) * tf.math.sqrt(tf.abs(inputs))
```

### 🐞 Debugging Custom Preprocessing

To debug custom preprocessing pipelines:

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Create a model with your custom preprocessing
model = PreprocessingModel(path_data="data.csv", features_specs=features)

# Inspect the model layers
model.build_preprocessor()
model.model.summary()

# Test with small batch
small_batch = data.head(5)
result = model.transform(small_batch)
print(result)
```
