# 🔗 Integrations: Connect KDP to Your World

## 📋 Quick Overview

KDP is designed to play nicely with your existing ML ecosystem. This guide shows you how to seamlessly connect KDP's powerful preprocessing with your models, pipelines, and production systems.

## 🚀 One-Minute Integration

```python
from kdp import PreprocessingModel, FeatureType
import tensorflow as tf

# Define features
features = {
    "age": FeatureType.FLOAT_NORMALIZED,
    "income": FeatureType.FLOAT_RESCALED,
    "occupation": FeatureType.STRING_CATEGORICAL
}

# Create & build preprocessor
preprocessor = PreprocessingModel(
    path_data="customers.csv",
    features_specs=features
).build_preprocessor()["model"]  # This is a Keras model!

# Connect to your model - just like any Keras layer
inputs = preprocessor.input
x = preprocessor.output
x = tf.keras.layers.Dense(64, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

# Your full model with preprocessing built-in!
full_model = tf.keras.Model(inputs=inputs, outputs=outputs)
full_model.compile(optimizer="adam", loss="binary_crossentropy")
```

## 🏗️ Integration Patterns

Choose the pattern that fits your workflow:

### Pattern 1: Functional API (Most Flexible)

```python
# This approach gives you complete flexibility
def create_full_model(preprocessor, num_classes=1):
    # Get preprocessor inputs and outputs
    inputs = preprocessor.input
    x = preprocessor.output

    # Build your model architecture
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    # Classification or regression output
    if num_classes == 1:
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    # Create the combined model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create your complete pipeline
model = create_full_model(preprocessor, num_classes=3)
```

### Pattern 2: Model Subclassing (For Complex Models)

```python
class CustomerChurnModel(tf.keras.Model):
    def __init__(self, preprocessor):
        super().__init__()
        # Store the preprocessor
        self.preprocessor = preprocessor

        # Define your layers
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        # First run through preprocessor
        x = self.preprocessor(inputs)

        # Then through your model
        x = self.dense1(x)
        x = self.bn1(x)
        if training:
            x = self.dropout(x)
        x = self.dense2(x)
        return self.output_layer(x)

# Create your model
churn_model = CustomerChurnModel(preprocessor)
```

## 📊 tf.data Integration

KDP works beautifully with tf.data for efficient data loading:

```python
# Create a tf.data pipeline with KDP
def create_dataset(file_path, batch_size=32):
    # Create dataset from CSV
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=batch_size,
        select_columns=list(features.keys()) + ["target"],
        num_epochs=1,
        shuffle=True
    )

    # Split features and labels
    def split_features_label(data):
        label = data.pop("target")
        return data, label

    dataset = dataset.map(split_features_label)

    # Performance optimizations
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# Create your datasets
train_ds = create_dataset("train.csv", batch_size=64)
val_ds = create_dataset("val.csv")

# Train with the data pipeline
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
```

## 🌐 Real-World Examples

### E-Commerce Recommendation System

```python
import tensorflow as tf
from kdp import FeatureType, PreprocessingModel

# Create a recommendation system with KDP preprocessing
def build_recommendation_system(user_features, item_features):
    # User branch
    user_inputs = {k: tf.keras.Input(shape=(1,), name=k) for k in user_features}
    user_preprocessor = PreprocessingModel(
        features_specs=user_features,
        name="user_preprocessor",   # each model in one graph needs its own name
    ).build_preprocessor()["model"]
    user_vector = user_preprocessor(user_inputs)

    # Item branch
    item_inputs = {k: tf.keras.Input(shape=(1,), name=k) for k in item_features}
    item_preprocessor = PreprocessingModel(
        features_specs=item_features,
        name="item_preprocessor",
    ).build_preprocessor()["model"]
    item_vector = item_preprocessor(item_inputs)

    # Combine branches
    dot_product = tf.keras.layers.Dot(axes=1)([user_vector, item_vector])
    output = tf.keras.layers.Dense(1, activation="sigmoid")(dot_product)

    # Create final model
    model = tf.keras.Model(
        inputs={**user_inputs, **item_inputs},
        outputs=output
    )
    return model

# Define features
user_features = {
    "user_id": FeatureType.STRING_CATEGORICAL,
    "age_group": FeatureType.STRING_CATEGORICAL,
    "purchase_history": FeatureType.TEXT
}

item_features = {
    "item_id": FeatureType.STRING_CATEGORICAL,
    "category": FeatureType.STRING_CATEGORICAL,
    "description": FeatureType.TEXT
}

# Build the recommendation system
rec_model = build_recommendation_system(user_features, item_features)
```

### Fraud Detection System

```python
import tensorflow as tf
from kdp import FeatureType, NumericalFeature, PreprocessingModel

# Build a fraud detection pipeline with KDP
preprocessor = PreprocessingModel(
    path_data="transactions.csv",
    features_specs={
        "amount": NumericalFeature(
            name="amount",
            feature_type=FeatureType.FLOAT_RESCALED,
            use_embedding=True
        ),
        "timestamp": FeatureType.DATE,
        "merchant_category": FeatureType.STRING_CATEGORICAL,
        "location": FeatureType.STRING_CATEGORICAL,
        "user_history": FeatureType.TEXT
    },
    # Enable advanced preprocessing
    use_distribution_aware=True,
    tabular_attention=True
).build_preprocessor()["model"]

# Add a time-series layer for sequential patterns
inputs = preprocessor.input
features = preprocessor.output

# Add LSTM for sequence modeling
x = tf.keras.layers.Reshape((-1, features.shape[-1]))(features)
x = tf.keras.layers.LSTM(64, return_sequences=False)(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

# Create the fraud detection model
fraud_model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

## 🚀 Production Deployment

### Method 1: All-in-One Deployment

```python
# Save the complete model (preprocessing + prediction)
full_model.save("production_model.keras")

# Later in production:
from tensorflow import keras
model = keras.models.load_model("production_model.keras")

# Single-step inference
prediction = model({
    "age": [35],
    "income": [75000],
    "occupation": ["engineer"]
})
```

### Method 2: Modular Deployment

```python
# Save components separately
preprocessor.save_model("preprocessor.keras")
prediction_model.save("predictor.keras")

# In production:
preprocessor = keras.models.load_model("preprocessor.keras")
prediction_model = keras.models.load_model("predictor.keras")

# Two-step inference
features = preprocessor(input_data)
prediction = prediction_model(features)
```

### Method 3: TensorFlow Serving

```python
# Export for TensorFlow Serving
import tensorflow as tf

# Define serving signature
@tf.function(input_signature=[{
    "age": tf.TensorSpec(shape=[None], dtype=tf.float32),
    "income": tf.TensorSpec(shape=[None], dtype=tf.float32),
    "occupation": tf.TensorSpec(shape=[None], dtype=tf.string)
}])
def serving_fn(inputs):
    return {"prediction": full_model(inputs)}

# Save with signature
tf.saved_model.save(
    full_model,
    "serving_model/1",
    signatures={"serving_default": serving_fn}
)

# Now deploy with TensorFlow Serving
# docker run -p 8501:8501 --mount type=bind,source=/path/to/serving_model,target=/models/my_model -e MODEL_NAME=my_model tensorflow/serving
```

## 💡 Pro Tips for Production

1. **Optimize for Speed**
   ```python
   # Make preprocessing faster in production
   preprocessor = PreprocessingModel(
       features_specs=features,
       use_caching=True,        # Cache intermediate results
       batch_size=100,             # Process in batches
       output_dtypes="float32"     # Use smaller precision if possible
   )
   ```

2. **Staged Training Strategy**
   ```python
   # Train in stages for better results

   # Stage 1: Freeze preprocessing, train model
   preprocessor.trainable = False
   model.fit(train_data, epochs=5)

   # Stage 2: Fine-tune everything with lower rate
   preprocessor.trainable = True
   model.compile(optimizer=tf.keras.optimizers.Adam(1e-5))
   model.fit(train_data, epochs=3)
   ```

3. **A/B Testing Different Preprocessors**
   ```python
   # Create different preprocessors
   basic_preprocessor = PreprocessingModel(features_specs=features)
   advanced_preprocessor = PreprocessingModel(
       features_specs=features,
       use_distribution_aware=True
   )

   # Create models with identical architecture
   model_A = create_model(basic_preprocessor)
   model_B = create_model(advanced_preprocessor)

   # Train and compare
   history_A = model_A.fit(train_data, validation_data=val_data)
   history_B = model_B.fit(train_data, validation_data=val_data)

   # Compare validation metrics
   import matplotlib.pyplot as plt
   plt.plot(history_A.history['val_accuracy'], label='Basic')
   plt.plot(history_B.history['val_accuracy'], label='Advanced')
   plt.legend()
   plt.show()
   ```

## 🔄 Integrating with Other Frameworks

### PyTorch Integration

```python
# Use KDP preprocessing with PyTorch models
import tensorflow as tf
import torch
import numpy as np

# Build your preprocessor
preprocessor = PreprocessingModel(
    features_specs=features
).build_preprocessor()["model"]

# TensorFlow preprocessing + PyTorch model
class HybridModel:
    def __init__(self, tf_preprocessor, torch_model):
        self.preprocessor = tf_preprocessor
        self.torch_model = torch_model

    def predict(self, inputs):
        # TensorFlow preprocessing
        features = self.preprocessor(inputs).numpy()

        # Convert to PyTorch tensor
        features_torch = torch.from_numpy(features).float()

        # PyTorch inference
        with torch.no_grad():
            output = self.torch_model(features_torch)

        return output.numpy()
```

### Scikit-learn Integration

```python
# Create a scikit-learn compatible wrapper
from sklearn.base import BaseEstimator, TransformerMixin

class KDPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_specs=None):
        self.feature_specs = feature_specs
        self.preprocessor = None

    def fit(self, X, y=None):
        # Create and build preprocessor
        self.preprocessor = PreprocessingModel(
            features_specs=self.feature_specs
        ).build_preprocessor()["model"]
        return self

    def transform(self, X):
        # Apply preprocessing
        return self.preprocessor(X).numpy()

# Use in scikit-learn pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('kdp_preprocessor', KDPTransformer(feature_specs=features)),
    ('classifier', RandomForestClassifier())
])

# Train and use the pipeline
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## 🔗 Next Steps

- [TensorFlow Best Practices](https://www.tensorflow.org/guide/effective_tf2) - Optimize your model performance
- [Quick Start Guide](../getting-started/quick-start.md) - Review KDP basics
- [Tabular Optimization](../optimization/tabular-optimization.md) - Further optimize your pipeline
- [Feature Selection](../optimization/feature-selection.md) - Enhance model efficiency

## Memory and Performance Optimization

KDP provides several optimization techniques to improve model performance:

- **Memory Optimization**: Techniques for reducing memory usage. See [Tabular Optimization](../optimization/tabular-optimization.md).
- **Feature Selection**: Gate features through variable-selection layers. See [Feature Selection](../optimization/feature-selection.md).
