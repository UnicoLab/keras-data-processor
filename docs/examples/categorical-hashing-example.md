# Categorical Feature Hashing Example

This example demonstrates how to use feature hashing for categorical variables in the KDP library.

## What is Categorical Feature Hashing?

Feature hashing (also known as the "hashing trick") is a technique used to transform high-cardinality categorical features into a fixed-size vector representation. It's particularly useful for:

- Handling categorical features with very large numbers of unique values
- Dealing with previously unseen categories at inference time
- Reducing memory usage for high-cardinality features

## When to Use Hashing vs. Embeddings or One-Hot Encoding

- **One-Hot Encoding**: Best for low-cardinality features (typically <10 categories)
- **Embeddings**: Good for medium-cardinality features where the relationships between categories are important
- **Hashing**: Ideal for high-cardinality features (hundreds or thousands of unique values)

## Basic Example

```python
from kdp.features import CategoricalFeature, FeatureType, CategoryEncodingOptions
from kdp.processor import PreprocessingModel

# Define a categorical feature with hashing
features = {
    "high_cardinality_feature": CategoricalFeature(
        name="high_cardinality_feature",
        feature_type=FeatureType.STRING_CATEGORICAL,
        category_encoding=CategoryEncodingOptions.HASHING,
        hash_bucket_size=1024  # Number of hash buckets
    )
}

# Create a preprocessing model with the features
model = PreprocessingModel(features_specs=features)
```

## Advanced Hashing Options

### Hash with Embeddings

You can combine hashing with embeddings to reduce dimensionality further:

```python
from kdp.features import CategoricalFeature, FeatureType, CategoryEncodingOptions

features = {
    "hashed_with_embedding": CategoricalFeature(
        name="hashed_with_embedding",
        feature_type=FeatureType.STRING_CATEGORICAL,
        category_encoding=CategoryEncodingOptions.HASHING,
        hash_bucket_size=512,     # Number of hash buckets
        hash_with_embedding=True, # Enable embedding layer after hashing
        embedding_size=16         # Size of the embedding vectors
    )
}
```

### Custom Hash Salt

Adding a salt value to the hash function can help prevent collisions between different features:

```python
features = {
    "product_id": CategoricalFeature(
        name="product_id",
        feature_type=FeatureType.STRING_CATEGORICAL,
        category_encoding=CategoryEncodingOptions.HASHING,
        hash_bucket_size=2048,
        salt=42  # Custom salt value for hashing
    )
}
```

## Comparison of Different Encoding Methods

```python
features = {
    # Small cardinality - one hot encoding
    "product_category": CategoricalFeature(
        name="product_category",
        feature_type=FeatureType.STRING_CATEGORICAL,
        category_encoding=CategoryEncodingOptions.ONE_HOT_ENCODING
    ),

    # Medium cardinality - embeddings
    "store_id": CategoricalFeature(
        name="store_id",
        feature_type=FeatureType.STRING_CATEGORICAL,
        category_encoding=CategoryEncodingOptions.EMBEDDING,
        embedding_size=8
    ),

    # High cardinality - hashing
    "customer_id": CategoricalFeature(
        name="customer_id",
        feature_type=FeatureType.STRING_CATEGORICAL,
        category_encoding=CategoryEncodingOptions.HASHING,
        hash_bucket_size=1024
    ),

    # Very high cardinality - hashing with embedding
    "product_id": CategoricalFeature(
        name="product_id",
        feature_type=FeatureType.STRING_CATEGORICAL,
        category_encoding=CategoryEncodingOptions.HASHING,
        hash_bucket_size=2048,
        hash_with_embedding=True,
        embedding_size=16
    )
}
```

## Automatic Configuration with ModelAdvisor

KDP's `ModelAdvisor` can automatically determine the best encoding strategy for each feature based on data statistics:

```python
from kdp.stats import DatasetStatistics
from kdp.model_advisor import recommend_model_configuration

# First, analyze your dataset
dataset_stats = DatasetStatistics("e_commerce_data.csv")
dataset_stats.compute_statistics()

# Get recommendations from the ModelAdvisor
recommendations = recommend_model_configuration(dataset_stats.features_stats)

# Print feature-specific recommendations
for feature, config in recommendations["features"].items():
    if "HASHING" in config.get("preprocessing", []):
        print(f"Feature '{feature}' recommended for hashing:")
        print(f"  - Hash bucket size: {config['config'].get('hash_bucket_size')}")
        print(f"  - Use embeddings: {config['config'].get('hash_with_embedding', False)}")
        if config['config'].get('hash_with_embedding'):
            print(f"  - Embedding size: {config['config'].get('embedding_size')}")
        print(f"  - Salt value: {config['config'].get('salt')}")
        print(f"  - Notes: {', '.join(config.get('notes', []))}")
        print()

# Generate ready-to-use code
print("Generated code snippet:")
print(recommendations["code_snippet"])
```

The ModelAdvisor uses these heuristics for categorical features:
- For features with <50 unique values: ONE_HOT_ENCODING
- For features with 50-1000 unique values: EMBEDDING
- For features with >1000 unique values: HASHING
- For features with >10,000 unique values: HASHING with embeddings

It also automatically determines:
- The appropriate hash bucket size based on cardinality
- Whether to add salt values to prevent collisions
- Embedding dimensions when using hash_with_embedding=True

## Choosing the Right Hash Bucket Size

The number of hash buckets is a critical parameter that affects model performance:

- Too few buckets: Many categories will hash to the same bucket (high collision rate)
- Too many buckets: Sparse representation that might not generalize well

A good rule of thumb is to use a bucket size that is 2-4 times the number of unique categories in your data.

## Handling Hash Collisions

Hash collisions occur when different category values hash to the same bucket. There are two common strategies to mitigate this:

1. **Increase bucket size**: Use more buckets to reduce collision probability
2. **Multi-hashing**: Apply multiple hash functions and use all outputs:

```python
# Example of using multi-hash technique (available in advanced settings)
features = {
    "complex_id": CategoricalFeature(
        name="complex_id",
        feature_type=FeatureType.STRING_CATEGORICAL,
        category_encoding=CategoryEncodingOptions.HASHING,
        hash_bucket_size=1024,
        hash_with_embedding=True
    )
}
```

## Performance Considerations

Hashing is computationally efficient compared to maintaining a large vocabulary mapping, especially when:

- You have a very large number of unique categories
- New categories appear frequently in production
- Memory is constrained

Feature hashing trades off a small amount of accuracy (due to potential collisions) for significant efficiency gains with very high-cardinality features.

## Complete End-to-End Example

Here's a complete example showing how to use feature hashing for e-commerce product data:

```python
import pandas as pd
from kdp.features import CategoricalFeature, FeatureType, CategoryEncodingOptions, NumericalFeature
from kdp.processor import PreprocessingModel

# Create sample e-commerce data
data = {
    "product_id": [f"p{i}" for i in range(1000)],  # High cardinality
    "category": ["electronics", "clothing", "books", "home"] * 250,  # Low cardinality
    "store_id": [f"store_{i % 100}" for i in range(1000)],  # Medium cardinality
    "user_id": [f"user_{i % 10000}" for i in range(1000)],  # Very high cardinality
    "price": [i * 0.1 for i in range(1000)]  # Numerical
}
df = pd.DataFrame(data)
df.to_csv("ecommerce.csv", index=False)

# Define features with appropriate encoding strategies
features = {
    # Low cardinality - one hot encoding
    "category": CategoricalFeature(
        name="category",
        feature_type=FeatureType.STRING_CATEGORICAL,
        category_encoding=CategoryEncodingOptions.ONE_HOT_ENCODING
    ),

    # Medium cardinality - embedding
    "store_id": CategoricalFeature(
        name="store_id",
        feature_type=FeatureType.STRING_CATEGORICAL,
        category_encoding=CategoryEncodingOptions.EMBEDDING,
        embedding_size=8
    ),

    # High cardinality - hashing
    "product_id": CategoricalFeature(
        name="product_id",
        feature_type=FeatureType.STRING_CATEGORICAL,
        category_encoding=CategoryEncodingOptions.HASHING,
        hash_bucket_size=2048,
        salt=1  # Use different salt values for different features
    ),

    # Very high cardinality - hashing with embedding
    "user_id": CategoricalFeature(
        name="user_id",
        feature_type=FeatureType.STRING_CATEGORICAL,
        category_encoding=CategoryEncodingOptions.HASHING,
        hash_bucket_size=4096,
        hash_with_embedding=True,
        embedding_size=16,
        salt=2  # Different salt to avoid collisions with product_id
    ),

    # Numerical feature
    "price": NumericalFeature(
        name="price",
        feature_type=FeatureType.FLOAT_NORMALIZED
    )
}

# Create and build the model
model = PreprocessingModel(
    path_data="ecommerce.csv",
    features_specs=features,
    output_mode="CONCAT"
)

# Build the preprocessor
preprocessor = model.build_preprocessor()

# Use the preprocessor for inference
input_data = {
    "category": ["electronics"],
    "store_id": ["store_42"],
    "product_id": ["p999"],  # Known product
    "user_id": ["user_new"],  # New user, not seen in training
    "price": [99.99]
}

# Process the data - note how hashing handles both known and unknown values
processed = preprocessor(input_data)
print("Output shape:", processed.shape)
