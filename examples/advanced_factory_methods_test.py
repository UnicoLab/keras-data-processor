import tensorflow as tf
import numpy as np
from kdp.layers_factory import PreprocessorLayerFactory

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("Testing KDP's Advanced Factory Methods\n" + "=" * 40)

# Create sample data with different distributions
n_samples = 1000
# Log-normal distribution (right-skewed)
log_normal_data = np.random.lognormal(mean=0, sigma=1, size=(n_samples, 1)).astype(
    np.float32,
)
# Bimodal distribution
bimodal_data = np.concatenate(
    [
        np.random.normal(-3, 1, (n_samples // 2, 1)),
        np.random.normal(3, 1, (n_samples // 2, 1)),
    ],
    axis=0,
).astype(np.float32)
# Uniformly distributed data
uniform_data = np.random.uniform(-5, 5, (n_samples, 1)).astype(np.float32)

print("1. Testing Distribution Transform Layer")
print("-" * 40)

# Create distribution transform layers with different transformations
log_transform = PreprocessorLayerFactory.distribution_transform_layer(
    transform_type="log",
    name="log_transform",
)
box_cox_transform = PreprocessorLayerFactory.distribution_transform_layer(
    transform_type="box-cox",
    lambda_param=0.5,
    name="box_cox_transform",
)
auto_transform = PreprocessorLayerFactory.distribution_transform_layer(
    transform_type="auto",
    name="auto_transform",
)

# Apply transformations
log_transformed = log_transform(log_normal_data)
box_cox_transformed = box_cox_transform(log_normal_data)
auto_transformed = auto_transform(log_normal_data)

print(f"Original data shape: {log_normal_data.shape}")
print(f"Log-transformed data shape: {log_transformed.shape}")
print(f"Box-Cox transformed data shape: {box_cox_transformed.shape}")
print(f"Auto-transformed data shape: {auto_transformed.shape}")
print(
    f"Original data range: [{np.min(log_normal_data):.2f}, {np.max(log_normal_data):.2f}]",
)
print(
    f"Log-transformed range: [{np.min(log_transformed):.2f}, {np.max(log_transformed):.2f}]",
)
print(
    f"Box-Cox transformed range: [{np.min(box_cox_transformed):.2f}, {np.max(box_cox_transformed):.2f}]",
)
print(
    f"Auto-transformed range: [{np.min(auto_transformed):.2f}, {np.max(auto_transformed):.2f}]",
)

print("\n2. Testing Numerical Embedding Layer")
print("-" * 40)

# Create numerical embedding layer
numerical_embedding = PreprocessorLayerFactory.numerical_embedding_layer(
    embedding_dim=16,
    mlp_hidden_units=32,
    num_bins=20,
    init_min=0,
    init_max=10,
    dropout_rate=0.2,
    use_batch_norm=True,
    name="numerical_embedding",
)

# Apply embedding
embedded = numerical_embedding(log_normal_data)
print(f"Original data shape: {log_normal_data.shape}")
print(f"Embedded data shape: {embedded.shape}")

print("\n3. Testing Global Numerical Embedding Layer")
print("-" * 40)

# Create multi-feature data
multi_feature_data = np.concatenate(
    [log_normal_data, uniform_data, bimodal_data],
    axis=1,
)
print(f"Multi-feature data shape: {multi_feature_data.shape}")

# Create global numerical embedding layer
global_embedding = PreprocessorLayerFactory.global_numerical_embedding_layer(
    global_embedding_dim=32,
    global_mlp_hidden_units=64,
    global_num_bins=15,
    global_dropout_rate=0.1,
    global_use_batch_norm=True,
    global_pooling="average",
    name="global_embedding",
)

# Apply global embedding
global_embedded = global_embedding(multi_feature_data)
print(f"Global embedded data shape: {global_embedded.shape}")

print("\n4. Testing Gated Linear Unit Layer")
print("-" * 40)

# Create gated linear unit layer
glu = PreprocessorLayerFactory.gated_linear_unit_layer(units=8, name="glu")

# Apply GLU
glu_output = glu(log_normal_data)
print(f"Original data shape: {log_normal_data.shape}")
print(f"GLU output shape: {glu_output.shape}")

print("\n5. Testing Gated Residual Network Layer")
print("-" * 40)

# Create gated residual network layer
grn = PreprocessorLayerFactory.gated_residual_network_layer(
    units=8,
    dropout_rate=0.2,
    name="grn",
)

# Apply GRN
grn_output = grn(log_normal_data)
print(f"Original data shape: {log_normal_data.shape}")
print(f"GRN output shape: {grn_output.shape}")

print("\nAll tests completed successfully!")
