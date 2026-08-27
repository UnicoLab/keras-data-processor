"""
Custom Preprocessing Example

This example demonstrates how to define and use custom preprocessing pipelines
for various feature types in the KDP framework.
"""
# ruff: noqa: E402

import keras
import os
import sys
import tempfile
from pathlib import Path

# Add the project root to the Python path to allow module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from kdp.processor import PreprocessingModel
from kdp.features import (
    NumericalFeature,
    CategoricalFeature,
    TextFeature,
    FeatureType,
    CategoryEncodingOptions,
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

#################################################
# Define Custom Preprocessing Layers
#################################################


class LogTransformLayer(keras.layers.Layer):
    """
    Custom layer that applies log(x + offset) transformation.
    Useful for handling right-skewed data.
    """

    def __init__(self, offset=1.0, **kwargs):
        super().__init__(**kwargs)
        self.offset = offset

    def call(self, inputs):
        return tf.math.log(inputs + self.offset)

    def get_config(self):
        config = super().get_config()
        config.update({"offset": self.offset})
        return config


class ClippingLayer(keras.layers.Layer):
    """
    Custom layer that clips values between min_value and max_value.
    Useful for handling outliers.
    """

    def __init__(self, min_value=None, max_value=None, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min_value, self.max_value)

    def get_config(self):
        config = super().get_config()
        config.update({"min_value": self.min_value, "max_value": self.max_value})
        return config


class StringCleaningLayer(keras.layers.Layer):
    """
    Custom layer that performs basic text cleaning operations.
    - Converts to lowercase
    - Removes specified patterns
    - Trims whitespace
    """

    def __init__(self, patterns_to_remove=None, **kwargs):
        super().__init__(**kwargs)
        self.patterns_to_remove = patterns_to_remove or []

    def call(self, inputs):
        # Convert to lowercase
        text = tf.strings.lower(inputs)

        # Remove specified patterns
        for pattern in self.patterns_to_remove:
            text = tf.strings.regex_replace(text, pattern, "")

        # Trim whitespace and replace multiple spaces with a single space
        text = tf.strings.regex_replace(text, r"\s+", " ")
        return tf.strings.strip(text)

    def get_config(self):
        config = super().get_config()
        config.update({"patterns_to_remove": self.patterns_to_remove})
        return config


#################################################
# Generate Sample Dataset
#################################################


JOB_DESCRIPTION_SECTORS = [
    "tech",
    "finance",
    "healthcare",
    "education",
    "manufacturing",
]
JOB_DESCRIPTION_SKILLS = [
    "programming",
    "analysis",
    "management",
    "communication",
    "design",
]
# The words a hand-built TextVectorization has to know about. KDP's own text
# path derives this from the computed statistics; a custom pipeline supplies it.
JOB_DESCRIPTION_VOCABULARY = [
    "this",
    "job",
    "in",
    "requires",
    "skills",
    "and",
    "experience",
    "level",
    *JOB_DESCRIPTION_SECTORS,
    *JOB_DESCRIPTION_SKILLS,
]

JOB_SECTORS = [
    "Technology",
    "Finance",
    "Healthcare",
    "Education",
    "Manufacturing",
    "Other",
]


def generate_sample_data(n_samples=1000):
    """Generate a synthetic dataset for demonstration."""

    # Numeric features with different distributions
    age = np.random.normal(35, 10, n_samples)  # Normal distribution
    salary = np.random.lognormal(10, 1, n_samples)  # Log-normal (right-skewed)
    experience = np.random.poisson(5, n_samples)  # Poisson
    expense_ratio = np.random.beta(2, 5, n_samples)  # Beta distribution

    # Categorical features
    education = np.random.choice(
        ["High School", "Bachelor's", "Master's", "PhD", "Other"],
        n_samples,
    )
    job_sector = np.random.choice(JOB_SECTORS, n_samples)

    # Text feature with some variations
    job_descriptions = []
    sectors = JOB_DESCRIPTION_SECTORS
    skills = JOB_DESCRIPTION_SKILLS
    for _ in range(n_samples):
        sector = np.random.choice(sectors)
        skill1, skill2 = np.random.choice(skills, 2, replace=False)
        job_descriptions.append(
            f"This job in {sector} requires skills in {skill1} and {skill2}. "
            f"Experience level: {np.random.randint(1, 10)} years.",
        )

    # Target variable: synthetic credit score
    # Affected by all features with some noise
    credit_score = (
        300  # Base score
        + age * 2  # Age matters
        + np.log1p(salary) * 15  # Log of salary (diminishing returns)
        - np.log1p(expense_ratio * salary) * 20  # Expense ratio (negative impact)
        + experience * 10  # Experience (positive impact)
        + np.random.normal(0, 50, n_samples)  # Random noise
    )

    # Clip to realistic credit score range
    credit_score = np.clip(credit_score, 300, 850)

    # Create DataFrame
    return pd.DataFrame(
        {
            "age": age,
            "salary": salary,
            "experience": experience,
            "expense_ratio": expense_ratio,
            "education": education,
            "job_sector": job_sector,
            "job_description": job_descriptions,
            "credit_score": credit_score,
        },
    )


#################################################
# Define Features with Custom Preprocessing
#################################################


def define_features():
    """Define features with custom preprocessing pipelines."""

    return {
        # Age: Standard numerical feature with normalization
        "age": NumericalFeature(name="age", feature_type=FeatureType.FLOAT_NORMALIZED),
        # Salary: Log-transform to handle skewness, then normalize
        "salary": NumericalFeature(
            name="salary",
            feature_type=FeatureType.FLOAT,
            preprocessors=[LogTransformLayer, "Normalization", "Dense"],
            offset=1.0,  # for LogTransformLayer
            units=16,  # for Dense
            activation="relu",
        ),
        # Experience: Clip to handle outliers, then apply standard preprocessing
        "experience": NumericalFeature(
            name="experience",
            feature_type=FeatureType.FLOAT,
            preprocessors=[ClippingLayer, "Rescaling"],
            min_value=0,  # for ClippingLayer
            max_value=20,  # for ClippingLayer
            scale=0.1,  # for Rescaling
        ),
        # Expense ratio: Beta distribution, apply special binning.
        # Note that the keyword arguments below are offered to *every* layer in
        # the pipeline, and each takes the ones its constructor accepts. Both
        # Discretization and CategoryEncoding accept `output_mode`, so chaining
        # them here would one-hot the values twice; Discretization alone already
        # emits the one-hot encoding.
        "expense_ratio": NumericalFeature(
            name="expense_ratio",
            feature_type=FeatureType.FLOAT,
            preprocessors=["Discretization"],
            bin_boundaries=[0.1, 0.2, 0.3, 0.4, 0.5],  # for Discretization
            output_mode="one_hot",  # for Discretization
        ),
        # Education: Standard categorical with embedding
        "education": CategoricalFeature(
            name="education",
            feature_type=FeatureType.STRING_CATEGORICAL,
            category_encoding=CategoryEncodingOptions.EMBEDDING,
            embedding_size=8,
        ),
        # Job sector: Custom preprocessing to add special handling
        "job_sector": CategoricalFeature(
            name="job_sector",
            feature_type=FeatureType.STRING_CATEGORICAL,
            preprocessors=["StringLookup", "Embedding", "Dense"],
            # A raw StringLookup in a custom pipeline needs its vocabulary up
            # front: KDP's own categorical path fills this in from the computed
            # statistics, but a hand-built pipeline has to supply it.
            vocabulary=JOB_SECTORS,  # for StringLookup
            num_oov_indices=1,  # for StringLookup
            input_dim=len(JOB_SECTORS) + 1,  # for Embedding: vocabulary + OOV
            output_dim=12,  # for Embedding
            units=8,  # for Dense
            activation="relu",
        ),
        # Job description: Custom text cleaning + standard text preprocessing
        "job_description": TextFeature(
            name="job_description",
            feature_type=FeatureType.TEXT,
            preprocessors=[
                StringCleaningLayer,
                "TextVectorization",
                "Embedding",
                "GlobalAveragePooling1D",
            ],
            patterns_to_remove=["[0-9]+", "years"],  # for StringCleaningLayer
            vocabulary=JOB_DESCRIPTION_VOCABULARY,  # for TextVectorization
            output_sequence_length=30,  # for TextVectorization
            # Embedding must cover the vocabulary plus the padding and OOV slots
            # TextVectorization prepends.
            input_dim=len(JOB_DESCRIPTION_VOCABULARY) + 2,  # for Embedding
            output_dim=32,  # for Embedding
        ),
    }


#################################################
# Main Execution
#################################################


def main():
    """Build a preprocessor from custom pipelines and train a model on its output."""
    # Generate sample data
    print("Generating sample data...")
    data = generate_sample_data()
    print(f"Dataset shape: {data.shape}")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("credit_score", axis=1),
        data["credit_score"],
        test_size=0.2,
        random_state=42,
    )

    # Define features with custom preprocessing
    print("\nDefining features with custom preprocessing pipelines...")
    features = define_features()

    # KDP computes its statistics from data on disk, so the training split is
    # written out first and the preprocessor is pointed at that file.
    work_dir = Path(tempfile.mkdtemp(prefix="kdp_custom_preprocessing_"))
    train_csv = work_dir / "train.csv"
    X_train.to_csv(train_csv, index=False)

    print("Creating preprocessing model...")
    preprocessor = PreprocessingModel(
        path_data=str(train_csv),
        features_specs=features,
        features_stats_path=str(work_dir / "features_stats.json"),
        overwrite_stats=True,
        output_mode="concat",
    )

    print("Building preprocessing model...")
    preprocessor.build_preprocessor()

    def to_model_inputs(frame: pd.DataFrame) -> dict:
        """Turn a DataFrame into the per-feature tensors the model expects."""
        return {
            column: tf.constant(frame[column].to_numpy().reshape(-1, 1))
            for column in features
        }

    print("Transforming data with custom preprocessing...")
    X_train_processed = np.asarray(preprocessor.model(to_model_inputs(X_train)))
    X_test_processed = np.asarray(preprocessor.model(to_model_inputs(X_test)))

    print(f"Processed training data shape: {X_train_processed.shape}")
    print(f"Processed testing data shape: {X_test_processed.shape}")

    # Build prediction model using the preprocessed data
    print("\nBuilding prediction model...")
    model = keras.Sequential(
        [
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ],
    )

    # Compile the model
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Train the model
    print("Training model...")
    model.fit(
        X_train_processed,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
    )

    # Evaluate the model
    print("\nEvaluating model...")
    y_pred = model.predict(X_test_processed)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Root Mean Squared Error: {rmse:.2f}")

    # Save the preprocessing model and prediction model
    print("\nSaving models...")
    preprocessor.save_model(str(work_dir / "custom_preprocessing_model"))
    model.save(str(work_dir / "credit_score_prediction_model.keras"))

    print(
        "Done! The preprocessor with custom pipelines and prediction model are saved.",
    )

    # Example of how to use the saved models for inference
    print("\nExample inference with a new sample:")
    new_sample = pd.DataFrame(
        {
            "age": [30],
            "salary": [85000],
            "experience": [5],
            "expense_ratio": [0.25],
            "education": ["Master's"],
            "job_sector": ["Technology"],
            "job_description": [
                "This job in tech requires skills in programming and management.",
            ],
        },
    )

    # Preprocess the new sample
    new_sample_processed = np.asarray(preprocessor.model(to_model_inputs(new_sample)))

    # Make prediction
    prediction = model.predict(new_sample_processed)

    print(f"Sample data: {new_sample.iloc[0].to_dict()}")
    print(f"Predicted credit score: {prediction[0][0]:.2f}")


if __name__ == "__main__":
    main()
