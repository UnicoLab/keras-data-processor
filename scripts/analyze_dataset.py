#!/usr/bin/env python
"""
Script to analyze a dataset and recommend optimal preprocessing configurations.
This script uses the DatasetStatistics and ModelAdvisor classes to analyze
the statistical properties of a dataset and suggest the best preprocessing
approaches for each feature.
"""

import argparse
import json
import os
from pathlib import Path

from loguru import logger

from kdp.features import (
    FeatureType,
)
from kdp.stats import DatasetStatistics


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze a dataset and recommend optimal preprocessing configurations.",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        required=True,
        help="Path to the CSV data file or directory containing CSV files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="recommendations.json",
        help="Path to save the recommendations (JSON format)",
    )
    parser.add_argument(
        "--stats",
        "-s",
        type=str,
        default="features_stats.json",
        help="Path to save/load feature statistics (JSON format)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=50000,
        help="Batch size for processing data",
    )
    parser.add_argument(
        "--overwrite",
        "-w",
        action="store_true",
        help="Overwrite existing statistics file",
    )
    parser.add_argument(
        "--feature-types",
        "-f",
        type=str,
        default=None,
        help=(
            "JSON file specifying feature types (optional). "
            "If not provided, all features will be treated as numerical."
        ),
    )
    return parser.parse_args()


def load_feature_types(file_path: Path) -> dict[str, FeatureType] | None:
    """Load feature types from a JSON file."""
    try:
        with open(file_path) as f:
            feature_types = json.load(f)

        features_specs = {}
        for feature_name, feature_type in feature_types.items():
            feature_type = feature_type.upper()
            try:
                if feature_type == "NUMERICAL" or feature_type == "NUMERIC":
                    features_specs[feature_name] = FeatureType.FLOAT_NORMALIZED
                elif feature_type == "CATEGORICAL":
                    features_specs[feature_name] = FeatureType.STRING_CATEGORICAL
                elif feature_type == "TEXT":
                    features_specs[feature_name] = FeatureType.TEXT
                elif feature_type == "DATE":
                    features_specs[feature_name] = FeatureType.DATE
                else:
                    # Try to convert to enum directly
                    features_specs[feature_name] = FeatureType[feature_type]
            except KeyError:
                logger.warning(
                    f"Unknown feature type '{feature_type}' for feature '{feature_name}'. "
                    f"Defaulting to FLOAT_NORMALIZED.",
                )
                features_specs[feature_name] = FeatureType.FLOAT_NORMALIZED

        return features_specs
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load feature types from {file_path}: {e}")
        return None


def infer_feature_types_from_csv(filepath, sample_size=1000):
    """Infer feature types from a sample of CSV data."""
    import pandas as pd

    # Read a sample of the data
    df = pd.read_csv(filepath, nrows=sample_size)

    features_specs = {}
    for column in df.columns:
        # Check for date patterns
        try:
            if pd.to_datetime(df[column], errors="coerce").notna().mean() > 0.9:
                features_specs[column] = FeatureType.DATE
                continue
        except (ValueError, TypeError, pd.errors.ParserError):
            pass

        # Check data type
        if df[column].dtype == "object":
            # Check if it's text or categorical
            unique_ratio = df[column].nunique() / len(df[column].dropna())
            avg_word_count = (
                df[column].fillna("").astype(str).str.split().str.len().mean()
            )

            if unique_ratio > 0.7 and avg_word_count > 3:
                # Likely text
                features_specs[column] = FeatureType.TEXT
            else:
                # Likely categorical
                features_specs[column] = FeatureType.STRING_CATEGORICAL
        elif pd.api.types.is_integer_dtype(df[column]):
            # Check if it's categorical or numerical
            unique_ratio = df[column].nunique() / len(df[column].dropna())
            if unique_ratio < 0.05 or df[column].nunique() < 20:
                # Likely categorical
                features_specs[column] = FeatureType.INTEGER_CATEGORICAL
            else:
                # Likely numerical
                features_specs[column] = FeatureType.FLOAT_NORMALIZED
        elif pd.api.types.is_float_dtype(df[column]):
            # Numerical
            features_specs[column] = FeatureType.FLOAT_NORMALIZED
        else:
            # Default to numerical
            features_specs[column] = FeatureType.FLOAT_NORMALIZED

    return features_specs


def print_recommendations_summary(recommendations):
    """Print a summary of the recommendations in a nice format."""
    print("\n" + "=" * 80)
    print(" " * 25 + "DATASET ANALYSIS SUMMARY")
    print("=" * 80 + "\n")

    # Print feature recommendations
    feature_recs = recommendations.get("features", {})
    print(f"Analyzed {len(feature_recs)} features:\n")

    # Group by feature type
    feature_types = {}
    for feature, rec in feature_recs.items():
        ftype = rec.get("feature_type", "Unknown")
        if ftype not in feature_types:
            feature_types[ftype] = []
        feature_types[ftype].append((feature, rec))

    # Print by type
    for ftype, features in feature_types.items():
        print(f"\n{ftype}s ({len(features)}):")
        print("-" * 40)
        for feature, rec in features:
            preprocessing = ", ".join(rec.get("preprocessing", []))
            notes = rec.get("notes", [])
            print(f"  • {feature}")
            print(f"    - Preprocessing: {preprocessing}")
            if "detected_distribution" in rec:
                print(f"    - Distribution: {rec['detected_distribution']}")
            if notes:
                print(f"    - Notes: {notes[0]}")
                for note in notes[1:]:
                    print(f"             {note}")
        print()

    # Print global recommendations
    global_config = recommendations.get("global_config", {})
    print("\nGlobal Configuration Recommendations:")
    print("-" * 40)
    for key, value in global_config.items():
        if key == "notes":
            continue
        print(f"  • {key}: {value}")

    if "notes" in global_config:
        print("\nNotes:")
        for note in global_config["notes"]:
            print(f"  • {note}")

    # Print example usage section
    print("\n" + "=" * 80)
    print(" " * 30 + "EXAMPLE USAGE")
    print("=" * 80 + "\n")
    print("The following code snippet implements the recommended configuration:")
    print("\n```python")
    print(recommendations.get("code_snippet", "# No code snippet available"))
    print("```\n")


def save_recommendations(recommendations, output_path):
    """Save recommendations to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(recommendations, f, indent=2)
    print(f"\nRecommendations saved to: {output_path}")


def main():
    """Main function to analyze dataset and generate recommendations."""
    args = parse_arguments()

    if not os.path.exists(args.data):
        logger.error(f"Data file {args.data} does not exist")
        return

    # Load or infer feature types
    feature_specs = None
    if args.feature_types:
        feature_specs = load_feature_types(args.feature_types)
    if not feature_specs:
        feature_specs = infer_feature_types_from_csv(args.data)

    # Initialize dataset statistics calculator
    stats_calculator = DatasetStatistics(
        path_data=args.data,
        features_specs=feature_specs,
        features_stats_path=args.stats,
        overwrite_stats=args.overwrite,
        batch_size=args.batch_size,
    )

    # Calculate statistics and generate recommendations
    stats_calculator.main()
    recommendations = stats_calculator.recommend_model_configuration()

    # Print and save recommendations
    print_recommendations_summary(recommendations)
    save_recommendations(recommendations, args.output)


if __name__ == "__main__":
    main()
