"""Automatic model configuration module that provides a simple interface for
analyzing datasets and generating optimal preprocessing configurations.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from kdp.stats import DatasetStatistics
from kdp.model_advisor import ModelAdvisor
from kdp.features import FeatureType

# A whole-numbered column with at most this many distinct values reads as a
# code rather than a quantity.
_CATEGORICAL_MAX_DISTINCT = 20
# A string column is a date when nearly every value parses as one.
_DATE_MATCH_RATIO = 0.95
# Values of at most this many words are labels; longer ones are prose.
_CATEGORICAL_MAX_WORDS = 3


def _infer_features_specs(
    data_path: Path,
    sample_rows: int = 5_000,
) -> dict[str, FeatureType]:
    """Guess a feature type for every column in the data.

    Without this, `auto_configure` handed `features_specs=None` to
    `DatasetStatistics`, which then had no per-type feature lists, computed
    nothing, and returned empty recommendations -- for the call the docstring
    leads with. Inferring the types is what makes it automatic.

    The rules mirror how the feature types are described in the documentation:

    * a numeric column with few distinct whole values is a categorical code,
      not a quantity;
    * a string column that parses as a date is a date;
    * a string column whose values are short and repeat is categorical;
    * anything else textual is text.

    Args:
        data_path: CSV file, or a directory of CSVs sharing a header.
        sample_rows: How many rows to read when judging a column.

    Returns:
        dict[str, FeatureType]: One entry per column.

    Raises:
        ValueError: If no CSV file can be found at `data_path`.
    """
    csv_file = data_path
    if data_path.is_dir():
        candidates = sorted(data_path.glob("*.csv"))
        if not candidates:
            raise ValueError(f"No CSV files found in {data_path}")
        csv_file = candidates[0]

    frame = pd.read_csv(csv_file, nrows=sample_rows)
    specs: dict[str, FeatureType] = {}

    for column in frame.columns:
        values = frame[column].dropna()
        if values.empty:
            specs[column] = FeatureType.FLOAT_NORMALIZED
            continue

        if pd.api.types.is_numeric_dtype(values):
            distinct = values.nunique()
            whole = bool((values % 1 == 0).all())
            if whole and distinct <= _CATEGORICAL_MAX_DISTINCT:
                specs[column] = FeatureType.INTEGER_CATEGORICAL
            else:
                specs[column] = FeatureType.FLOAT_NORMALIZED
            continue

        text = values.astype(str)
        parsed = pd.to_datetime(text, errors="coerce", format="mixed")
        if parsed.notna().mean() >= _DATE_MATCH_RATIO:
            specs[column] = FeatureType.DATE
            continue

        # Single-token values are labels even when almost all of them differ:
        # an id column belongs in a categorical feature, where hashing keeps it
        # bounded, rather than in a text feature that would build a vocabulary
        # the size of the dataset.
        longest_phrase = int(text.str.split().str.len().max())
        if longest_phrase <= _CATEGORICAL_MAX_WORDS:
            specs[column] = FeatureType.STRING_CATEGORICAL
        else:
            specs[column] = FeatureType.TEXT

    logger.info(f"Inferred feature types: { {k: v.name for k, v in specs.items()} }")
    return specs


def auto_configure(
    data_path: str | Path,
    features_specs: dict[str, Any] | None = None,
    batch_size: int = 50_000,
    save_stats: bool = True,
    stats_path: str | Path | None = None,
    overwrite_stats: bool = False,
) -> dict[str, Any]:
    """Automatically analyze a dataset and generate optimal preprocessing configurations.

    This is a high-level function that handles all the complexity of analyzing your dataset
    and recommending the best preprocessing strategies. It will:
    1. Calculate comprehensive statistics about your features
    2. Analyze the distributions and characteristics of each feature
    3. Generate specific recommendations for preprocessing each feature
    4. Provide global configuration recommendations
    5. Generate ready-to-use code implementing the recommendations

    Args:
        data_path: Path to your dataset (CSV file or directory of CSVs)
        features_specs: Optional dictionary specifying feature types and configurations
        batch_size: Batch size for processing large datasets (default: 50000)
        save_stats: Whether to save the computed statistics (default: True)
        stats_path: Optional path to save/load statistics (default: features_stats.json)
        overwrite_stats: Whether to overwrite existing statistics file (default: False)

    Returns:
        Dictionary containing:
        - feature-specific recommendations
        - global configuration recommendations
        - ready-to-use code snippet
        - computed statistics (if save_stats=True)

    Example:
        >>> config = auto_configure("data/my_dataset.csv")
        >>> print(config["code_snippet"])  # Get ready-to-use code
        >>> print(config["recommendations"])  # Get feature-specific recommendations
    """
    # Convert paths to Path objects
    data_path = Path(data_path)
    stats_path = Path("features_stats.json") if stats_path is None else Path(stats_path)

    # Without specs there is nothing to compute statistics for, so infer them.
    if not features_specs:
        features_specs = _infer_features_specs(data_path)

    # Initialize statistics calculator
    stats_calculator = DatasetStatistics(
        path_data=str(data_path),
        features_specs=features_specs,
        features_stats_path=stats_path,
        overwrite_stats=overwrite_stats,
        batch_size=batch_size,
    )

    # Calculate statistics
    logger.info("Calculating dataset statistics...")
    stats = stats_calculator.main()

    # Generate recommendations
    logger.info("Generating preprocessing recommendations...")
    advisor = ModelAdvisor(stats)
    recommendations = advisor.analyze_feature_stats()

    # Generate code snippet
    logger.info("Generating code snippet...")
    code_snippet = advisor.generate_code_snippet()

    # Prepare output
    output = {
        "recommendations": recommendations,
        "code_snippet": code_snippet,
    }

    if save_stats:
        output["statistics"] = stats

    return output
