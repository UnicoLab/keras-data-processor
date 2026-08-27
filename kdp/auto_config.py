"""Automatic model configuration module that provides a simple interface for
analyzing datasets and generating optimal preprocessing configurations.
"""

from pathlib import Path
from typing import Any

from loguru import logger

from kdp.stats import DatasetStatistics
from kdp.model_advisor import ModelAdvisor


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
