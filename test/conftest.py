"""
Pytest configuration and shared fixtures for optimized test execution.

This module provides:
- Shared fixtures for common test data and models
- Performance optimizations for TensorFlow
- Memory management utilities
- Parallel execution support
- Aggressive speed optimizations
"""

import keras
import gc
from pathlib import Path
import os
import tempfile
import shutil
from typing import Dict, Any, Generator
import warnings
import threading

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kdp.features import (
    FeatureType,
    NumericalFeature,
    CategoricalFeature,
    TextFeature,
    DateFeature,
)
from kdp.processor import PreprocessingModel


# Global test data cache to avoid regeneration
_TEST_DATA_CACHE = {}
_CACHE_LOCK = threading.Lock()


# Configure TensorFlow for maximum test performance
def configure_tensorflow():
    """Configure TensorFlow for maximum test performance."""
    # Disable GPU for consistent test results and faster startup
    tf.config.set_visible_devices([], "GPU")

    # Keep eager execution enabled for compatibility
    # tf.compat.v1.disable_eager_execution()  # Disabled - causes test failures

    # Set memory growth to avoid OOM issues
    physical_devices = tf.config.list_physical_devices("CPU")
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            # Limit memory usage for faster tests
            tf.config.experimental.set_memory_limit(
                physical_devices[0], 2048
            )  # 2GB limit
        except (ValueError, RuntimeError):
            pass  # Invalid device or cannot modify virtual devices once initialized

    # Aggressive TensorFlow optimizations
    os.environ.update(
        {
            "TF_ENABLE_ONEDNN_OPTS": "0",
            "TF_CPP_MIN_LOG_LEVEL": "3",  # Only fatal errors
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "TF_GPU_THREAD_MODE": "gpu_private",
            "TF_GPU_THREAD_COUNT": "1",
            "TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT": "1",
            "TF_ENABLE_WINOGRAD_NONFUSED": "1",
            "TF_AUTOTUNE_THRESHOLD": "1",
            "TF_DISABLE_MKL": "1",  # Disable Intel MKL for faster startup
            "TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS": "1",
            "CUDA_VISIBLE_DEVICES": "",  # Ensure no GPU usage
        }
    )

    # Reduce TensorFlow logging to minimum
    tf.get_logger().setLevel("FATAL")
    tf.autograph.set_verbosity(0)

    # Suppress all warnings for speed
    warnings.filterwarnings("ignore")
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)


# Configure TensorFlow at module import
configure_tensorflow()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment for maximum performance."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Configure pandas for maximum performance
    pd.set_option("mode.chained_assignment", None)
    pd.set_option("compute.use_bottleneck", True)
    pd.set_option("compute.use_numexpr", True)

    # Configure NumPy for performance
    np.seterr(all="ignore")  # Ignore numpy warnings for speed

    # Pre-warm TensorFlow
    with tf.device("/CPU:0"):
        _ = tf.constant([1.0])

    yield

    # Cleanup after all tests
    keras.backend.clear_session()
    gc.collect()


@pytest.fixture(scope="function", autouse=True)
def cleanup_after_test():
    """Lightweight cleanup after each test."""
    yield
    # Keras 3 has no TF1-style global session, so there is nothing to probe:
    # clearing is cheap and keeps layer-name counters from leaking across tests.
    keras.backend.clear_session()


@pytest.fixture(scope="session")
def temp_directory() -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="kdp_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# Cached data fixtures for speed
@pytest.fixture(scope="session")
def sample_data() -> pd.DataFrame:
    """Generate sample data for testing (cached for session)."""
    cache_key = "sample_data"
    with _CACHE_LOCK:
        if cache_key not in _TEST_DATA_CACHE:
            np.random.seed(42)  # For reproducible test data
            _TEST_DATA_CACHE[cache_key] = pd.DataFrame(
                {
                    "numerical_feature": np.random.randn(50),  # Reduced size for speed
                    "categorical_feature": np.random.choice(["A", "B", "C"], 50),
                    "text_feature": np.random.choice(["text1", "text2", "text3"], 50),
                    "date_feature": pd.date_range(
                        "2020-01-01", periods=50, freq="D"
                    ).strftime("%Y-%m-%d"),
                    "target": np.random.randn(50),
                }
            )
        return _TEST_DATA_CACHE[cache_key].copy()


@pytest.fixture(scope="session")
def time_series_data() -> pd.DataFrame:
    """Generate time series data for testing (cached for session)."""
    cache_key = "time_series_data"
    with _CACHE_LOCK:
        if cache_key not in _TEST_DATA_CACHE:
            np.random.seed(42)
            dates = pd.date_range("2020-01-01", periods=50, freq="D")  # Reduced size
            _TEST_DATA_CACHE[cache_key] = pd.DataFrame(
                {
                    "date": dates.strftime("%Y-%m-%d"),
                    "store_id": np.random.choice(["A", "B"], 50),
                    "sales": np.random.randn(50) * 10 + 100,
                    "temperature": np.random.randn(50) * 5 + 20,
                }
            )
        return _TEST_DATA_CACHE[cache_key].copy()


@pytest.fixture(scope="session")
def basic_features_specs() -> Dict[str, Any]:
    """Basic feature specifications for testing (cached)."""
    return {
        "numerical_feature": FeatureType.FLOAT_NORMALIZED,
        "categorical_feature": FeatureType.STRING_CATEGORICAL,
        "text_feature": FeatureType.TEXT,
        "date_feature": FeatureType.DATE,
    }


@pytest.fixture(scope="session")
def advanced_features_specs() -> Dict[str, Any]:
    """Advanced feature specifications with class instances (cached)."""
    cache_key = "advanced_features_specs"
    with _CACHE_LOCK:
        if cache_key not in _TEST_DATA_CACHE:
            _TEST_DATA_CACHE[cache_key] = {
                "numerical_feature": NumericalFeature(
                    name="numerical_feature", feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "categorical_feature": CategoricalFeature(
                    name="categorical_feature",
                    feature_type=FeatureType.STRING_CATEGORICAL,
                ),
                "text_feature": TextFeature(
                    name="text_feature", feature_type=FeatureType.TEXT
                ),
                "date_feature": DateFeature(
                    name="date_feature",
                    feature_type=FeatureType.DATE,
                    date_format="%Y-%m-%d",
                ),
            }
        return _TEST_DATA_CACHE[cache_key]


@pytest.fixture(scope="session")
def sample_csv_file(temp_directory: str, sample_data: pd.DataFrame) -> str:
    """Create a sample CSV file for testing (cached)."""
    file_path = os.path.join(temp_directory, "sample_data.csv")
    if not os.path.exists(file_path):
        sample_data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture(scope="session")
def time_series_csv_file(temp_directory: str, time_series_data: pd.DataFrame) -> str:
    """Create a time series CSV file for testing (cached)."""
    file_path = os.path.join(temp_directory, "time_series_data.csv")
    if not os.path.exists(file_path):
        time_series_data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture(scope="session")
def stats_file_path(temp_directory: str) -> str:
    """Path for feature stats file."""
    return os.path.join(temp_directory, "features_stats.json")


@pytest.fixture(scope="session")
def basic_preprocessor(
    sample_csv_file: str, basic_features_specs: Dict[str, Any], stats_file_path: str
) -> PreprocessingModel:
    """Create a basic preprocessing model for testing (cached)."""
    cache_key = f"basic_preprocessor_{sample_csv_file}"
    with _CACHE_LOCK:
        if cache_key not in _TEST_DATA_CACHE:
            _TEST_DATA_CACHE[cache_key] = PreprocessingModel(
                path_data=sample_csv_file,
                features_specs=basic_features_specs,
                features_stats_path=stats_file_path,
                overwrite_stats=True,
            )
        return _TEST_DATA_CACHE[cache_key]


@pytest.fixture(scope="session")
def built_preprocessor(basic_preprocessor: PreprocessingModel) -> Dict[str, Any]:
    """Create and build a preprocessing model (cached)."""
    cache_key = "built_preprocessor"
    with _CACHE_LOCK:
        if cache_key not in _TEST_DATA_CACHE:
            _TEST_DATA_CACHE[cache_key] = basic_preprocessor.build_preprocessor()
        return _TEST_DATA_CACHE[cache_key]


# Fast data generation utilities
def generate_fast_data(
    features_specs: Dict[str, Any], num_rows: int = 20
) -> pd.DataFrame:
    """
    Generate fake data quickly with minimal rows.

    Optimized for speed with reduced data size and vectorized operations.
    """
    cache_key = f"fast_data_{hash(str(features_specs))}_{num_rows}"
    with _CACHE_LOCK:
        if cache_key not in _TEST_DATA_CACHE:
            np.random.seed(42)  # For reproducible test data
            data = {}

            for feature_name, spec in features_specs.items():
                if isinstance(spec, (NumericalFeature, type(FeatureType.FLOAT))):
                    data[feature_name] = np.random.randn(num_rows).astype(np.float32)
                elif isinstance(
                    spec, (CategoricalFeature, type(FeatureType.STRING_CATEGORICAL))
                ):
                    data[feature_name] = np.random.choice(
                        ["A", "B"], num_rows
                    )  # Reduced categories
                elif isinstance(spec, (TextFeature, type(FeatureType.TEXT))):
                    data[feature_name] = np.random.choice(
                        ["text1", "text2"], num_rows
                    )  # Reduced options
                elif isinstance(spec, (DateFeature, type(FeatureType.DATE))):
                    # Use a smaller date range for speed
                    dates = pd.date_range("2020-01-01", periods=num_rows, freq="D")
                    data[feature_name] = dates.strftime("%Y-%m-%d")
                else:
                    # Default to numerical
                    data[feature_name] = np.random.randn(num_rows).astype(np.float32)

            _TEST_DATA_CACHE[cache_key] = pd.DataFrame(data)

        return _TEST_DATA_CACHE[cache_key].copy()


# Performance utilities
class PerformanceMonitor:
    """Monitor test performance and memory usage."""

    def __init__(self):
        self.start_memory = None
        self.peak_memory = None

    def start(self):
        """Start monitoring."""
        self.start_memory = self._get_memory_usage()
        self.peak_memory = self.start_memory

    def update_peak(self):
        """Update peak memory usage."""
        current = self._get_memory_usage()
        if current > self.peak_memory:
            self.peak_memory = current

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def get_memory_delta(self) -> float:
        """Get memory usage delta in MB."""
        if self.start_memory is None:
            return 0.0
        return self._get_memory_usage() - self.start_memory


@pytest.fixture
def performance_monitor() -> PerformanceMonitor:
    """Performance monitoring fixture."""
    monitor = PerformanceMonitor()
    monitor.start()
    yield monitor


# Test markers and utilities
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "fast: mark test as fast running")
    config.addinivalue_line("markers", "micro: mark test as micro test (fastest)")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and optimize test order."""
    # Sort tests by estimated execution time (fastest first)
    fast_patterns = ["quick", "simple", "basic", "init", "config"]
    slow_patterns = ["large", "complex", "end_to_end", "integration", "model"]

    fast_tests = []
    medium_tests = []
    slow_tests = []

    for item in items:
        # Exactly one group marker per test, and always one: CI selects tests by
        # these markers, so a test that gets none never runs there. The speed
        # markers below used to be assigned *instead of* `unit`, which quietly
        # hid 26 tests from CI -- the whole save/load round-trip suite among
        # them, because every one of its names contains "model".
        if "test_processor" in item.fspath.basename:
            item.add_marker(pytest.mark.processor)
        elif "test_time_series" in item.fspath.basename:
            item.add_marker(pytest.mark.time_series)
        elif "layers" in str(item.fspath):
            item.add_marker(pytest.mark.layers)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

        # Extra labels, never a substitute for the group marker above.
        if "inference" in item.fspath.basename:
            item.add_marker(pytest.mark.inference)

        # Categorize by speed
        test_name_lower = item.name.lower()
        if any(pattern in test_name_lower for pattern in fast_patterns):
            item.add_marker(pytest.mark.fast)
            item.add_marker(pytest.mark.micro)
            fast_tests.append(item)
        elif any(pattern in test_name_lower for pattern in slow_patterns):
            item.add_marker(pytest.mark.slow)
            slow_tests.append(item)
        else:
            medium_tests.append(item)

    # Reorder items for optimal execution (fast tests first)
    items[:] = fast_tests + medium_tests + slow_tests


# Utility functions for test data generation
def generate_fake_data(
    features_specs: Dict[str, Any], num_rows: int = 20
) -> pd.DataFrame:
    """
    Generate fake data based on feature specifications (optimized version).

    This is a shared utility function that can be used across tests.
    Reduced default size for faster execution.
    """
    return generate_fast_data(features_specs, num_rows)


# Pytest hooks for better performance
def pytest_runtest_setup(item):
    """Setup for each test run with optimizations and conditional skipping."""
    # Skip slow tests if running in fast mode
    if item.config.getoption("--fast-only", default=False):
        if item.get_closest_marker("slow"):
            pytest.skip("Skipping slow test in fast mode")

    # Only clear session if it's a slow test
    if item.get_closest_marker("slow") or item.get_closest_marker("integration"):
        keras.backend.clear_session()


def pytest_runtest_teardown(item, nextitem):
    """Teardown after each test run (minimal overhead)."""
    # Only do expensive cleanup for slow tests
    if item.get_closest_marker("slow") or item.get_closest_marker("integration"):
        keras.backend.clear_session()
        gc.collect()


# Memory optimization hook
def pytest_sessionstart(session):
    """Optimize memory settings at session start."""
    # Force garbage collection
    gc.collect()

    # Set aggressive garbage collection
    gc.set_threshold(100, 5, 5)  # More frequent GC


def pytest_sessionfinish(session, exitstatus):
    """Clean up at session end."""
    # Clear all caches
    global _TEST_DATA_CACHE
    _TEST_DATA_CACHE.clear()

    # Final cleanup
    keras.backend.clear_session()
    gc.collect()


@pytest.fixture(autouse=True)
def _no_stray_stats_file():
    """Keep `./features_stats.json` from leaking between tests.

    `PreprocessingModel` and `DatasetStatistics` default `features_stats_path`
    to `features_stats.json` in the working directory. A test that omits the
    argument therefore leaves a file behind, and a later test asserting that a
    missing `path_data` raises instead finds those statistics and passes for
    the wrong reason -- which is exactly how two such assertions went green
    locally and failed in CI, where each job starts from a fresh checkout.
    """
    default_stats = Path("features_stats.json")
    existed = default_stats.exists()
    yield
    if default_stats.exists() and not existed:
        default_stats.unlink()
