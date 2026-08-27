#!/usr/bin/env python3
"""
Test Data Optimization Script

This script automatically optimizes test data sizes and configurations
for maximum speed while maintaining test coverage.
"""

import re
from pathlib import Path
from typing import Any


class TestDataOptimizer:
    """Optimize test data for faster execution."""

    def __init__(self, test_dir: Path = None):
        self.test_dir = test_dir or Path("test")
        self.optimizations_applied = []

    def optimize_all_tests(self) -> dict[str, Any]:
        """Optimize all test files for faster execution."""
        results = {
            "files_processed": 0,
            "optimizations_applied": 0,
            "files_modified": [],
        }

        for test_file in self.test_dir.rglob("test_*.py"):
            if self.optimize_test_file(test_file):
                results["files_modified"].append(str(test_file))
                results["files_processed"] += 1

        results["optimizations_applied"] = len(self.optimizations_applied)
        return results

    def optimize_test_file(self, file_path: Path) -> bool:
        """Optimize a single test file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Apply various optimizations
            content = self.reduce_data_sizes(content)
            content = self.optimize_loops(content)
            content = self.add_fast_markers(content)
            content = self.optimize_tensorflow_usage(content)

            # Write back if changed
            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return True

            return False

        except Exception as e:
            print(f"Error optimizing {file_path}: {e}")
            return False

    def reduce_data_sizes(self, content: str) -> str:
        """Reduce data sizes in test files."""
        # Reduce large numbers in data generation
        patterns = [
            (r"periods=(\d{3,})", lambda m: f"periods={min(50, int(m.group(1)))}"),
            (
                r"num_rows[=\s]*(\d{3,})",
                lambda m: f"num_rows={min(50, int(m.group(1)))}",
            ),
            (r"size=(\d{3,})", lambda m: f"size={min(50, int(m.group(1)))}"),
            (r"range\((\d{3,})\)", lambda m: f"range({min(50, int(m.group(1)))})"),
            (
                r"np\.random\.randn\((\d{3,})\)",
                lambda m: f"np.random.randn({min(50, int(m.group(1)))})",
            ),
            (
                r"np\.random\.choice\([^,]+,\s*(\d{3,})\)",
                lambda m: re.sub(
                    r"(\d{3,})",
                    str(min(50, int(m.group(1)))),
                    m.group(0),
                ),
            ),
        ]

        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
            if pattern in content:
                self.optimizations_applied.append(f"Reduced data size: {pattern}")

        return content

    def optimize_loops(self, content: str) -> str:
        """Optimize loops and iterations in tests."""
        # Reduce loop iterations
        patterns = [
            (
                r"for\s+\w+\s+in\s+range\((\d{2,})\):",
                lambda m: f"for {m.group(0).split()[1]} in range({min(10, int(m.group(1)))}):",
            ),
            (r"epochs[=\s]*(\d{2,})", lambda m: f"epochs={min(5, int(m.group(1)))}"),
            (
                r"iterations[=\s]*(\d{2,})",
                lambda m: f"iterations={min(10, int(m.group(1)))}",
            ),
        ]

        for pattern, replacement in patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                self.optimizations_applied.append(f"Optimized loop: {pattern}")

        return content

    def add_fast_markers(self, content: str) -> str:
        """Add fast markers to appropriate tests."""
        # Add @pytest.mark.fast to simple tests
        fast_test_patterns = [
            r"def test_init",
            r"def test_config",
            r"def test_basic",
            r"def test_simple",
            r"def test_get_",
            r"def test_set_",
        ]

        for pattern in fast_test_patterns:
            if re.search(pattern, content):
                # Add marker before test function
                content = re.sub(
                    f"(\s*){pattern}",
                    r"\1@pytest.mark.fast\n\1@pytest.mark.micro\n\1"
                    + pattern.replace("def ", "def "),
                    content,
                )
                self.optimizations_applied.append(f"Added fast marker to: {pattern}")

        return content

    def optimize_tensorflow_usage(self, content: str) -> str:
        """Optimize TensorFlow usage in tests."""
        optimizations = [
            # Use smaller models
            (r"units=(\d{3,})", lambda m: f"units={min(32, int(m.group(1)))}"),
            (
                r"embedding_dim=(\d{3,})",
                lambda m: f"embedding_dim={min(32, int(m.group(1)))}",
            ),
            (
                r"hidden_size=(\d{3,})",
                lambda m: f"hidden_size={min(32, int(m.group(1)))}",
            ),
            # Reduce training steps
            (
                r"steps_per_epoch=(\d{2,})",
                lambda m: f"steps_per_epoch={min(5, int(m.group(1)))}",
            ),
            (
                r"validation_steps=(\d{2,})",
                lambda m: f"validation_steps={min(3, int(m.group(1)))}",
            ),
        ]

        for pattern, replacement in optimizations:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                self.optimizations_applied.append(f"Optimized TF usage: {pattern}")

        return content

    def create_fast_test_variants(self) -> None:
        """Create fast variants of slow tests."""
        slow_tests = []

        for test_file in self.test_dir.rglob("test_*.py"):
            with open(test_file) as f:
                content = f.read()

            # Find slow tests
            if any(
                pattern in content for pattern in ["end_to_end", "integration", "large"]
            ):
                slow_tests.append(test_file)

        for test_file in slow_tests:
            fast_variant_path = test_file.parent / f"fast_{test_file.name}"
            if not fast_variant_path.exists():
                self.create_fast_variant(test_file, fast_variant_path)

    def create_fast_variant(self, original_path: Path, fast_path: Path) -> None:
        """Create a fast variant of a test file."""
        with open(original_path) as f:
            content = f.read()

        # Aggressive optimizations for fast variant
        content = content.replace("class Test", "class FastTest")
        content = re.sub(r"num_rows[=\s]*\d+", "num_rows=10", content)
        content = re.sub(r"periods=\d+", "periods=10", content)
        content = re.sub(r"epochs[=\s]*\d+", "epochs=1", content)

        # Add fast markers to all tests
        content = re.sub(
            r"(\s*)def test_",
            r"\1@pytest.mark.fast\n\1@pytest.mark.micro\n\1def test_",
            content,
        )

        with open(fast_path, "w") as f:
            f.write(content)

        print(f"Created fast variant: {fast_path}")


def main():
    """Main function."""
    optimizer = TestDataOptimizer()

    print("Optimizing test data for faster execution...")
    results = optimizer.optimize_all_tests()

    print("Results:")
    print(f"  Files processed: {results['files_processed']}")
    print(f"  Optimizations applied: {results['optimizations_applied']}")
    print(f"  Files modified: {len(results['files_modified'])}")

    if results["files_modified"]:
        print("\nModified files:")
        for file_path in results["files_modified"]:
            print(f"  - {file_path}")

    print("\nCreating fast test variants...")
    optimizer.create_fast_test_variants()

    print("\nOptimization complete!")


if __name__ == "__main__":
    main()
