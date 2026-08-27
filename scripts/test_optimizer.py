#!/usr/bin/env python3
"""
Test Optimization Script for KDP

This script provides utilities for:
- Analyzing test performance
- Identifying slow tests
- Suggesting test optimizations
- Running different test configurations
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree  # noqa: S405 - parses our own pytest output


class TestOptimizer:
    """Test optimization and analysis utilities."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.test_dir = self.project_root / "test"

    def analyze_test_performance(
        self,
        junit_xml_path: str = "pytest.xml",
    ) -> dict[str, Any]:
        """Analyze test performance from JUnit XML output."""
        if not Path(junit_xml_path).exists():
            print(f"JUnit XML file not found: {junit_xml_path}")
            return {}

        try:
            tree = ElementTree.parse(  # noqa: S314 - input is this script's own JUnit report
                junit_xml_path,
            )
            root = tree.getroot()

            test_results = []
            total_time = 0

            for testcase in root.findall(".//testcase"):
                name = testcase.get("name", "")
                classname = testcase.get("classname", "")
                time_taken = float(testcase.get("time", 0))

                test_results.append(
                    {
                        "name": name,
                        "classname": classname,
                        "time": time_taken,
                        "full_name": f"{classname}::{name}",
                    },
                )
                total_time += time_taken

            # Sort by time taken (slowest first)
            test_results.sort(key=lambda x: x["time"], reverse=True)

            return {
                "total_tests": len(test_results),
                "total_time": total_time,
                "average_time": total_time / len(test_results) if test_results else 0,
                "slowest_tests": test_results[:10],  # Top 10 slowest
                "fastest_tests": test_results[-10:],  # Top 10 fastest
                "slow_threshold": 5.0,  # Tests taking more than 5 seconds
                "slow_tests": [t for t in test_results if t["time"] > 5.0],
            }

        except Exception as e:
            print(f"Error analyzing test performance: {e}")
            return {}

    def suggest_optimizations(self, analysis: dict[str, Any]) -> list[str]:
        """Suggest optimizations based on test analysis."""
        suggestions = []

        if not analysis:
            return ["Run tests with --junitxml=pytest.xml to generate performance data"]

        slow_tests = analysis.get("slow_tests", [])
        if slow_tests:
            suggestions.append(f"Found {len(slow_tests)} slow tests (>5s):")
            for test in slow_tests[:5]:  # Show top 5
                suggestions.append(f"  - {test['full_name']}: {test['time']:.2f}s")
            suggestions.append("Consider:")
            suggestions.append("  - Adding @pytest.mark.slow to these tests")
            suggestions.append("  - Using fixtures to reduce setup time")
            suggestions.append("  - Mocking expensive operations")

        avg_time = analysis.get("average_time", 0)
        if avg_time > 2.0:
            suggestions.append(
                f"Average test time is {avg_time:.2f}s - consider optimization",
            )

        total_time = analysis.get("total_time", 0)
        if total_time > 300:  # 5 minutes
            suggestions.append(
                f"Total test time is {total_time:.2f}s - consider parallel execution",
            )

        return suggestions

    def run_test_suite(
        self,
        test_type: str = "all",
        parallel: bool = True,
        coverage: bool = False,
        verbose: bool = False,
    ) -> tuple[int, str]:
        """Run test suite with specified configuration."""
        cmd = ["poetry", "run", "pytest"]

        # Add parallel execution
        if parallel:
            cmd.extend(["-n", "auto"])

        # Add coverage
        if coverage:
            cmd.extend(
                ["--cov=kdp", "--cov-report=term-missing", "--cov-report=html:htmlcov"],
            )

        # Add verbosity
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        # Add test type markers
        if test_type != "all":
            cmd.extend(["-m", test_type])

        # Add performance tracking
        cmd.extend(["--durations=10", "--junitxml=pytest.xml"])

        print(f"Running command: {' '.join(cmd)}")
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            end_time = time.time()

            execution_time = end_time - start_time
            output = f"Execution time: {execution_time:.2f}s\n"
            output += f"Return code: {result.returncode}\n"
            output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"

            return result.returncode, output

        except Exception as e:
            return 1, f"Error running tests: {e}"

    def benchmark_test_configurations(self) -> dict[str, Any]:
        """Benchmark different test configurations."""
        configurations = [
            ("sequential", {"parallel": False, "coverage": False}),
            ("parallel", {"parallel": True, "coverage": False}),
            ("parallel_with_coverage", {"parallel": True, "coverage": True}),
            ("fast_only", {"parallel": True, "coverage": False, "test_type": "fast"}),
            ("unit_only", {"parallel": True, "coverage": False, "test_type": "unit"}),
        ]

        results = {}

        for config_name, config in configurations:
            print(f"\nBenchmarking configuration: {config_name}")
            start_time = time.time()

            returncode, output = self.run_test_suite(
                test_type=config.get("test_type", "all"),
                parallel=config.get("parallel", True),
                coverage=config.get("coverage", False),
                verbose=False,
            )

            end_time = time.time()
            execution_time = end_time - start_time

            results[config_name] = {
                "execution_time": execution_time,
                "returncode": returncode,
                "config": config,
                "success": returncode == 0,
            }

            print(
                f"Configuration {config_name}: {execution_time:.2f}s (exit code: {returncode})",
            )

        return results

    def generate_test_report(self, output_file: str = "test_optimization_report.json"):
        """Generate a comprehensive test optimization report."""
        print("Generating test optimization report...")

        # Run a quick analysis
        returncode, output = self.run_test_suite(
            parallel=True,
            coverage=False,
            verbose=False,
        )

        # Analyze performance
        analysis = self.analyze_test_performance()
        suggestions = self.suggest_optimizations(analysis)

        # Count test files
        test_files = list(self.test_dir.rglob("test_*.py"))

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "project_root": str(self.project_root),
            "test_files_count": len(test_files),
            "test_files": [str(f.relative_to(self.project_root)) for f in test_files],
            "performance_analysis": analysis,
            "optimization_suggestions": suggestions,
            "last_run_success": returncode == 0,
            "configuration_recommendations": {
                "parallel_execution": True,
                "recommended_markers": ["fast", "unit", "integration", "slow"],
                "timeout_seconds": 300,
                "coverage_threshold": 80,
            },
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Report saved to: {output_file}")
        return report


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Test Optimization Tool for KDP")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze test performance from existing results",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark different test configurations",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate comprehensive optimization report",
    )
    parser.add_argument(
        "--run",
        choices=["all", "fast", "unit", "integration", "slow"],
        help="Run specific test suite",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Enable parallel execution",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel execution",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable coverage reporting",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    optimizer = TestOptimizer()

    if args.analyze:
        print("Analyzing test performance...")
        analysis = optimizer.analyze_test_performance()
        if analysis:
            print(f"Total tests: {analysis['total_tests']}")
            print(f"Total time: {analysis['total_time']:.2f}s")
            print(f"Average time: {analysis['average_time']:.2f}s")
            print(f"Slow tests (>5s): {len(analysis['slow_tests'])}")

            suggestions = optimizer.suggest_optimizations(analysis)
            if suggestions:
                print("\nOptimization suggestions:")
                for suggestion in suggestions:
                    print(f"  {suggestion}")
        else:
            print(
                "No performance data available. Run tests with --junitxml=pytest.xml first.",
            )

    elif args.benchmark:
        print("Benchmarking test configurations...")
        results = optimizer.benchmark_test_configurations()

        print("\nBenchmark Results:")
        for config, result in results.items():
            status = "✓" if result["success"] else "✗"
            print(f"{status} {config}: {result['execution_time']:.2f}s")

    elif args.report:
        report = optimizer.generate_test_report()
        print("\nTest Optimization Report Summary:")
        print(f"Test files: {report['test_files_count']}")
        if report["performance_analysis"]:
            print(f"Total tests: {report['performance_analysis']['total_tests']}")
            print(f"Total time: {report['performance_analysis']['total_time']:.2f}s")
        print(f"Suggestions: {len(report['optimization_suggestions'])}")

    elif args.run:
        parallel = args.parallel and not args.no_parallel
        returncode, output = optimizer.run_test_suite(
            test_type=args.run,
            parallel=parallel,
            coverage=args.coverage,
            verbose=args.verbose,
        )

        print(output)
        sys.exit(returncode)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
