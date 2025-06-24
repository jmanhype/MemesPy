"""Script to run quality assurance checks."""

from typing import Dict, Any, List, Optional
import sys
import subprocess
import yaml
import json
from pathlib import Path
import logging
from datetime import datetime
import asyncio
import aiohttp
import time
import statistics
import os

logger = logging.getLogger(__name__)


class QAChecker:
    """Quality assurance checker for the DSPy Meme Generation project."""

    def __init__(self, config_path: str = "qa/config.yml") -> None:
        """Initialize QA checker.

        Args:
            config_path: Path to QA configuration file
        """
        self.config = self._load_config(config_path)
        self.results: Dict[str, Any] = {}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load QA configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        with open(config_path) as f:
            return yaml.safe_load(f)

    async def run_all_checks(self) -> bool:
        """Run all QA checks.

        Returns:
            bool: True if all checks pass
        """
        type_check_result = await self.run_type_checking()
        lint_result = await self.run_linting()
        coverage_result = await self.run_coverage()
        benchmark_result = await self.run_benchmarks()
        quality_result = await self.run_quality_metrics()

        # Aggregate results
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "type_checking": type_check_result,
            "linting": lint_result,
            "coverage": coverage_result,
            "benchmarks": benchmark_result,
            "quality_metrics": quality_result,
            "overall_status": all(
                [
                    type_check_result["passed"],
                    lint_result["passed"],
                    coverage_result["passed"],
                    benchmark_result["passed"],
                    quality_result["passed"],
                ]
            ),
        }

        # Save results
        self._save_results()

        return self.results["overall_status"]

    async def run_type_checking(self) -> Dict[str, Any]:
        """Run type checking with mypy.

        Returns:
            Check results
        """
        logger.info("Running type checking...")

        try:
            # Create mypy config file
            mypy_config = self.config["type_checking"]["mypy"]
            with open("mypy.ini", "w") as f:
                f.write("[mypy]\n")
                for key, value in mypy_config.items():
                    if isinstance(value, list):
                        f.write(f"{key} = {','.join(value)}\n")
                    else:
                        f.write(f"{key} = {value}\n")

            # Run mypy
            process = await asyncio.create_subprocess_exec(
                "mypy", "src", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            return {
                "passed": process.returncode == 0,
                "output": stdout.decode(),
                "errors": stderr.decode() if stderr else None,
            }

        except Exception as e:
            logger.error(f"Type checking failed: {str(e)}")
            return {"passed": False, "error": str(e)}

    async def run_linting(self) -> Dict[str, Any]:
        """Run linting with ruff.

        Returns:
            Check results
        """
        logger.info("Running linting...")

        try:
            # Create ruff config file
            ruff_config = self.config["linting"]["ruff"]
            with open("ruff.toml", "w") as f:
                toml_content = "\n".join(
                    [f"{key} = {json.dumps(value)}" for key, value in ruff_config.items()]
                )
                f.write(toml_content)

            # Run ruff
            process = await asyncio.create_subprocess_exec(
                "ruff",
                "check",
                "src",
                "tests",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            return {
                "passed": process.returncode == 0,
                "output": stdout.decode(),
                "errors": stderr.decode() if stderr else None,
            }

        except Exception as e:
            logger.error(f"Linting failed: {str(e)}")
            return {"passed": False, "error": str(e)}

    async def run_coverage(self) -> Dict[str, Any]:
        """Run code coverage checks.

        Returns:
            Check results
        """
        logger.info("Running coverage checks...")

        try:
            # Create coverage config
            coverage_config = self.config["coverage"]
            with open(".coveragerc", "w") as f:
                f.write("[run]\n")
                f.write("branch = true\n")
                f.write(f"omit = {','.join(coverage_config['exclude_paths'])}\n")

                f.write("\n[report]\n")
                f.write(f"fail_under = {coverage_config['fail_under']}\n")
                f.write("show_missing = true\n")

            # Run pytest with coverage
            process = await asyncio.create_subprocess_exec(
                "pytest",
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html",
                "--cov-report=xml",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            # Parse coverage report
            coverage_info = self._parse_coverage_report("coverage.xml")

            return {
                "passed": coverage_info["total_coverage"] >= coverage_config["minimum_coverage"],
                "coverage": coverage_info,
                "output": stdout.decode(),
                "errors": stderr.decode() if stderr else None,
            }

        except Exception as e:
            logger.error(f"Coverage check failed: {str(e)}")
            return {"passed": False, "error": str(e)}

    async def run_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks.

        Returns:
            Benchmark results
        """
        logger.info("Running performance benchmarks...")

        try:
            benchmark_config = self.config["benchmarks"]
            results = {
                "meme_generation": await self._benchmark_meme_generation(),
                "database": await self._benchmark_database(),
                "redis": await self._benchmark_redis(),
                "external_services": await self._benchmark_external_services(),
            }

            # Check if all benchmarks pass their thresholds
            passed = all(
                [
                    results["meme_generation"]["p95"]
                    <= benchmark_config["meme_generation"]["max_latency_p95"],
                    results["meme_generation"]["p99"]
                    <= benchmark_config["meme_generation"]["max_latency_p99"],
                    results["database"]["p95"]
                    <= benchmark_config["database"]["max_query_time_p95"],
                    results["redis"]["p95"] <= benchmark_config["redis"]["max_latency_p95"],
                    all(
                        service["latency"]
                        <= benchmark_config["external_services"][service_name]["max_latency"]
                        for service_name, service in results["external_services"].items()
                    ),
                ]
            )

            return {"passed": passed, "results": results}

        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            return {"passed": False, "error": str(e)}

    async def run_quality_metrics(self) -> Dict[str, Any]:
        """Run code quality metric checks.

        Returns:
            Quality check results
        """
        logger.info("Running quality metrics checks...")

        try:
            quality_config = self.config["quality_metrics"]
            results = {
                "complexity": await self._check_complexity(),
                "documentation": await self._check_documentation(),
                "dependencies": await self._check_dependencies(),
                "testing": await self._check_testing(),
            }

            # Check if all quality metrics pass their thresholds
            passed = all(
                [
                    results["complexity"]["passed"],
                    results["documentation"]["passed"],
                    results["dependencies"]["passed"],
                    results["testing"]["passed"],
                ]
            )

            return {"passed": passed, "results": results}

        except Exception as e:
            logger.error(f"Quality metrics check failed: {str(e)}")
            return {"passed": False, "error": str(e)}

    def _parse_coverage_report(self, report_path: str) -> Dict[str, Any]:
        """Parse coverage XML report.

        Args:
            report_path: Path to coverage XML report

        Returns:
            Coverage information
        """
        import xml.etree.ElementTree as ET

        tree = ET.parse(report_path)
        root = tree.getroot()

        return {
            "total_coverage": float(root.attrib["line-rate"]) * 100,
            "branch_coverage": float(root.attrib["branch-rate"]) * 100,
            "covered_lines": int(root.attrib["lines-covered"]),
            "total_lines": int(root.attrib["lines-valid"]),
        }

    async def _benchmark_meme_generation(self) -> Dict[str, float]:
        """Run meme generation benchmarks.

        Returns:
            Benchmark metrics
        """
        logger.info("Running meme generation benchmarks...")

        from ..src.dspy_meme_gen.pipeline import MemeGenerationPipeline

        # Create test pipeline
        pipeline = MemeGenerationPipeline()
        latencies = []

        # Run benchmark requests
        for _ in range(100):  # 100 test requests
            start_time = time.time()
            try:
                await pipeline({"topic": "python programming", "style": "funny"})
                latency = (time.time() - start_time) * 1000  # Convert to ms
                latencies.append(latency)
            except Exception as e:
                logger.error(f"Benchmark request failed: {str(e)}")

        # Calculate metrics
        latencies.sort()
        return {
            "p95": statistics.quantiles(latencies, n=20)[18],  # 95th percentile
            "p99": statistics.quantiles(latencies, n=100)[98],  # 99th percentile
            "mean": statistics.mean(latencies),
            "throughput": len(latencies) / (sum(latencies) / 1000),  # requests per second
        }

    async def _benchmark_database(self) -> Dict[str, float]:
        """Run database benchmarks.

        Returns:
            Benchmark metrics
        """
        logger.info("Running database benchmarks...")

        from sqlalchemy import create_engine, text
        from sqlalchemy.pool import QueuePool

        # Get database URL from environment or config
        db_url = os.getenv("DATABASE_URL", "sqlite:///test.db")
        engine = create_engine(db_url, poolclass=QueuePool, pool_size=20, max_overflow=10)

        query_times = []
        pool_usage = []

        # Run benchmark queries
        async with engine.connect() as conn:
            for _ in range(100):
                start_time = time.time()
                try:
                    await conn.execute(text("SELECT 1"))
                    query_times.append((time.time() - start_time) * 1000)
                except Exception as e:
                    logger.error(f"Database query failed: {str(e)}")

                # Check pool usage
                pool_usage.append(engine.pool.size() / engine.pool.maxsize * 100)

        query_times.sort()
        return {
            "p95": statistics.quantiles(query_times, n=20)[18],
            "p99": statistics.quantiles(query_times, n=100)[98],
            "mean": statistics.mean(query_times),
            "max_pool_usage": max(pool_usage),
        }

    async def _benchmark_redis(self) -> Dict[str, float]:
        """Run Redis benchmarks.

        Returns:
            Benchmark metrics
        """
        logger.info("Running Redis benchmarks...")

        import aioredis

        # Get Redis URL from environment or config
        redis_url = os.getenv("REDIS_URL", "redis://localhost")
        redis = await aioredis.from_url(redis_url)

        latencies = []
        memory_usage = []

        # Run benchmark operations
        for _ in range(100):
            start_time = time.time()
            try:
                await redis.set("benchmark_key", "value")
                await redis.get("benchmark_key")
                latencies.append((time.time() - start_time) * 1000)

                # Check memory usage
                info = await redis.info("memory")
                used_memory = info["used_memory"]
                max_memory = info["maxmemory"] if info["maxmemory"] != 0 else used_memory * 2
                memory_usage.append(used_memory / max_memory * 100)
            except Exception as e:
                logger.error(f"Redis operation failed: {str(e)}")

        await redis.close()

        latencies.sort()
        return {
            "p95": statistics.quantiles(latencies, n=20)[18],
            "p99": statistics.quantiles(latencies, n=100)[98],
            "mean": statistics.mean(latencies),
            "max_memory_usage": max(memory_usage),
        }

    async def _benchmark_external_services(self) -> Dict[str, Dict[str, float]]:
        """Run external services benchmarks.

        Returns:
            Benchmark metrics
        """
        logger.info("Running external services benchmarks...")

        async with aiohttp.ClientSession() as session:
            results = {}

            # OpenAI benchmarks
            openai_latencies = []
            for _ in range(10):  # Fewer requests due to API limits
                start_time = time.time()
                try:
                    async with session.post(
                        "https://api.openai.com/v1/completions",
                        headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
                        json={
                            "model": "text-davinci-003",
                            "prompt": "Generate a meme caption",
                            "max_tokens": 50,
                        },
                    ) as response:
                        await response.json()
                        openai_latencies.append((time.time() - start_time) * 1000)
                except Exception as e:
                    logger.error(f"OpenAI request failed: {str(e)}")

            # Cloudinary benchmarks
            cloudinary_latencies = []
            for _ in range(10):
                start_time = time.time()
                try:
                    async with session.get(
                        "https://api.cloudinary.com/v1_1/demo/resources/image",
                        headers={"Authorization": f"Basic {os.getenv('CLOUDINARY_API_KEY')}"},
                    ) as response:
                        await response.json()
                        cloudinary_latencies.append((time.time() - start_time) * 1000)
                except Exception as e:
                    logger.error(f"Cloudinary request failed: {str(e)}")

            results["openai"] = {
                "latency": statistics.mean(openai_latencies) if openai_latencies else float("inf"),
                "success_rate": len(openai_latencies) / 10 * 100,
            }

            results["cloudinary"] = {
                "latency": (
                    statistics.mean(cloudinary_latencies) if cloudinary_latencies else float("inf")
                ),
                "success_rate": len(cloudinary_latencies) / 10 * 100,
            }

            return results

    async def _check_complexity(self) -> Dict[str, Any]:
        """Check code complexity metrics.

        Returns:
            Complexity check results
        """
        logger.info("Checking code complexity...")

        from radon.complexity import cc_visit
        from radon.metrics import mi_visit
        import ast

        complexity_config = self.config["quality_metrics"]["complexity"]
        results = {
            "passed": True,
            "violations": [],
            "metrics": {
                "max_cyclomatic": 0,
                "max_cognitive": 0,
                "min_maintainability": float("inf"),
            },
        }

        # Scan all Python files
        for path in Path("src").rglob("*.py"):
            with open(path) as f:
                code = f.read()

            # Check cyclomatic complexity
            try:
                complexity = cc_visit(code)
                for item in complexity:
                    if item.complexity > complexity_config["max_cyclomatic_complexity"]:
                        results["violations"].append(
                            {
                                "file": str(path),
                                "function": item.name,
                                "complexity": item.complexity,
                                "type": "cyclomatic",
                            }
                        )
                        results["passed"] = False
                    results["metrics"]["max_cyclomatic"] = max(
                        results["metrics"]["max_cyclomatic"], item.complexity
                    )
            except Exception as e:
                logger.error(f"Failed to check complexity for {path}: {str(e)}")

            # Check maintainability index
            try:
                mi_score = mi_visit(code, multi=True)
                if mi_score < complexity_config["max_maintainability_index"]:
                    results["violations"].append(
                        {
                            "file": str(path),
                            "maintainability_index": mi_score,
                            "type": "maintainability",
                        }
                    )
                    results["passed"] = False
                results["metrics"]["min_maintainability"] = min(
                    results["metrics"]["min_maintainability"], mi_score
                )
            except Exception as e:
                logger.error(f"Failed to check maintainability for {path}: {str(e)}")

        return results

    async def _check_documentation(self) -> Dict[str, Any]:
        """Check documentation coverage and quality.

        Returns:
            Documentation check results
        """
        logger.info("Checking documentation...")

        from pydocstyle import check
        import ast

        doc_config = self.config["quality_metrics"]["documentation"]
        results = {
            "passed": True,
            "violations": [],
            "metrics": {"docstring_coverage": 0, "total_functions": 0, "documented_functions": 0},
        }

        # Scan all Python files
        for path in Path("src").rglob("*.py"):
            try:
                # Check docstring presence and quality
                for error in check([str(path)]):
                    results["violations"].append(
                        {
                            "file": error.filename,
                            "line": error.line,
                            "message": error.message,
                            "type": "style",
                        }
                    )
                    results["passed"] = False

                # Check docstring coverage
                with open(path) as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        results["metrics"]["total_functions"] += 1
                        if ast.get_docstring(node):
                            results["metrics"]["documented_functions"] += 1
            except Exception as e:
                logger.error(f"Failed to check documentation for {path}: {str(e)}")

        # Calculate coverage
        if results["metrics"]["total_functions"] > 0:
            coverage = (
                results["metrics"]["documented_functions"]
                / results["metrics"]["total_functions"]
                * 100
            )
            results["metrics"]["docstring_coverage"] = coverage

            if coverage < doc_config["min_docstring_coverage"]:
                results["passed"] = False
                results["violations"].append(
                    {
                        "message": f"Docstring coverage {coverage:.1f}% below minimum {doc_config['min_docstring_coverage']}%",
                        "type": "coverage",
                    }
                )

        return results

    async def _check_dependencies(self) -> Dict[str, Any]:
        """Check dependency metrics.

        Returns:
            Dependency check results
        """
        logger.info("Checking dependencies...")

        import pkg_resources
        from safety.safety import check as safety_check

        dep_config = self.config["quality_metrics"]["dependencies"]
        results = {
            "passed": True,
            "violations": [],
            "metrics": {"total_dependencies": 0, "unpinned_dependencies": 0, "security_issues": 0},
        }

        # Check requirements.txt
        try:
            requirements = pkg_resources.parse_requirements(open("requirements.txt"))
            results["metrics"]["total_dependencies"] = len(list(requirements))

            # Check for unpinned versions
            if dep_config["require_pinned_versions"]:
                for req in requirements:
                    if not req.specs or not any(op in ["=="] for op, _ in req.specs):
                        results["violations"].append(
                            {
                                "package": req.name,
                                "message": "Version not pinned",
                                "type": "unpinned",
                            }
                        )
                        results["metrics"]["unpinned_dependencies"] += 1
                        results["passed"] = False

            # Check total dependencies
            if results["metrics"]["total_dependencies"] > dep_config["max_direct_dependencies"]:
                results["violations"].append(
                    {
                        "message": f"Too many direct dependencies ({results['metrics']['total_dependencies']})",
                        "type": "count",
                    }
                )
                results["passed"] = False

            # Security check
            if dep_config["security_check"]:
                vulns = safety_check(requirements)
                results["metrics"]["security_issues"] = len(vulns)
                for vuln in vulns:
                    results["violations"].append(
                        {
                            "package": vuln["package"],
                            "message": vuln["description"],
                            "type": "security",
                        }
                    )
                    results["passed"] = False
        except Exception as e:
            logger.error(f"Failed to check dependencies: {str(e)}")
            results["passed"] = False

        return results

    async def _check_testing(self) -> Dict[str, Any]:
        """Check testing metrics.

        Returns:
            Testing check results
        """
        logger.info("Checking testing metrics...")

        import pytest
        import hypothesis
        from typing import get_type_hints
        import ast

        test_config = self.config["quality_metrics"]["testing"]
        results = {
            "passed": True,
            "violations": [],
            "metrics": {
                "total_functions": 0,
                "tested_functions": 0,
                "property_tests": 0,
                "integration_tests": 0,
            },
        }

        # Collect all test files
        test_files = list(Path("tests").rglob("test_*.py"))

        # Check test coverage per function
        for path in Path("src").rglob("*.py"):
            try:
                with open(path) as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        results["metrics"]["total_functions"] += 1

                        # Count test cases for this function
                        test_count = 0
                        for test_file in test_files:
                            with open(test_file) as f:
                                if f"test_{node.name}" in f.read():
                                    test_count += 1

                        if test_count < test_config["min_test_cases_per_function"]:
                            results["violations"].append(
                                {
                                    "file": str(path),
                                    "function": node.name,
                                    "test_count": test_count,
                                    "type": "coverage",
                                }
                            )
                            results["passed"] = False
                        else:
                            results["metrics"]["tested_functions"] += 1
            except Exception as e:
                logger.error(f"Failed to check tests for {path}: {str(e)}")

        # Check for property-based tests
        if test_config["require_property_based_tests"]:
            for test_file in test_files:
                try:
                    with open(test_file) as f:
                        if "@given" in f.read():
                            results["metrics"]["property_tests"] += 1
                except Exception as e:
                    logger.error(f"Failed to check property tests in {test_file}: {str(e)}")

            if results["metrics"]["property_tests"] == 0:
                results["violations"].append(
                    {"message": "No property-based tests found", "type": "property"}
                )
                results["passed"] = False

        # Check for integration tests
        if test_config["require_integration_tests"]:
            integration_test_dir = Path("tests/integration")
            if not integration_test_dir.exists() or not any(integration_test_dir.iterdir()):
                results["violations"].append(
                    {"message": "No integration tests found", "type": "integration"}
                )
                results["passed"] = False
            else:
                results["metrics"]["integration_tests"] = len(
                    list(integration_test_dir.glob("test_*.py"))
                )

        return results

    def _save_results(self) -> None:
        """Save QA check results to file."""
        results_dir = Path("qa/results")
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"qa_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"QA results saved to {results_file}")


async def main() -> int:
    """Run QA checks.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run QA checks
    checker = QAChecker()
    success = await checker.run_all_checks()

    if success:
        logger.info("All QA checks passed!")
        return 0
    else:
        logger.error("Some QA checks failed. See results for details.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
