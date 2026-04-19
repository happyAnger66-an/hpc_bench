#!/usr/bin/env python3
"""Batch evaluation script for hpc_bench datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hpc_bench.cli import _load_problem
from hpc_bench.core.data import Definition, Solution, Workload
from hpc_bench.core.data.json_utils import load_json_file, load_jsonl_file, save_jsonl_file
from hpc_bench.core.data.trace import EvaluationStatus
from hpc_bench.driver import ProblemPackager


def find_problems(dataset_dir: Path, categories: list[str] | None = None) -> list[Path]:
    """Find all problem directories in the dataset."""
    problems = []

    if (dataset_dir / "definition.json").exists():
        # Single problem directory
        return [dataset_dir]

    # Dataset with categories
    for category_dir in dataset_dir.iterdir():
        if not category_dir.is_dir():
            continue
        if categories and category_dir.name not in categories:
            continue

        for problem_dir in category_dir.iterdir():
            if problem_dir.is_dir() and (problem_dir / "definition.json").exists():
                problems.append(problem_dir)

    return sorted(problems)


def auto_create_solution(solution_path: Path, definition: Definition) -> Solution:
    """Auto-create a solution from a Python file or use reference."""
    if solution_path.suffix == ".py":
        # Wrap Python file into solution
        source_code = solution_path.read_text()
        solution_name = solution_path.stem

        # Detect language from imports
        languages = ["pytorch"]
        if "triton" in source_code.lower():
            languages = ["triton"]
        elif "cutlass" in source_code.lower() or "cute" in source_code.lower():
            languages = ["cute_dsl"]

        return Solution(
            name=solution_name,
            definition=definition.name,
            author="auto",
            spec={
                "languages": languages,
                "target_hardware": ["LOCAL"],
                "entry_point": f"{solution_path.name}::run",
                "destination_passing_style": True,
                "dependencies": ["torch"],
            },
            sources=[
                {
                    "path": solution_path.name,
                    "content": source_code,
                }
            ],
        )
    else:
        # Load JSON solution
        return Solution(**load_json_file(solution_path))


def main() -> int:
    """Main entry point for batch evaluation."""
    parser = argparse.ArgumentParser(
        description="Run hpc_bench problems using reference implementations."
    )
    parser.add_argument(
        "problems_dir",
        type=Path,
        help="Path to a single problem directory or a dataset root with category sub-directories",
    )
    parser.add_argument(
        "--category",
        nargs="+",
        choices=["L1", "L2", "FlashInfer-Bench", "Quant"],
        help="Restrict to one or more categories",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Max number of problems to evaluate",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./out"),
        help="Output directory for traces and summary (default: ./out)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-problem GPU evaluation timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--max-workloads",
        type=int,
        help="Max number of workloads per problem",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of timing iterations per workload (default: 50)",
    )
    parser.add_argument(
        "--solution-name",
        type=str,
        help="Filename to look for in each problem directory as the solution (e.g. solution.py, solution.json). .py files are wrapped into a solution JSON automatically",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-evaluate problems that already have results",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output",
    )

    args = parser.parse_args()

    # Find problems
    problems = find_problems(args.problems_dir, args.category)
    if args.limit:
        problems = problems[: args.limit]

    print(f"Found {len(problems)} problems to evaluate")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Track results
    results_summary = []
    total_passed = 0
    total_failed = 0

    for problem_dir in problems:
        problem_name = problem_dir.name
        category = problem_dir.parent.name if problem_dir.parent != args.problems_dir else "default"

        print(f"\n[{category}/{problem_name}] Evaluating...")

        # Check if already evaluated
        result_file = args.output / f"{category}_{problem_name}.jsonl"
        if result_file.exists() and not args.rerun:
            print(f"  Skipping (already evaluated, use --rerun to force)")
            continue

        # Load definition and workloads
        try:
            definition_path = problem_dir / "definition.json"
            workload_path = problem_dir / "workload.jsonl"
            definition = Definition(**load_json_file(definition_path))
            workloads = [Workload(**w) for w in load_jsonl_file(workload_path)]

            if args.max_workloads:
                workloads = workloads[: args.max_workloads]
        except Exception as e:
            print(f"  [ERROR] Failed to load problem: {e}")
            results_summary.append({
                "problem": problem_name,
                "category": category,
                "status": "LOAD_ERROR",
                "error": str(e),
            })
            total_failed += 1
            continue

        # Get solution
        if args.solution_name:
            solution_path = problem_dir / args.solution_name
            if not solution_path.exists():
                print(f"  [ERROR] Solution not found: {solution_path}")
                continue
            try:
                solution = auto_create_solution(solution_path, definition)
            except Exception as e:
                print(f"  [ERROR] Failed to create solution: {e}")
                continue
        else:
            # Use reference as solution
            solution = Solution(
                name=f"{problem_name}_reference",
                definition=definition.name,
                author="reference",
                spec={
                    "languages": ["pytorch"],
                    "target_hardware": ["LOCAL"],
                    "entry_point": "_reference.py::run",
                    "destination_passing_style": False,
                    "dependencies": ["torch"],
                },
                sources=[
                    {
                        "path": "_reference.py",
                        "content": definition.reference,
                    }
                ],
            )

        # Evaluate
        import tempfile
        staging_dir = Path(tempfile.mkdtemp(prefix="hpc_bench_"))

        try:
            packager = ProblemPackager(
                definition=definition,
                workloads=workloads,
                solution=solution,
                output_dir=staging_dir,
            )

            packager.package()
            success, traces = packager.execute(timeout=args.timeout)

            # Save traces
            save_jsonl_file(result_file, traces)

            # Count results
            passed = sum(
                1 for t in traces
                if t.get("evaluation", {}).get("status") == EvaluationStatus.PASSED.value
            )
            failed = len(traces) - passed

            print(f"  Passed: {passed}/{len(traces)}, Failed: {failed}/{len(traces)}")

            results_summary.append({
                "problem": problem_name,
                "category": category,
                "status": "PASSED" if failed == 0 else "FAILED",
                "passed": passed,
                "failed": failed,
                "total": len(traces),
            })

            if failed == 0:
                total_passed += 1
            else:
                total_failed += 1

        except Exception as e:
            print(f"  [ERROR] Evaluation failed: {e}")
            results_summary.append({
                "problem": problem_name,
                "category": category,
                "status": "ERROR",
                "error": str(e),
            })
            total_failed += 1

        finally:
            import shutil
            shutil.rmtree(staging_dir, ignore_errors=True)

    # Save summary
    summary_file = args.output / "summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "total_problems": len(problems),
            "passed": total_passed,
            "failed": total_failed,
            "results": results_summary,
        }, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Summary: {total_passed} passed, {total_failed} failed out of {len(problems)} problems")
    print(f"Results saved to {args.output}")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
