"""CLI entrypoint for hpc_bench."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from hpc_bench.core.data import Definition, Solution, Workload
from hpc_bench.core.data.json_utils import load_json_file, load_jsonl_file
from hpc_bench.core.data.trace import EvaluationStatus
from hpc_bench.driver import ProblemPackager


console = Console()


def _load_problem(
    problem_dir: Optional[Path],
    definition_path: Optional[Path],
    workload_path: Optional[Path],
) -> tuple[Definition, list[Workload]]:
    """Load definition and workloads from various input methods."""
    if problem_dir is not None:
        definition_path = problem_dir / "definition.json"
        workload_path = problem_dir / "workload.jsonl"

    if definition_path is None:
        raise click.UsageError(
            "Must provide either PROBLEM_DIR or --definition/--workload"
        )

    definition = Definition(**load_json_file(definition_path))

    workloads = []
    if workload_path is not None and workload_path.exists():
        workloads = [Workload(**w) for w in load_jsonl_file(workload_path)]

    return definition, workloads


@click.command()
@click.argument("problem_dir", required=False, type=click.Path(path_type=Path))
@click.option(
    "--definition",
    type=click.Path(path_type=Path),
    help="Path to definition.json",
)
@click.option(
    "--workload",
    type=click.Path(path_type=Path),
    help="Path to workload.jsonl",
)
@click.option(
    "--solution",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to solution.json",
)
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    help="Path to benchmark config JSON",
)
@click.option(
    "--compile-timeout",
    type=int,
    default=300,
    help="Compilation timeout in seconds (C++/CUDA only)",
)
@click.option(
    "--timeout",
    type=int,
    default=300,
    help="Evaluation timeout in seconds",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Write trace JSONL to this file",
)
@click.option(
    "--json",
    is_flag=True,
    help="Print trace JSON to stdout",
)
@click.option(
    "--lock-clocks",
    is_flag=True,
    help="Require GPU clocks to be locked",
)
@click.option(
    "--keep-staging",
    is_flag=True,
    help="Keep the staging directory after evaluation",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show verbose output",
)
def cli(
    problem_dir: Optional[Path],
    definition: Optional[Path],
    workload: Optional[Path],
    solution: Optional[Path],
    config: Optional[Path],
    compile_timeout: int,
    timeout: int,
    output: Optional[Path],
    json: bool,
    lock_clocks: bool,
    keep_staging: bool,
    verbose: bool,
) -> None:
    """Evaluate a hpc_bench solution on GPU.

    Two ways to specify the problem:
      1) Positional: hpc-bench <problem_dir> --solution sol.json
         (reads definition.json and workload.jsonl from problem_dir)
      2) Explicit:   hpc-bench --definition def.json --workload wkl.jsonl --solution sol.json
    """
    # Load inputs
    try:
        definition_obj, workloads = _load_problem(problem_dir, definition, workload)
    except Exception as e:
        console.print(f"[red]Error loading problem: {e}[/red]")
        raise click.Exit(1)

    if not workloads:
        console.print("[yellow]Warning: No workloads loaded[/yellow]")

    try:
        solution_obj = Solution(**load_json_file(solution))
    except Exception as e:
        console.print(f"[red]Error loading solution: {e}[/red]")
        raise click.Exit(1)

    # Validate solution matches definition
    if solution_obj.definition != definition_obj.name:
        console.print(
            f"[yellow]Warning: Solution targets '{solution_obj.definition}' "
            f"but definition is '{definition_obj.name}'[/yellow]"
        )

    # Load config
    config_dict = {}
    if config is not None:
        config_dict = load_json_file(config)
    if lock_clocks:
        config_dict["lock_clocks"] = True

    # Create staging directory
    if keep_staging and output is not None:
        staging_dir = output.parent / f"{output.stem}_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
    else:
        staging_dir = Path(tempfile.mkdtemp(prefix="hpc_bench_"))

    try:
        # Package and execute
        packager = ProblemPackager(
            definition=definition_obj,
            workloads=workloads,
            solution=solution_obj,
            output_dir=staging_dir,
        )

        packager.package()

        if verbose:
            console.print(f"[dim]Staging directory: {staging_dir}[/dim]")

        # Compile if needed
        try:
            packager.compile(timeout=compile_timeout)
        except Exception as e:
            console.print(f"[red]Compilation failed: {e}[/red]")
            raise click.Exit(1)

        # Execute
        try:
            success, traces = packager.execute(timeout=timeout)
        except Exception as e:
            console.print(f"[red]Execution failed: {e}[/red]")
            raise click.Exit(1)

        if not success:
            console.print("[red]Evaluation failed[/red]")
            raise click.Exit(1)

        # Output results
        if json:
            for trace in traces:
                click.echo(json.dumps(trace))

        if output is not None:
            with open(output, "w") as f:
                for trace in traces:
                    f.write(json.dumps(trace) + "\n")
            console.print(f"[green]Traces written to {output}[/green]")

        # Print summary table
        if not json:
            table = Table(title=f"Evaluation Results: {definition_obj.name}")
            table.add_column("Workload", style="cyan")
            table.add_column("Status", style="bold")
            table.add_column("Latency (ms)", justify="right")
            table.add_column("Speedup", justify="right")

            for trace in traces:
                workload_uuid = trace.get("workload", {}).get("uuid", "unknown")
                eval_data = trace.get("evaluation", {})
                status = eval_data.get("status", "UNKNOWN")
                perf = eval_data.get("performance", {})
                latency = perf.get("latency_ms", 0.0)
                speedup = perf.get("speedup_factor", 0.0)

                status_color = {
                    EvaluationStatus.PASSED.value: "green",
                    EvaluationStatus.INCORRECT_NUMERICAL.value: "red",
                    EvaluationStatus.INCORRECT_SHAPE.value: "red",
                    EvaluationStatus.INCORRECT_DTYPE.value: "red",
                    EvaluationStatus.RUNTIME_ERROR.value: "red",
                    EvaluationStatus.TIMEOUT.value: "yellow",
                }.get(status, "white")

                table.add_row(
                    workload_uuid[:8],
                    f"[{status_color}]{status}[/{status_color}]",
                    f"{latency:.3f}" if latency > 0 else "N/A",
                    f"{speedup:.2f}x" if speedup > 0 else "N/A",
                )

            console.print(table)

        # Exit with error if any workload failed
        all_passed = all(
            trace.get("evaluation", {}).get("status") == EvaluationStatus.PASSED.value
            for trace in traces
        )
        if not all_passed:
            raise click.Exit(1)

    finally:
        if not keep_staging:
            import shutil
            shutil.rmtree(staging_dir, ignore_errors=True)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
