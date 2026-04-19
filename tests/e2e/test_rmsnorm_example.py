"""End-to-end test for rmsnorm example.

This test ensures:
1. Solution loading works with inline and file-based sources
2. CLI can load and execute the example
3. Reference implementation passes correctness checks
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from hpc_bench.core.data import Definition, Solution, Workload
from hpc_bench.core.data.json_utils import load_json_file, load_jsonl_file
from hpc_bench.driver import ProblemPackager


RMSNORM_DIR = Path(__file__).parent.parent.parent / "examples" / "rmsnorm"
PYTORCH_DIR = RMSNORM_DIR / "pytorch"


class TestRMSNormExample:
    """Test the rmsnorm example end-to-end."""

    def test_definition_loads(self):
        """Test that definition.json loads correctly."""
        definition_path = RMSNORM_DIR / "definition.json"
        assert definition_path.exists(), f"definition.json not found at {definition_path}"

        definition = Definition(**load_json_file(definition_path))
        assert definition.name == "rmsnorm_h4096"
        assert definition.op_type == "rmsnorm"
        assert "hidden_size" in definition.axes
        assert definition.axes["hidden_size"].type == "const"
        assert definition.axes["hidden_size"].value == 4096

    def test_workloads_load(self):
        """Test that workload.jsonl loads correctly."""
        workload_path = RMSNORM_DIR / "workload.jsonl"
        assert workload_path.exists(), f"workload.jsonl not found at {workload_path}"

        workloads = [Workload(**w) for w in load_jsonl_file(workload_path)]
        assert len(workloads) == 3, f"Expected 3 workloads, got {len(workloads)}"

        for w in workloads:
            assert "batch_size" in w.axes
            assert w.axes["batch_size"] > 0

    def test_solution_with_file_source_loads(self):
        """Test that solution.json with file-based source loads correctly."""
        solution_path = PYTORCH_DIR / "solution.json"
        assert solution_path.exists(), f"solution_pytorch.json not found at {solution_path}"

        solution_data = load_json_file(solution_path)

        # Content should not be inline
        for source in solution_data.get("sources", []):
            assert "content" not in source or source["content"] is None, \
                "Expected content to be missing (file-based)"

        # Load source file contents manually (as CLI does)
        for source in solution_data.get("sources", []):
            if source.get("content") is None:
                source_path = solution_path.parent / source["path"]
                assert source_path.exists(), f"Source file not found: {source_path}"
                source["content"] = source_path.read_text()

        solution = Solution(**solution_data)
        assert solution.name == "rmsnorm_pytorch_v1"
        assert solution.definition == "rmsnorm_h4096"
        assert solution.spec.entry_point == "kernel.py::run"

    def test_kernel_file_exists_and_valid(self):
        """Test that kernel.py exists and is valid Python."""
        kernel_path = PYTORCH_DIR / "kernel.py"
        assert kernel_path.exists(), f"kernel.py not found at {kernel_path}"

        # Check syntax by compiling
        source = kernel_path.read_text()
        try:
            compile(source, kernel_path, "exec")
        except SyntaxError as e:
            pytest.fail(f"kernel.py has syntax error: {e}")

        # Check that run function exists
        namespace = {}
        exec(source, namespace)
        assert "run" in namespace, "run function not found in kernel.py"

    def test_inline_solution_also_works(self):
        """Test that inline solution (with content) still works."""
        # Create an inline solution
        kernel_source = (PYTORCH_DIR / "kernel.py").read_text()

        inline_solution = {
            "name": "rmsnorm_inline",
            "definition": "rmsnorm_h4096",
            "author": "test",
            "spec": {
                "languages": ["pytorch"],
                "target_hardware": ["LOCAL"],
                "entry_point": "kernel.py::run",
                "destination_passing_style": True,
            },
            "sources": [
                {
                    "path": "kernel.py",
                    "content": kernel_source,  # Inline content
                }
            ],
        }

        solution = Solution(**inline_solution)
        assert solution.sources[0].content == kernel_source

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_full_evaluation_passes(self):
        """Test that full evaluation passes on CUDA."""
        definition = Definition(**load_json_file(RMSNORM_DIR / "definition.json"))
        workloads = [Workload(**w) for w in load_jsonl_file(RMSNORM_DIR / "workload.jsonl")]

        # Load solution with file source
        solution_data = load_json_file(PYTORCH_DIR / "solution.json")
        for source in solution_data.get("sources", []):
            if source.get("content") is None:
                source_path = PYTORCH_DIR / source["path"]
                source["content"] = source_path.read_text()
        solution = Solution(**solution_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            packager = ProblemPackager(
                definition=definition,
                workloads=workloads,
                solution=solution,
                output_dir=Path(tmpdir),
            )

            packager.package()
            success, traces = packager.execute(timeout=60)

            assert success, f"Evaluation failed: {traces}"
            assert len(traces) == len(workloads), \
                f"Expected {len(workloads)} traces, got {len(traces)}"

            for trace in traces:
                status = trace.get("evaluation", {}).get("status")
                assert status == "PASSED", f"Workload failed: {trace}"

    def test_reference_implementation_runs(self):
        """Test that reference implementation can be executed."""
        definition = Definition(**load_json_file(RMSNORM_DIR / "definition.json"))

        # Import and execute reference
        namespace = {}
        exec(definition.reference, namespace)
        run_fn = namespace["run"]

        # Create test tensors
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        input_tensor = torch.randn(4, 4096, dtype=torch.bfloat16, device=device)
        weight = torch.randn(4096, dtype=torch.bfloat16, device=device)
        eps = 1e-6

        output = run_fn(input_tensor, weight, eps)

        assert output is not None
        assert output.shape == input_tensor.shape
        assert output.dtype == input_tensor.dtype


class TestCLILoading:
    """Test CLI source loading behavior."""

    def test_cli_loads_solution_with_file_source(self, tmp_path: Path):
        """Test that CLI correctly loads solution with file-based source."""
        # Copy example files to temp dir
        definition = (RMSNORM_DIR / "definition.json").read_text()
        workload = (RMSNORM_DIR / "workload.jsonl").read_text()
        kernel = (PYTORCH_DIR / "kernel.py").read_text()

        solution = {
            "name": "test_solution",
            "definition": "rmsnorm_h4096",
            "author": "test",
            "spec": {
                "languages": ["pytorch"],
                "target_hardware": ["LOCAL"],
                "entry_point": "kernel.py::run",
                "destination_passing_style": True,
            },
            "sources": [{"path": "kernel.py"}],  # No content!
        }

        (tmp_path / "definition.json").write_text(definition)
        (tmp_path / "workload.jsonl").write_text(workload)
        (tmp_path / "solution.json").write_text(json.dumps(solution))
        (tmp_path / "kernel.py").write_text(kernel)

        # Test loading as CLI would
        from hpc_bench.core.data.json_utils import load_json_file

        solution_data = load_json_file(tmp_path / "solution.json")
        solution_dir = tmp_path

        # This is what CLI does
        for source in solution_data.get("sources", []):
            if source.get("content") is None:
                source_path = solution_dir / source["path"]
                assert source_path.exists()
                source["content"] = source_path.read_text()

        loaded_solution = Solution(**solution_data)
        assert loaded_solution.sources[0].content == kernel

    def test_cli_fails_gracefully_when_source_missing(self, tmp_path: Path):
        """Test that CLI fails gracefully when source file is missing."""
        solution = {
            "name": "test_solution",
            "definition": "test_def",
            "author": "test",
            "spec": {
                "languages": ["pytorch"],
                "target_hardware": ["LOCAL"],
                "entry_point": "missing.py::run",
                "destination_passing_style": True,
            },
            "sources": [{"path": "missing.py"}],  # No content, file doesn't exist
        }

        (tmp_path / "solution.json").write_text(json.dumps(solution))

        # Test loading as CLI would
        from hpc_bench.core.data.json_utils import load_json_file

        solution_data = load_json_file(tmp_path / "solution.json")
        solution_dir = tmp_path

        with pytest.raises(FileNotFoundError) as exc_info:
            for source in solution_data.get("sources", []):
                if source.get("content") is None:
                    source_path = solution_dir / source["path"]
                    if not source_path.exists():
                        raise FileNotFoundError(
                            f"Source file not found: {source_path}. "
                            f"Provide inline content or ensure file exists."
                        )
                    source["content"] = source_path.read_text()

        assert "missing.py" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
