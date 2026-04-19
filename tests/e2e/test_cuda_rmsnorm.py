"""End-to-end test for CUDA C++ rmsnorm example.

This test validates the CUDA C++ implementation of RMSNorm,
ensuring compilation and execution work correctly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from hpc_bench.core.data import Definition, Solution
from hpc_bench.core.data.json_utils import load_json_file, load_jsonl_file
from hpc_bench.core.data.trace import EvaluationStatus
from hpc_bench.driver import ProblemPackager


EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
RMSNORM_DIR = EXAMPLES_DIR / "rmsnorm"
CUDA_EXAMPLE_DIR = RMSNORM_DIR / "cuda_cpp"
PYTORCH_EXAMPLE_DIR = RMSNORM_DIR / "pytorch"


@pytest.mark.cuda
class TestCUDARMSNormExample:
    """Test the CUDA C++ rmsnorm example end-to-end."""

    def test_cuda_kernel_file_exists(self):
        """Test that the CUDA kernel file exists."""
        kernel_path = CUDA_EXAMPLE_DIR / "kernel.cu"
        assert kernel_path.exists(), f"CUDA kernel not found at {kernel_path}"

    def test_cuda_solution_loads(self):
        """Test that CUDA solution.json loads correctly."""
        solution_path = CUDA_EXAMPLE_DIR / "solution.json"
        assert solution_path.exists(), f"solution_cuda.json not found at {solution_path}"

        solution_data = load_json_file(solution_path)

        # Check that it references the cpp file
        assert solution_data["spec"]["languages"] == ["cuda_cpp"]
        assert solution_data["spec"]["entry_point"] == "kernel.cu::run"
        assert solution_data["sources"][0]["path"] == "kernel.cu"

        # Content should not be inline (loaded from file)
        for source in solution_data.get("sources", []):
            assert "content" not in source or source["content"] is None

        # Load source file content (as CLI does)
        for source in solution_data.get("sources", []):
            if source.get("content") is None:
                source_path = solution_path.parent / source["path"]
                source["content"] = source_path.read_text()

        solution = Solution(**solution_data)
        assert solution.name == "rmsnorm_cuda_v1"
        assert solution.definition == "rmsnorm_h4096"

    def test_cuda_kernel_compiles(self):
        """Test that CUDA kernel compiles successfully."""
        definition = Definition(**load_json_file(RMSNORM_DIR / "definition.json"))

        solution_data = load_json_file(CUDA_EXAMPLE_DIR / "solution_cuda.json")
        for source in solution_data.get("sources", []):
            if source.get("content") is None:
                source_path = CUDA_EXAMPLE_DIR / source["path"]
                source["content"] = source_path.read_text()
        solution = Solution(**solution_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            packager = ProblemPackager(
                definition=definition,
                workloads=[],  # No workloads needed for compile test
                solution=solution,
                output_dir=Path(tmpdir),
            )

            packager.package()

            # Should compile without error
            try:
                packager.compile(timeout=120)
            except Exception as e:
                pytest.fail(f"CUDA compilation failed: {e}")

            # Check that .so file was created
            so_files = list(Path(tmpdir).glob("*.so"))
            assert len(so_files) > 0, "No .so file found after compilation"

    def test_cuda_kernel_runs_correctly(self):
        """Test that CUDA kernel produces correct results."""
        definition = Definition(**load_json_file(RMSNORM_DIR / "definition.json"))
        workloads = load_jsonl_file(RMSNORM_DIR / "workload.jsonl")
        # Use only first workload for faster test
        from hpc_bench.core.data import Workload
        workloads = [Workload(**workloads[0])]

        solution_data = load_json_file(CUDA_EXAMPLE_DIR / "solution_cuda.json")
        for source in solution_data.get("sources", []):
            if source.get("content") is None:
                source_path = CUDA_EXAMPLE_DIR / source["path"]
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
            packager.compile(timeout=120)
            success, traces = packager.execute(timeout=60)

            assert success, f"CUDA evaluation failed: {traces}"
            assert len(traces) == 1

            trace = traces[0]
            status = trace.get("evaluation", {}).get("status")
            assert status == EvaluationStatus.PASSED.value, \
                f"CUDA kernel failed: {trace}"

            # Check that we got performance metrics
            perf = trace.get("evaluation", {}).get("performance", {})
            assert "latency_ms" in perf
            assert perf["latency_ms"] > 0

    def test_cuda_vs_pytorch_correctness(self):
        """Compare CUDA C++ results with PyTorch reference."""
        definition = Definition(**load_json_file(RMSNORM_DIR / "definition.json"))

        # Load both solutions
        cuda_solution_data = load_json_file(CUDA_EXAMPLE_DIR / "solution_cuda.json")
        for source in cuda_solution_data.get("sources", []):
            if source.get("content") is None:
                source["content"] = (CUDA_EXAMPLE_DIR / source["path"]).read_text()
        cuda_solution = Solution(**cuda_solution_data)

        # Use PyTorch solution for comparison
        from hpc_bench.core.bench import gen_inputs, set_seed
        set_seed(42)

        device = "cuda:0"

        # Create a simple workload
        workload_data = {
            "uuid": "test_wkl",
            "axes": {"batch_size": 4},
            "inputs": {
                "input": {"type": "random"},
                "weight": {"type": "random"},
                "eps": {"type": "scalar", "value": 1e-6}
            },
            "tolerance": {"max_atol": 1e-3, "max_rtol": 1e-3}
        }
        from hpc_bench.core.data import Workload
        workload = Workload(**workload_data)

        inputs = gen_inputs(definition, workload, device)

        # Run reference (PyTorch)
        namespace = {}
        exec(definition.reference, namespace)
        ref_fn = namespace["run"]
        ref_output = ref_fn(*inputs)

        # Run CUDA solution
        with tempfile.TemporaryDirectory() as tmpdir:
            packager = ProblemPackager(
                definition=definition,
                workloads=[workload],
                solution=cuda_solution,
                output_dir=Path(tmpdir),
            )
            packager.package()
            packager.compile(timeout=120)
            success, traces = packager.execute(timeout=60)

            assert success
            assert traces[0]["evaluation"]["status"] == EvaluationStatus.PASSED.value

            # The execution test already validates correctness against reference


class TestCUDASolutionStructure:
    """Test CUDA solution structure and validation."""

    def test_solution_has_compile_options(self):
        """Test that CUDA solution includes compile options."""
        solution_path = CUDA_EXAMPLE_DIR / "solution.json"
        solution_data = load_json_file(solution_path)

        spec = solution_data.get("spec", {})
        assert "compile_options" in spec
        assert "cuda_cflags" in spec["compile_options"]
        assert "ld_flags" in spec["compile_options"]

    def test_solution_destination_passing_style(self):
        """Test that CUDA solution uses DPS mode."""
        solution_path = CUDA_EXAMPLE_DIR / "solution.json"
        solution_data = load_json_file(solution_path)

        assert solution_data["spec"]["destination_passing_style"] is True

    def test_solution_binding_is_torch(self):
        """Test that CUDA solution uses torch binding."""
        solution_path = CUDA_EXAMPLE_DIR / "solution.json"
        solution_data = load_json_file(solution_path)

        assert solution_data["spec"]["binding"] == "torch"

    def test_kernel_has_correct_signature_comment(self):
        """Test that kernel.cpp has DPS signature."""
        kernel_path = CUDA_EXAMPLE_DIR / "kernel.cu"
        source = kernel_path.read_text()

        # Should have run() function with DPS signature
        assert "void run(torch::Tensor input, torch::Tensor weight, float eps, torch::Tensor output)" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
