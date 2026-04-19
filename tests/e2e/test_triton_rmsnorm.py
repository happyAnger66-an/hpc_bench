"""E2E tests for Triton RMSNorm example."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import pytest
import torch

from hpc_bench.core.data import Definition, Solution, Workload
from hpc_bench.core.data.json_utils import load_json_file, load_jsonl_file
from hpc_bench.core.data.trace import EvaluationStatus
from hpc_bench.driver import ProblemPackager

RMSNORM_DIR = Path(__file__).parent.parent.parent / "examples" / "rmsnorm"
TRITON_DIR = RMSNORM_DIR / "triton"

pytest.importorskip("triton", reason="install triton: pip install 'hpc-bench[triton]' or pip install triton")


def _load_triton_solution() -> Solution:
    data = load_json_file(TRITON_DIR / "solution.json")
    for source in data.get("sources", []):
        if source.get("content") is None:
            source["content"] = (TRITON_DIR / source["path"]).read_text()
    return Solution(**data)


class TestTritonRMSNormStructure:
    """No GPU required."""

    def test_files_exist(self):
        assert (TRITON_DIR / "kernel.py").exists()
        assert (TRITON_DIR / "solution.json").exists()

    def test_solution_loads(self):
        sol = _load_triton_solution()
        assert sol.name == "rmsnorm_triton_v1"
        assert sol.spec.languages[0].value == "triton"
        assert sol.spec.entry_point == "kernel.py::run"
        assert sol.spec.destination_passing_style is True

    def test_kernel_imports(self):
        # Triton @jit needs a real file (inspect.getsourcelines); avoid exec().
        path = TRITON_DIR / "kernel.py"
        spec = importlib.util.spec_from_file_location("rmsnorm_triton_kernel", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        assert callable(mod.run)


@pytest.mark.triton
@pytest.mark.cuda
class TestTritonRMSNormGPU:
    """Requires CUDA + Triton."""

    def test_full_evaluation_first_workload(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        definition = Definition(**load_json_file(RMSNORM_DIR / "definition.json"))
        raw_wl = load_jsonl_file(RMSNORM_DIR / "workload.jsonl")
        workloads = [Workload(**raw_wl[0])]
        solution = _load_triton_solution()

        with tempfile.TemporaryDirectory() as tmpdir:
            packager = ProblemPackager(
                definition=definition,
                workloads=workloads,
                solution=solution,
                output_dir=Path(tmpdir),
            )
            packager.package()
            success, traces = packager.execute(timeout=120)

        assert success, traces
        assert len(traces) == 1
        ev = traces[0].get("evaluation") or {}
        assert ev.get("status") == EvaluationStatus.PASSED.value
        perf = ev.get("performance") or {}
        assert perf.get("latency_ms", 0) > 0

    def test_all_example_workloads(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        definition = Definition(**load_json_file(RMSNORM_DIR / "definition.json"))
        workloads = [Workload(**w) for w in load_jsonl_file(RMSNORM_DIR / "workload.jsonl")]
        solution = _load_triton_solution()

        with tempfile.TemporaryDirectory() as tmpdir:
            packager = ProblemPackager(
                definition=definition,
                workloads=workloads,
                solution=solution,
                output_dir=Path(tmpdir),
            )
            packager.package()
            success, traces = packager.execute(timeout=300)

        assert success
        assert len(traces) == len(workloads)
        for t in traces:
            assert (t.get("evaluation") or {}).get("status") == EvaluationStatus.PASSED.value
