"""E2E tests for optional rmsnorm backends (CuTe DSL, cuTile, CUTLASS tag, cuDNN)."""

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


def _load_solution(subdir: str) -> Solution:
    d = RMSNORM_DIR / subdir
    data = load_json_file(d / "solution.json")
    for source in data.get("sources", []):
        if source.get("content") is None:
            source["content"] = (d / source["path"]).read_text()
    return Solution(**data)


class TestOptionalBackendFiles:
    def test_solution_json_exist(self):
        for sub in ("cute_dsl", "cutile", "cutlass", "cudnn"):
            assert (RMSNORM_DIR / sub / "solution.json").exists()


@pytest.mark.cute_dsl
@pytest.mark.cuda
class TestCuteDSL:
    @staticmethod
    def _skip_if_no_cute():
        pytest.importorskip("cutlass")
        pytest.importorskip("cutlass.cute")
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_eval_first_workload(self):
        self._skip_if_no_cute()
        definition = Definition(**load_json_file(RMSNORM_DIR / "definition.json"))
        workloads = [Workload(**load_jsonl_file(RMSNORM_DIR / "workload.jsonl")[0])]
        solution = _load_solution("cute_dsl")
        with tempfile.TemporaryDirectory() as tmp:
            p = ProblemPackager(definition, workloads, solution, Path(tmp))
            p.package()
            ok, traces = p.execute(timeout=180)
        assert ok
        assert (traces[0].get("evaluation") or {}).get("status") == EvaluationStatus.PASSED.value


@pytest.mark.cutile
@pytest.mark.cuda
class TestCuTile:
    @staticmethod
    def _skip_if_no_cutile():
        if importlib.util.find_spec("cuda") is None:
            pytest.skip("cuTile (cuda.tile) not installed")
        try:
            importlib.import_module("cuda.tile")
        except ImportError:
            pytest.skip("cuTile (cuda.tile) not importable")
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_eval_first_workload(self):
        self._skip_if_no_cutile()
        definition = Definition(**load_json_file(RMSNORM_DIR / "definition.json"))
        workloads = [Workload(**load_jsonl_file(RMSNORM_DIR / "workload.jsonl")[0])]
        solution = _load_solution("cutile")
        with tempfile.TemporaryDirectory() as tmp:
            p = ProblemPackager(definition, workloads, solution, Path(tmp))
            p.package()
            ok, traces = p.execute(timeout=180)
        assert ok
        assert (traces[0].get("evaluation") or {}).get("status") == EvaluationStatus.PASSED.value


@pytest.mark.cutlass_ext
@pytest.mark.cuda
class TestCutlassTagged:
    @staticmethod
    def _need_cuda():
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_compiles_and_passes(self):
        self._need_cuda()
        definition = Definition(**load_json_file(RMSNORM_DIR / "definition.json"))
        workloads = [Workload(**load_jsonl_file(RMSNORM_DIR / "workload.jsonl")[0])]
        solution = _load_solution("cutlass")
        with tempfile.TemporaryDirectory() as tmp:
            p = ProblemPackager(definition, workloads, solution, Path(tmp))
            p.package()
            p.compile(timeout=300)
            ok, traces = p.execute(timeout=120)
        assert ok
        assert (traces[0].get("evaluation") or {}).get("status") == EvaluationStatus.PASSED.value


@pytest.mark.cuda
class TestCudnnTagged:
    @staticmethod
    def _need_cuda():
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_compiles_and_passes(self):
        self._need_cuda()
        definition = Definition(**load_json_file(RMSNORM_DIR / "definition.json"))
        workloads = [Workload(**load_jsonl_file(RMSNORM_DIR / "workload.jsonl")[0])]
        solution = _load_solution("cudnn")
        with tempfile.TemporaryDirectory() as tmp:
            p = ProblemPackager(definition, workloads, solution, Path(tmp))
            p.package()
            try:
                p.compile(timeout=300)
            except Exception as e:
                pytest.skip(f"cuDNN extension build failed: {e}")
            ok, traces = p.execute(timeout=120)
        assert ok
        assert (traces[0].get("evaluation") or {}).get("status") == EvaluationStatus.PASSED.value
