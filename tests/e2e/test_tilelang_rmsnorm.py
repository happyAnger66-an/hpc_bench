"""E2E tests for TileLang RMSNorm example (optional tilelang + CUDA)."""

from __future__ import annotations

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


class TestTileLangSolutionJson:
    def test_exists_and_language(self):
        path = RMSNORM_DIR / "tilelang" / "solution.json"
        assert path.exists()
        sol = _load_solution("tilelang")
        assert sol.name == "rmsnorm_tilelang_v1"
        assert sol.spec.languages[0].value == "tilelang"


@pytest.mark.tilelang
@pytest.mark.cuda
class TestTileLangRMSNormGPU:
    @staticmethod
    def _skip_if_no_tilelang():
        pytest.importorskip("tilelang")
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_eval_all_workloads(self):
        self._skip_if_no_tilelang()
        definition = Definition(**load_json_file(RMSNORM_DIR / "definition.json"))
        workloads = [Workload(**row) for row in load_jsonl_file(RMSNORM_DIR / "workload.jsonl")]
        solution = _load_solution("tilelang")
        with tempfile.TemporaryDirectory() as tmp:
            p = ProblemPackager(definition, workloads, solution, Path(tmp))
            p.package()
            ok, traces = p.execute(timeout=300)
        assert ok
        for t in traces:
            assert (t.get("evaluation") or {}).get("status") == EvaluationStatus.PASSED.value
