"""Driver for compiling and executing kernel solutions."""

import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from hpc_bench.core.data import (
    Definition,
    Solution,
    SupportedLanguages,
    Workload,
)


def _is_cpp_language(languages: List[SupportedLanguages]) -> bool:
    """Check if any of the languages is a C++ language."""
    cpp_languages = {
        SupportedLanguages.CUTLASS,
        SupportedLanguages.CUDNN,
        SupportedLanguages.CUBLAS,
        SupportedLanguages.CUDA_CPP,
    }
    return any(lang in cpp_languages for lang in languages)


class ProblemPackager:
    """Packages a problem for compilation and execution."""

    def __init__(
        self,
        definition: Definition,
        workloads: List[Workload],
        solution: Solution,
        output_dir: Path,
    ):
        self.definition = definition
        self.workloads = workloads
        self.solution = solution
        self.output_dir = Path(output_dir)
        self.staging_dir: Optional[Path] = None
        # Set in compile(); avoids importlib/PyInit mismatch when torch renames the .so (e.g. _v1).
        self._compiled_extension_module: Optional[Any] = None

    @staticmethod
    def _compute_capability_to_suffix(compute_cap: str) -> Optional[int]:
        """Parse nvidia-smi compute_cap (e.g. '8.9', '7.5') to XY for compute_XY/sm_XY."""
        s = compute_cap.strip()
        if not s:
            return None
        parts = s.split(".")
        try:
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            return None
        # e.g. 8.9 -> 89, 8.0 -> 80, 7.5 -> 75
        return major * 10 + minor

    def _get_local_capability_suffix(self) -> Optional[int]:
        """GPU capability as single integer (e.g. 89); used in nvcc -gencode."""
        if torch.cuda.is_available():
            try:
                major, minor = torch.cuda.get_device_capability(0)
                return major * 10 + minor
            except Exception:
                pass
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            line = result.stdout.strip().splitlines()[0].strip()
            return self._compute_capability_to_suffix(line)
        except Exception:
            return None

    def package(self) -> Path:
        """Create the staging directory with all necessary files."""
        self._compiled_extension_module = None
        self.staging_dir = self.output_dir
        self.staging_dir.mkdir(parents=True, exist_ok=True)

        # Write definition
        from hpc_bench.core.data.json_utils import save_json_file
        save_json_file(
            self.staging_dir / "definition.json",
            self.definition.model_dump(mode="json"),
        )

        # Write workloads
        from hpc_bench.core.data.json_utils import save_jsonl_file
        save_jsonl_file(
            self.staging_dir / "workload.jsonl",
            [w.model_dump(mode="json") for w in self.workloads],
        )

        # Write solution
        save_json_file(
            self.staging_dir / "solution.json",
            self.solution.model_dump(mode="json"),
        )

        # Write source files
        for source in self.solution.sources:
            source_path = self.staging_dir / source.path
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(source.content)

        return self.staging_dir

    def compile(self, timeout: int = 300) -> None:
        """Compile C++/CUDA sources using torch.utils.cpp_extension."""
        if not _is_cpp_language(self.solution.spec.languages):
            return

        if self.staging_dir is None:
            raise RuntimeError("Must call package() before compile()")

        from torch.utils.cpp_extension import load

        # Get compile options from solution spec
        compile_options = self.solution.spec.compile_options
        extra_cflags = compile_options.cflags if compile_options else []
        extra_cuda_cflags = compile_options.cuda_cflags if compile_options else ["-O3", "--use_fast_math"]
        extra_ldflags = compile_options.ld_flags if compile_options else ["-lcuda"]

        # Detect local SM if target includes LOCAL
        target_hw = {h.upper() for h in self.solution.spec.target_hardware}
        if "LOCAL" in target_hw:
            cap = self._get_local_capability_suffix()
            if cap is not None:
                # arch= must be virtual (compute_XX); code= is sm_XX (nvcc 12+ enforces this)
                extra_cuda_cflags.append(
                    f"-gencode=arch=compute_{cap},code=sm_{cap}"
                )

        langs = self.solution.spec.languages
        if SupportedLanguages.CUTLASS in langs:
            cutlass_path = os.environ.get("CUTLASS_PATH")
            if cutlass_path:
                inc = os.path.join(cutlass_path, "include")
                if os.path.isdir(inc):
                    extra_cuda_cflags.append(f"-I{inc}")
        if SupportedLanguages.CUDNN in langs:
            cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
            inc = os.path.join(cuda_home, "include")
            lib64 = os.path.join(cuda_home, "lib64")
            if os.path.isdir(inc):
                extra_cuda_cflags.append(f"-I{inc}")
            if os.path.isdir(lib64):
                extra_ldflags.append(f"-L{lib64}")
            ld_joined = " ".join(extra_ldflags)
            if "-lcudnn" not in ld_joined:
                extra_ldflags.append("-lcudnn")

        # Get source files
        source_files = [str(self.staging_dir / s.path) for s in self.solution.sources]

        # Compile extension; keep module handle so execute() matches PyInit_* after torch suffixes name.
        try:
            self._compiled_extension_module = load(
                name="benchmark_kernel",
                sources=source_files,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_ldflags=extra_ldflags,
                build_directory=str(self.staging_dir),
                verbose=True,
            )
        except Exception as e:
            raise RuntimeError(f"C++/CUDA compilation failed: {e}") from e

    def execute(
        self,
        timeout: int = 300,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Execute the solution and return traces."""
        if self.staging_dir is None:
            raise RuntimeError("Must call package() before execute()")

        # Run evaluation directly (in-process for simplicity)
        return self._execute_in_process(timeout)

    def _execute_in_process(
        self,
        timeout: int = 300,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Execute evaluation in the current process."""
        import gc

        from hpc_bench.core.bench import (
            allocate_outputs,
            compute_error_stats,
            gen_inputs,
            set_seed,
            time_runnable,
        )
        from hpc_bench.core.bench.io import normalize_outputs
        from hpc_bench.core.data.dtypes import dtype_str_to_torch_dtype
        from hpc_bench.core.data.trace import (
            Correctness,
            Evaluation,
            EvaluationStatus,
            Performance,
            Trace,
        )

        # Import reference
        ref_file = self.staging_dir / "_reference.py"
        ref_file.write_text(self.definition.reference)
        spec = importlib.util.spec_from_file_location("_reference", ref_file)
        ref_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref_module)
        ref_fn = ref_module.run

        # Import user solution
        entry_point = self.solution.spec.entry_point
        entry_file, entry_func = entry_point.split("::")

        if _is_cpp_language(self.solution.spec.languages):
            user_module = self._compiled_extension_module
            if user_module is None:
                so_files = list(self.staging_dir.glob("*.so"))
                if not so_files:
                    raise RuntimeError(
                        "No compiled .so file found. "
                        "Ensure compile() was called before execute()."
                    )
                so_path = max(so_files, key=lambda p: p.stat().st_mtime)
                # Stem must match TORCH_EXTENSION_NAME / PyInit_* inside the .so.
                spec = importlib.util.spec_from_file_location(so_path.stem, so_path)
                user_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(user_module)
            user_fn = getattr(user_module, entry_func)
        else:
            # Import Python module
            sys.path.insert(0, str(self.staging_dir))
            module_name = entry_file.replace(".py", "").replace("/", ".")
            user_module = importlib.import_module(module_name)
            user_fn = getattr(user_module, entry_func)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        set_seed(42)

        traces = []

        for workload in self.workloads:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            resolved_axes = self.definition.get_resolved_axes_values(workload.axes)

            # Generate inputs
            try:
                inputs = gen_inputs(self.definition, workload, device)
            except Exception as e:
                trace = Trace(
                    definition=self.definition.name,
                    solution=self.solution.name,
                    workload=workload.model_dump(),
                    evaluation=Evaluation(
                        status=EvaluationStatus.RUNTIME_ERROR,
                        device=device,
                        message=f"Failed to generate inputs: {e}",
                    ),
                )
                traces.append(trace.model_dump(mode="json"))
                continue

            # Run reference
            try:
                ref_outputs_raw = ref_fn(*inputs)
                if not isinstance(ref_outputs_raw, (list, tuple)):
                    ref_outputs = [ref_outputs_raw]
                else:
                    ref_outputs = list(ref_outputs_raw)
            except Exception as e:
                trace = Trace(
                    definition=self.definition.name,
                    solution=self.solution.name,
                    workload=workload.model_dump(),
                    evaluation=Evaluation(
                        status=EvaluationStatus.INVALID_REFERENCE,
                        device=device,
                        message=f"Reference failed: {e}",
                    ),
                )
                traces.append(trace.model_dump(mode="json"))
                continue

            # Run user solution
            try:
                if self.solution.spec.destination_passing_style:
                    outputs = allocate_outputs(self.definition, resolved_axes, device)
                    user_fn(*inputs, *outputs)
                    user_outputs = outputs
                else:
                    user_out = user_fn(*inputs)
                    user_out_dict = normalize_outputs(
                        user_out,
                        device=torch.device(device),
                        output_names=list(self.definition.outputs.keys()),
                        output_dtypes={
                            k: dtype_str_to_torch_dtype(v.dtype)
                            for k, v in self.definition.outputs.items()
                        },
                    )
                    user_outputs = [
                        user_out_dict[k] for k in self.definition.outputs.keys()
                    ]
            except Exception as e:
                trace = Trace(
                    definition=self.definition.name,
                    solution=self.solution.name,
                    workload=workload.model_dump(),
                    evaluation=Evaluation(
                        status=EvaluationStatus.RUNTIME_ERROR,
                        device=device,
                        message=f"User function failed: {e}",
                    ),
                )
                traces.append(trace.model_dump(mode="json"))
                continue

            # Check correctness
            all_correct = True
            max_correctness = Correctness()

            for ref_out, user_out in zip(ref_outputs, user_outputs):
                if ref_out.shape != user_out.shape:
                    trace = Trace(
                        definition=self.definition.name,
                        solution=self.solution.name,
                        workload=workload.model_dump(),
                        evaluation=Evaluation(
                            status=EvaluationStatus.INCORRECT_SHAPE,
                            device=device,
                        ),
                    )
                    traces.append(trace.model_dump(mode="json"))
                    all_correct = False
                    break

                if ref_out.dtype != user_out.dtype:
                    trace = Trace(
                        definition=self.definition.name,
                        solution=self.solution.name,
                        workload=workload.model_dump(),
                        evaluation=Evaluation(
                            status=EvaluationStatus.INCORRECT_DTYPE,
                            device=device,
                        ),
                    )
                    traces.append(trace.model_dump(mode="json"))
                    all_correct = False
                    break

                correctness, exceeds = compute_error_stats(
                    user_out, ref_out, workload.tolerance
                )
                if correctness.max_absolute_error > max_correctness.max_absolute_error:
                    max_correctness = correctness
                if correctness.has_nan:
                    max_correctness = correctness
                elif correctness.has_inf and not max_correctness.has_nan:
                    max_correctness = correctness

                if exceeds:
                    trace = Trace(
                        definition=self.definition.name,
                        solution=self.solution.name,
                        workload=workload.model_dump(),
                        evaluation=Evaluation(
                            status=EvaluationStatus.INCORRECT_NUMERICAL,
                            device=device,
                            correctness=correctness,
                        ),
                    )
                    traces.append(trace.model_dump(mode="json"))
                    all_correct = False
                    break

            if not all_correct:
                continue

            # Benchmark timing
            try:
                if self.solution.spec.destination_passing_style:
                    timing_outputs = allocate_outputs(
                        self.definition, resolved_axes, device
                    )
                else:
                    timing_outputs = []

                sol_latency_ms = time_runnable(
                    user_fn,
                    inputs,
                    timing_outputs,
                    device,
                    warmup=10,
                    rep=50,
                )
            except Exception as e:
                trace = Trace(
                    definition=self.definition.name,
                    solution=self.solution.name,
                    workload=workload.model_dump(),
                    evaluation=Evaluation(
                        status=EvaluationStatus.RUNTIME_ERROR,
                        device=device,
                        message=f"Timing failed: {e}",
                    ),
                )
                traces.append(trace.model_dump(mode="json"))
                continue

            # Benchmark reference (optional)
            ref_latency_ms = 0.0
            try:
                ref_latency_ms = time_runnable(
                    ref_fn,
                    inputs,
                    [],
                    device,
                    warmup=10,
                    rep=50,
                )
            except Exception:
                pass

            speedup = ref_latency_ms / sol_latency_ms if sol_latency_ms > 0 else 0.0

            trace = Trace(
                definition=self.definition.name,
                solution=self.solution.name,
                workload=workload.model_dump(),
                evaluation=Evaluation(
                    status=EvaluationStatus.PASSED,
                    device=device,
                    correctness=max_correctness,
                    performance=Performance(
                        latency_ms=sol_latency_ms,
                        reference_latency_ms=ref_latency_ms,
                        speedup_factor=speedup,
                    ),
                ),
            )
            traces.append(trace.model_dump(mode="json"))

        return True, traces
