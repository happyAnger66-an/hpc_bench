# hpc_bench

Independent GPU kernel benchmark framework with **data formats and CLI workflow** aligned to **SOL-ExecBench**-style `definition` / `workload` / `solution` / `trace` contracts. This repository implements evaluation **without** depending on the SOL-ExecBench package.

**What it does:** load a problem (`definition.json` + `workload.jsonl`), load your kernel (`solution.json`), optionally compile CUDA extensions, run **correctness** checks against the embedded reference, then **time** the kernel with CUDA events and emit JSON traces.

## Contents

- [Features](#features)
- [Installation](#installation)
- [CLI](#cli)
- [How evaluation works](#how-evaluation-works)
- [Quick start (RMSNorm)](#quick-start-rmsnorm)
- [Data formats](#data-formats)
- [Solution contracts (DPS)](#solution-contracts-dps)
- [Backend / language notes](#backend--language-notes)
- [Usage](#usage)
- [Results & traces](#results--traces)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)

## Features

- **Schema-oriented**: Pydantic models for Definition, Workload, Solution, Trace (SOL-ExecBench-compatible shapes).
- **Python solutions**: `pytorch`, `triton`, `cute_dsl`, `cutile` — import `kernel.py` (or equivalent) and call `entry_point`.
- **C++/CUDA extensions**: `cuda_cpp`, `cutlass`, `cudnn`, `cublas` (schema) — build via `torch.utils.cpp_extension.load`, bind with `torch` + pybind11.
- **Correctness**: Per-workload `max_atol` / `max_rtol`, `required_matched_ratio`, optional `max_error_cap` (`ToleranceSpec`).
- **Performance**: Warmup + repeated runs; GPU timing via CUDA events; speedup vs reference when reference timing succeeds.
- **Driver**: `ProblemPackager` stages files, compiles when needed, executes in-process, returns trace dicts.

## Installation

```bash
git clone <your-fork-or-mirror>
cd hpc_bench
pip install -e .
```

**Base requirements**

- Python ≥ 3.10  
- PyTorch ≥ 2.0 (CUDA build recommended for GPU examples)  
- For **CUDA extensions**: NVIDIA driver, CUDA toolkit matching PyTorch, `nvcc`; **`ninja`** is often required for PyTorch’s JIT build.

**Optional extras**

```bash
pip install -e ".[triton]"   # Triton examples
```

Other stacks (CuTe DSL, cuTile, CUTLASS headers, cuDNN) use separate wheels or system installs — see [Backend / language notes](#backend--language-notes).

## CLI

Two entry points (same implementation):

| Command | Purpose |
|---------|---------|
| `hpc-bench` | Primary CLI |
| `sol-execbench` | Drop-in alias for compatibility |

Invoke module without install:

```bash
PYTHONPATH=src python -m hpc_bench.cli <args>
```

## How evaluation works

1. **Load problem**: `definition.json` + `workload.jsonl` (from `PROBLEM_DIR` or `--definition` / `--workload`).
2. **Load solution**: `solution.json`; if `sources[].content` is omitted, the CLI reads `sources[].path` relative to the **directory containing `solution.json`**.
3. **Stage**: Write definition, workloads, solution, and sources into a temp directory (`hpc_bench_*`, or a kept path with `--keep-staging`).
4. **Compile** (if `languages` include `cuda_cpp`, `cutlass`, `cudnn`, …): `torch.utils.cpp_extension.load`; for `LOCAL` hardware, append `-gencode=arch=compute_XX,code=sm_XX` from the device capability.
5. **Execute**: Run reference `run(...)` and your `run(...)` per workload; compare outputs; if correct, benchmark and append `performance` to the trace.

## Quick start (RMSNorm)

Problems live under `examples/<problem>/` with shared `definition.json` and `workload.jsonl`. Each **backend** has its own subdirectory and `solution.json`:

```
examples/rmsnorm/
├── definition.json
├── workload.jsonl
├── pytorch/
├── cuda_cpp/
├── triton/
├── cute_dsl/
├── cutile/
├── cutlass/
└── cudnn/
```

**PyTorch**

```bash
hpc-bench examples/rmsnorm --solution examples/rmsnorm/pytorch/solution.json
```

**CUDA C++** (device code in `.cu`)

```bash
hpc-bench examples/rmsnorm --solution examples/rmsnorm/cuda_cpp/solution.json
```

**Triton**

```bash
pip install triton   # or: pip install -e ".[triton]"
hpc-bench examples/rmsnorm --solution examples/rmsnorm/triton/solution.json
```

**CuTe DSL / cuTile / CUTLASS-tagged / cuDNN** (optional toolchains)

```bash
hpc-bench examples/rmsnorm --solution examples/rmsnorm/cute_dsl/solution.json
hpc-bench examples/rmsnorm --solution examples/rmsnorm/cutile/solution.json
hpc-bench examples/rmsnorm --solution examples/rmsnorm/cutlass/solution.json
hpc-bench examples/rmsnorm --solution examples/rmsnorm/cudnn/solution.json
```

Expected: a Rich table with per-workload **Status**, **Latency (ms)**, **Speedup** (non-`PASSED` rows may show `N/A` for timing).

## Data formats

### `definition.json`

Describes axes, tensor I/O dtypes/shapes, and a **`reference`**: Python source (string) defining `run(...)` that returns outputs (or matches DPS — see below). Used as the numerical ground truth.

### `workload.jsonl`

One JSON object per line: `uuid`, `axes` (concrete axis values), `inputs` (e.g. `random`, `scalar`, `safetensors`), and optional `tolerance`.

Example (abbreviated):

```json
{"uuid": "wkl_001", "axes": {"batch_size": 16}, "inputs": {"input": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-6}}, "tolerance": {"max_atol": 1e-3, "max_rtol": 1e-3}}
```

### `solution.json`

- **`spec.entry_point`**: `"relative/path.py::function"` or `"kernel.cu::run"`.
- **`spec.languages`**: e.g. `["pytorch"]`, `["triton"]`, `["cuda_cpp"]`, …
- **`spec.destination_passing_style`**: if `true`, the last tensor arguments are pre-allocated outputs (DPS).
- **`sources`**: list of `{ "path": "..." , "content": "..." }`; **`content` may be omitted** — then the CLI loads `path` next to `solution.json`.
- **C++/CUDA**: optional `binding: "torch"`, `compile_options` (`cflags`, `cuda_cflags`, `ld_flags`).

## Solution contracts (DPS)

- **`destination_passing_style: true`**: implement `run(..., output)` and write into `output` (see `examples/rmsnorm/pytorch/kernel.py`).
- **`destination_passing_style: false`**: implement `run(...) -> tensor_or_tuple` like the inline `reference` string in examples.

The framework allocates outputs for DPS based on `definition.outputs`.

## Backend / language notes

| Backend folder | `languages` | Runtime / build |
|----------------|-------------|-------------------|
| `pytorch/` | `pytorch` | Python import |
| `triton/` | `triton` | Python + Triton JIT |
| `cute_dsl/` | `cute_dsl` | NVIDIA Cutlass Python / CuTe DSL stack |
| `cutile/` | `cutile` | `cuda.tile` (cuTile) |
| `cuda_cpp/` | `cuda_cpp` | `torch.utils.cpp_extension` + `.cu` |
| `cutlass/` | `cutlass` | Same driver as CUDA; optional **`CUTLASS_PATH`** adds `-I$CUTLASS_PATH/include` during compile |
| `cudnn/` | `cudnn` | cuDNN **OpTensor** / **ReduceTensor** (see `kernel.cu`) + small CUDA epilogue; **`CUDA_HOME`** used for `-I/-L` if needed |

**Compile hints**

- Use **`.cu`** for device code so `nvcc` is used.
- **`LOCAL`** in `target_hardware` adds correct `-gencode=arch=compute_XX,code=sm_XX` (virtual `compute_`, real `sm_`).
- After compile, the driver keeps the **`load()` return value** so the loaded module matches PyTorch’s extension name (e.g. `benchmark_kernel_v1.so`).

## Usage

### Problem directory + solution path

```bash
hpc-bench examples/rmsnorm --solution examples/rmsnorm/pytorch/solution.json
```

`PROBLEM_DIR` must contain `definition.json` and `workload.jsonl`. `--solution` may point anywhere (typically a backend subfolder).

### Explicit paths

```bash
hpc-bench --definition path/to/definition.json \
  --workload path/to/workload.jsonl \
  --solution path/to/solution.json \
  -o traces.jsonl
```

### JSON trace output

```bash
hpc-bench examples/rmsnorm --solution examples/rmsnorm/pytorch/solution.json --json
```

### Batch helper

```bash
python scripts/run_dataset.py data/benchmark \
  --category L1 L2 \
  --solution-name solution.json \
  -o ./results \
  --limit 10
```

Options depend on `scripts/run_dataset.py` (e.g. `--max-workloads`, `--rerun`).

### CLI options

| Option | Description | Default |
|--------|-------------|---------|
| `--solution` | Path to `solution.json` (**required**) | — |
| `--definition` | Path to `definition.json` | From `PROBLEM_DIR` |
| `--workload` | Path to `workload.jsonl` | From `PROBLEM_DIR` |
| `--compile-timeout` | C++/CUDA compile timeout (s) | 300 |
| `--timeout` | Evaluation timeout (s) | 300 |
| `-o`, `--output` | Write JSONL traces | — |
| `--json` | Print traces to stdout | off |
| `--lock-clocks` | Pass through to benchmark config | off |
| `--keep-staging` | Keep staging dir | off |
| `-v`, `--verbose` | Print staging path, etc. | off |

## Results & traces

Each workload yields a trace dict: `definition`, `solution`, `workload`, `evaluation` (`status`, `device`, optional `message`, `correctness`, `performance`, …).

**Note:** On failures, `evaluation.performance` may be `null`. The CLI treats missing/null nested dicts safely.

**Tolerance** (element-wise, `allclose`-style):

```
|output - reference| <= max_atol + max_rtol * |reference|
```

A minimum fraction of elements must satisfy this (`required_matched_ratio`, default `0.99` if not set in workload). Workload-level `tolerance` overrides model defaults (`max_atol` / `max_rtol` default `1e-2` in schema; RMSNorm examples use `1e-3`).

**Common statuses:** `PASSED`, `INCORRECT_NUMERICAL`, `INCORRECT_SHAPE`, `INCORRECT_DTYPE`, `RUNTIME_ERROR`, `TIMEOUT`, `INVALID_REFERENCE`, …

## Project layout

```
hpc_bench/
├── docs/
│   └── arch.md              # Architecture & interface spec
├── examples/
│   └── rmsnorm/             # Full multi-backend example
├── scripts/
│   └── run_dataset.py
├── tests/
│   └── e2e/
├── src/hpc_bench/
│   ├── cli.py
│   ├── core/data/           # definition, workload, solution, trace, …
│   ├── core/bench/          # correctness, timing, io, config
│   └── driver/
│       └── problem_packager.py
└── pyproject.toml
```

## Testing

```bash
pip install pytest
cd hpc_bench
PYTHONPATH=src pytest tests/
```

**Markers** (see `pytest.ini`):

- `cuda` — needs NVIDIA GPU + CUDA  
- `triton` — Triton + CUDA  
- `cute_dsl` — Cutlass/CuTe Python + CUDA  
- `cutile` — `cuda.tile` + CUDA  
- `cutlass_ext` — builds CUTLASS-tagged extension  

Examples:

```bash
pytest -m "not cuda"              # CPU-only machines
pytest -m "not triton and not cutile and not cute_dsl"   # minimal deps
```

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| `No module named 'hpc_bench'` | `pip install -e .` or `PYTHONPATH=src` |
| Stale CLI after `git pull` | Reinstall editable or run `python -m hpc_bench.cli` from `src` on `PYTHONPATH` |
| `cuda_runtime.h` / host `c++` on `.cpp` | Rename device code to **`.cu`** |
| `sm_8` / invalid `-arch` | Use current `hpc_bench` (capability → `compute_XX` + `sm_XX`) |
| `ninja` / build failures | `pip install ninja`; ensure `nvcc` matches PyTorch CUDA |
| `CUDNN_STATUS_NOT_SUPPORTED` on BF16 reduce | Prefer FP32 reduction path (see `examples/rmsnorm/cudnn/kernel.cu`) |
| OOM | Smaller `batch_size` in workloads; fewer workloads in `workload.jsonl` |
| Numerical drift | Match reference dtype flow (e.g. fp32 for reductions); relax `tolerance` if justified |

## Documentation

- **[docs/arch.md](docs/arch.md)** — architecture, data model details, extension points, SOL-ExecBench alignment notes.
- **[skills/openclaw-hpc-bench/SKILL.md](skills/openclaw-hpc-bench/SKILL.md)** — Cursor/OpenClaw-style agent skill: how to benchmark custom kernels with hpc_bench.

## License

MIT License
