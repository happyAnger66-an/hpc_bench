# hpc_bench

An independent GPU kernel benchmark framework with interface compatibility to SOL-ExecBench.

## Features

- **Schema Compatible**: Uses the same `definition.json`, `workload.jsonl`, `solution.json` formats
- **Multi-Language Support**: Python, Triton, CUDA C++, CuTe DSL
- **Correctness Checking**: Numerical validation with configurable tolerance
- **Performance Benchmarking**: CUDA events-based timing
- **Modular Architecture**: Clean separation of data models, benchmarking, and execution

## Installation

```bash
cd /home/zhangxa/codes/hpc_bench
pip install -e .
```

Requirements:
- Python >= 3.10
- PyTorch >= 2.0
- CUDA-capable GPU (for GPU kernel testing)

## Quick Start

### 1. Run the Example

We provide a complete `rmsnorm` example to demonstrate the workflow:

```bash
cd /home/zhangxa/codes/hpc_bench

# Run single problem evaluation
hpc-bench examples/rmsnorm --solution examples/rmsnorm/pytorch/solution.json

# Or with PYTHONPATH
PYTHONPATH=src python -m hpc_bench.cli examples/rmsnorm \
  --solution examples/rmsnorm/pytorch/solution.json
```

Expected output:
```
┌─────────────────────────────────────────────────────────────┐
│          Evaluation Results: rmsnorm_h4096                  │
├──────────┬───────────┬──────────────┬───────────────────────┤
│ Workload │ Status    │ Latency (ms) │ Speedup              │
├──────────┼───────────┼──────────────┼───────────────────────┤
│ rmsnorm_ │ PASSED    │ 0.234        │ 1.45x                │
│ rmsnorm_ │ PASSED    │ 0.523        │ 1.38x                │
│ rmsnorm_ │ PASSED    │ 2.156        │ 1.42x                │
└──────────┴───────────┴──────────────┴───────────────────────┘
```

### 2. Understanding the Data Format

#### Problem Definition (`definition.json`)

Defines the operator interface:

```json
{
  "name": "rmsnorm_h4096",
  "op_type": "rmsnorm",
  "axes": {
    "batch_size": {"type": "var"},
    "hidden_size": {"type": "const", "value": 4096}
  },
  "inputs": {
    "input": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"},
    "weight": {"shape": ["hidden_size"], "dtype": "bfloat16"},
    "eps": {"shape": null, "dtype": "float32"}
  },
  "outputs": {
    "output": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"}
  },
  "reference": "import torch\ndef run(input, weight, eps): ..."
}
```

Key fields:
- `axes`: Variable (`var`) and constant (`const`) dimensions
- `inputs`/`outputs`: Tensor specifications with shape and dtype
- `reference`: Ground-truth PyTorch implementation

#### Workload (`workload.jsonl`)

JSONL format with specific test configurations:

```json
{"uuid": "wkl_001", "axes": {"batch_size": 16}, "inputs": {"eps": {"type": "scalar", "value": 1e-6}}, "tolerance": {"max_atol": 1e-3}}
```

Input types:
- `"type": "random"` - Generate random tensor
- `"type": "scalar"` - Use literal value
- `"type": "safetensors"` - Load from file

#### Solution (`solution.json`)

Your kernel implementation:

```json
{
  "name": "rmsnorm_pytorch",
  "definition": "rmsnorm_h4096",
  "author": "your_name",
  "spec": {
    "languages": ["pytorch"],
    "target_hardware": ["LOCAL"],
    "entry_point": "kernel.py::run",
    "destination_passing_style": true
  },
  "sources": [{"path": "kernel.py", "content": "..."}]
}
```

## Usage Guide

### Single Problem Evaluation

#### Method 1: Problem Directory

```bash
hpc-bench <problem_dir> --solution solution.json
```

The directory must contain:
- `definition.json` - Problem specification
- `workload.jsonl` - Test configurations

#### Method 2: Explicit Paths

```bash
hpc-bench \
  --definition def.json \
  --workload wkl.jsonl \
  --solution sol.json \
  -o results.jsonl \
  --json
```

#### Method 3: Using Reference as Solution

To test the reference implementation itself:

```bash
hpc-bench examples/rmsnorm \
  --solution <(echo '{
    "name": "rmsnorm_ref",
    "definition": "rmsnorm_h4096",
    "author": "test",
    "spec": {
      "languages": ["pytorch"],
      "target_hardware": ["LOCAL"],
      "entry_point": "kernel.py::run",
      "destination_passing_style": false
    },
    "sources": [{
      "path": "kernel.py",
      "content": "import torch\ndef run(input, weight, eps):\n    variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)\n    rstd = torch.rsqrt(variance + eps)\n    hidden_states = input * rstd\n    return (hidden_states * weight).to(input.dtype)"
    }]
  }')
```

### Batch Evaluation

For evaluating multiple problems:

```bash
python scripts/run_dataset.py data/benchmark \
  --category L1 L2 \
  --solution-name solution.json \
  -o ./results \
  --limit 10
```

Options:
- `--category`: Filter by category (L1, L2, FlashInfer-Bench, Quant)
- `--solution-name`: Look for specific solution file in each problem dir
- `--limit`: Maximum problems to evaluate
- `--max-workloads`: Limit workloads per problem
- `--rerun`: Force re-evaluation of already processed problems

### Writing Your Own Solution

#### PyTorch Example

```python
# kernel.py
def run(input, weight, eps, output):
    """RMSNorm kernel with DPS (Destination Passing Style).
    
    Args:
        input: [batch_size, hidden_size] tensor
        weight: [hidden_size] tensor
        eps: scalar float
        output: pre-allocated output tensor [batch_size, hidden_size]
    """
    import torch
    variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    output[:] = (input * rstd * weight).to(input.dtype)
```

```json
{
  "name": "rmsnorm_optimized",
  "definition": "rmsnorm_h4096",
  "author": "your_name",
  "spec": {
    "languages": ["pytorch"],
    "target_hardware": ["LOCAL", "B200"],
    "entry_point": "kernel.py::run",
    "destination_passing_style": true,
    "dependencies": ["torch"]
  },
  "sources": [{"path": "kernel.py", "content": "<file_content>"}]
}
```

#### Triton Example

```python
# kernel.py
import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_kernel(
    input_ptr, weight_ptr, output_ptr,
    stride_m, stride_n,
    N, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    input_ptrs = input_ptr + row_idx * stride_m + col_offsets
    weight_ptrs = weight_ptr + col_offsets
    
    x = tl.load(input_ptrs, mask=mask, other=0.0)
    w = tl.load(weight_ptrs, mask=mask, other=0.0)
    
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    out = x * rstd * w
    tl.store(output_ptr + row_idx * stride_m + col_offsets, out, mask=mask)

def run(input, weight, eps, output):
    batch_size, hidden_size = input.shape
    grid = (batch_size,)
    rmsnorm_kernel[grid](
        input, weight, output,
        input.stride(0), input.stride(1),
        hidden_size, eps,
        BLOCK_SIZE=triton.next_power_of_2(hidden_size)
    )
```

```json
{
  "name": "rmsnorm_triton",
  "definition": "rmsnorm_h4096",
  "author": "your_name",
  "spec": {
    "languages": ["triton"],
    "target_hardware": ["LOCAL"],
    "entry_point": "kernel.py::run",
    "destination_passing_style": true,
    "dependencies": ["torch", "triton >= 2.3"]
  },
  "sources": [{"path": "kernel.py", "content": "<file_content>"}]
}
```

### CLI Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--solution` | Path to solution.json (required) | - |
| `--definition` | Path to definition.json | Auto-detect |
| `--workload` | Path to workload.jsonl | Auto-detect |
| `--timeout` | Execution timeout (seconds) | 300 |
| `--compile-timeout` | Compilation timeout (seconds) | 300 |
| `-o, --output` | Write traces to file | - |
| `--json` | Print JSON output to stdout | False |
| `--lock-clocks` | Require GPU clocks locked | False |
| `--keep-staging` | Keep staging directory | False |
| `-v, --verbose` | Verbose output | False |

## Understanding Results

### Trace Format

Each workload produces a trace:

```json
{
  "definition": "rmsnorm_h4096",
  "solution": "rmsnorm_optimized",
  "workload": {"uuid": "wkl_001", "axes": {"batch_size": 16}},
  "evaluation": {
    "status": "PASSED",
    "device": "cuda:0",
    "correctness": {
      "max_absolute_error": 0.00098,
      "max_relative_error": 0.00012,
      "has_nan": false,
      "has_inf": false
    },
    "performance": {
      "latency_ms": 0.523,
      "reference_latency_ms": 1.234,
      "speedup_factor": 2.36
    }
  }
}
```

### Status Codes

| Status | Meaning | Action |
|--------|---------|--------|
| `PASSED` | All checks passed | ✓ Solution is correct and fast |
| `INCORRECT_NUMERICAL` | Numerical error exceeds tolerance | Check algorithm precision |
| `INCORRECT_SHAPE` | Output shape mismatch | Verify output dimensions |
| `INCORRECT_DTYPE` | Output dtype mismatch | Add `.to(dtype)` |
| `RUNTIME_ERROR` | Execution error | Check code for bugs |
| `TIMEOUT` | Execution timeout | Check for infinite loops |

### Tolerance Formula

Numerical correctness uses the `allclose` formula:

```
|output - reference| <= atol + rtol * |reference|
```

With default tolerance:
- `atol` (absolute tolerance): 1e-3
- `rtol` (relative tolerance): 1e-3
- `required_matched_ratio`: 99% of elements must pass

## Project Structure

```
hpc_bench/
├── docs/
│   └── arch.md           # Detailed architecture documentation
├── examples/
│   └── rmsnorm/          # Complete working example
│       ├── definition.json
│       ├── workload.jsonl
│       └── solution_pytorch.json
├── scripts/
│   └── run_dataset.py    # Batch evaluation script
├── src/hpc_bench/
│   ├── cli.py            # Command-line interface
│   ├── core/
│   │   ├── data/         # Data models
│   │   │   ├── definition.py
│   │   │   ├── workload.py
│   │   │   ├── solution.py
│   │   │   └── trace.py
│   │   └── bench/        # Benchmarking logic
│   │       ├── correctness.py
│   │       ├── timing.py
│   │       ├── io.py
│   │       └── config.py
│   └── driver/
│       └── problem_packager.py
└── pyproject.toml
```

## Troubleshooting

### ImportError: No module named 'hpc_bench'

```bash
export PYTHONPATH=/home/zhangxa/codes/hpc_bench/src:$PYTHONPATH
```

### CUDA Out of Memory

Reduce workload batch sizes or use `--max-workloads` to limit concurrent tests.

### Numerical Errors

1. Check dtype conversions (use `to(torch.float32)` for intermediate compute)
2. Verify epsilon values are handled correctly
3. Consider tolerance adjustment for specific workloads

### Compilation Errors (C++/CUDA)

Ensure `nvcc` is in PATH and CUDA toolkit is properly installed:
```bash
nvcc --version
nvidia-smi
```

## Advanced Topics

See [docs/arch.md](docs/arch.md) for:
- Detailed architecture design
- API reference
- Extending the framework
- Compatibility notes with SOL-ExecBench

## License

MIT License
