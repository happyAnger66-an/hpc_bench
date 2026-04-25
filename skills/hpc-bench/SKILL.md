---
name: hpc-bench
description: >-
  Benchmarks custom GPU kernels and operators with the hpc_bench framework using
  definition.json, workload.jsonl, and solution.json (correctness vs reference +
  CUDA event timing). Use when the user asks to profile or benchmark a kernel,
  wire an operator into hpc_bench, run hpc-bench/sol-execbench CLI, add a
  solution entry point, or debug PASSED/INCORRECT_NUMERICAL traces.
---

# 使用 hpc_bench 对自写算子做性能评测

本 skill 说明如何让用户实现的算子在 **hpc_bench** 下完成：**数值正确性**（相对 `definition.reference`）与 **GPU 耗时**（CUDA events，含 warmup）

## 1. 前置条件

- 已安装 `hpc_bench`（`pip install -e .` 或 `PYTHONPATH=<repo>/src`）。
- GPU 评测需要 **PyTorch CUDA 版**、匹配版本的 **CUDA / nvcc**；编译 C++/CUDA 扩展时常需 **ninja**（`pip install ninja`）。
- 入口命令：`hpc-bench` 或 `sol-execbench`（等价），亦可 `PYTHONPATH=src python -m hpc_bench.cli`。

## 2. 三件数据与职责

| 文件 | 作用 |
|------|------|
| `definition.json` | 算子契约：`axes`、`inputs`/`outputs` 形状与 dtype；**`reference`** 为可信 PyTorch 参考实现（字符串形式的 Python 源码）。 |
| `workload.jsonl` | 每行一个 workload：`uuid`、`axes`（具体维度）、`inputs`（如 `random`/`scalar`/`safetensors`）、`tolerance`（`max_atol`/`max_rtol` 等）。 |
| `solution.json` | 用户算子：`spec.languages`、`spec.entry_point`（`文件::函数`）、`sources`（可省略 `content`，由 CLI 从 `solution.json` 同目录按 `path` 读文件）。 |

**约定**：`solution.spec.definition` 必须与 `definition.name` 一致。

## 3. 实现侧接口（必读）

### 3.1 DPS（Destination Passing Style）

当 `spec.destination_passing_style: true`（推荐与仓库示例一致）：

- `reference` 一般为 `def run(...): ... return tensor` **或** 与评测器约定一致；用户算子则为  
  `def run(..., output):`，**最后一个张量参数为预分配输出**，在函数内 **原地写入** `output`。
- 框架按 `definition.outputs` 分配 `output` 缓冲区。

### 3.2 非 DPS

`destination_passing_style: false` 时：用户与参考均为 `def run(...) -> Tensor | tuple`，返回值与 `outputs` 对齐。

### 3.3 语言与入口后缀

- **Python 类**（`pytorch`、`triton`、`cute_dsl`、`cutile`）：`entry_point` 必须为 **`*.py::符号`**。
- **C++/CUDA 类**（`cuda_cpp`、`cutlass`、`cudnn` 等）：`entry_point` 为 **`*.cu` / `*.cpp` 等::符号**；通常 `binding: "torch"`，由 `torch.utils.cpp_extension.load` 编译。

设备代码应放在 **`.cu`** 中，以便走 **nvcc**；勿把含 `__global__` 的代码放在仅由主机 `c++` 编译的 `.cpp` 里。

## 4. 最小评测步骤（推荐工作流）

1. **编写或拷贝** `definition.json`，确保 `reference` 与 I/O 形状、dtype 一致且可在 CUDA 上运行。
2. **编写** `workload.jsonl`：至少一条 workload，设合理 `tolerance`（bf16 常用 `1e-3` 量级需与实现一致）。
3. **编写** `solution.json` + 源码（与示例 `examples/rmsnorm/<backend>/` 对齐）。
4. **执行**：
   ```bash
   hpc-bench <problem_dir> --solution <path/to/solution.json> -v
   ```
   `problem_dir` 下需有 `definition.json` 与 `workload.jsonl`；`--solution` 可在任意子目录（如 `pytorch/solution.json`）。
5. **看结果**：终端表格中 `PASSED` 表示数值与参考在容差内且计时成功；`INCORRECT_NUMERICAL` / `RUNTIME_ERROR` 需对照 trace 中 `message` / `correctness`。

## 5. 性能指标含义

- **`latency_ms`**：用户算子单次有效计时（实现细节见 `core/bench/timing.py`，含 warmup 与重复）。
- **`reference_latency_ms`** / **`speedup_factor`**：在参考实现可稳定计时时给出；失败或非 `PASSED` 时 `performance` 可能为 `null`，属正常。

## 6. 常用 CLI 参数

- `--compile-timeout` / `--timeout`：编译与评测超时。
- `-o trace.jsonl`：写出 JSONL trace。
- `--json`：向 stdout 打印 trace。
- `--keep-staging`：保留临时目录便于查编译产物（配合 `-v` 看路径）。

## 7. 易错点（排错清单）

| 现象 | 处理方向 |
|------|----------|
| `ValidationError` on `Solution` | `sources` 是否包含 `entry_point` 对应文件；C++/Python 语言是否与后缀一致。 |
| `cuda_runtime.h` + 主机编译 | 改用 `.cu` 或确保 nvcc 参与编译。 |
| `sm_8` / gencode 错误 | 使用当前 hpc_bench：`arch=compute_XX` + `code=sm_XX`。 |
| `PyInit_*` / 扩展名不一致 | 驱动在 `compile()` 后持有 `load()` 返回模块；勿用手工 `importlib` 名与 `benchmark_kernel_v1.so` 不一致。 |
| 数值略超容差 | 对齐参考的 **dtype 与累加顺序**（例如先在 `float32` 归约再 cast）；调整 `tolerance` 需有依据。 |
| cuDNN / CUTLASS | `CUDA_HOME`、`CUTLASS_PATH`；`solution.json` 中 `ld_flags` 含 `-lcudnn` 等。 |

## 8. 仓库内参考

- **多后端示例**：`examples/rmsnorm/`（pytorch / cuda_cpp / triton / cute_dsl / cutile / cutlass / cudnn）。
- **架构与字段细节**：`docs/arch.md`。
- **人类可读总览**：`README.md`。
- **回归测试**：`tests/e2e/`（可按 marker 筛选 `cuda` / `triton` 等）。

## 9. 对模型的操作提示

- 用户只给出 kernel 代码时：**主动补全** `definition` / `workload` / `solution` 三件套，并选择 `destination_passing_style` 与参考一致。
- 修改 shape/dtype 时：**同步** `definition`、`reference`、用户 `run` 签名与 `workload` 的 `axes`。
- 应用评测前：在目标环境执行 `hpc-bench ...` 或 `pytest tests/e2e/...`，**不要**只描述步骤不跑命令（若环境允许）。
