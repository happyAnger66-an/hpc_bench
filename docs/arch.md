# hpc_bench 架构设计文档

## 1. 概述

`hpc_bench` 是一个独立的 GPU Kernel 评测框架，与 `SOL-ExecBench` 保持接口规范兼容。框架采用分层架构设计，核心设计原则包括：

- **Schema 兼容**：完全兼容 SOL-ExecBench 的 `definition.json`、`workload.jsonl`、`solution.json` 格式
- **多语言支持**：支持 PyTorch、Triton、CuTe DSL、CUDA C++ 等多种实现语言
- **可扩展性**：模块化设计，易于添加新的评测指标和语言支持
- **独立性**：不依赖 SOL-ExecBench 代码库，自包含实现

## 2. 架构分层

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI 层                                │
│  (hpc_bench.cli)                                            │
│  - 命令行接口                                                │
│  - 参数解析                                                  │
│  - 结果展示 (rich表格)                                       │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                     Driver 层                              │
│  (hpc_bench.driver)                                         │
│  - ProblemPackager: 问题打包与准备                           │
│  - 编译管理 (C++/CUDA)                                      │
│  - 子进程执行隔离                                            │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                     Core 层                                  │
│  ┌─────────────────┐  ┌──────────────────────────────────┐ │
│  │   Data 模块     │  │           Bench 模块             │ │
│  │  (core/data)    │  │          (core/bench)            │ │
│  │                 │  │                                  │ │
│  │ • Definition    │  │ • correctness: 数值正确性校验     │ │
│  │ • Workload      │  │ • timing: CUDA计时               │ │
│  │ • Solution      │  │ • io: 输入生成/输出分配            │ │
│  │ • Trace         │  │ • config: 评测配置               │ │
│  └─────────────────┘  └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 3. 接口规范

### 3.1 数据模型规范

#### Definition（算子定义）

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

**字段说明**：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `name` | string | ✓ | 唯一标识符 |
| `op_type` | string | | 算子类型分类 |
| `axes` | object | ✓ | 维度定义（const/var/expr） |
| `inputs` | object | ✓ | 输入张量规格 |
| `outputs` | object | ✓ | 输出张量规格 |
| `reference` | string | ✓ | 参考实现（Python代码） |
| `custom_inputs_entrypoint` | string | | 自定义输入生成函数名 |

#### Workload（具体评测配置）

```json
{
  "uuid": "wkl_001",
  "axes": {"batch_size": 16},
  "inputs": {
    "input": {"type": "random"},
    "weight": {"type": "random"},
    "eps": {"type": "scalar", "value": 1e-6}
  },
  "tolerance": {
    "max_atol": 1e-3,
    "max_rtol": 1e-3,
    "required_matched_ratio": 0.99
  }
}
```

**输入类型**：

| 类型 | 说明 | 额外字段 |
|------|------|----------|
| `random` | 随机生成 | 无 |
| `scalar` | 标量值 | `value` |
| `safetensors` | 从文件加载 | `path`, `tensor_key` |
| `custom` | 由reference生成 | 无 |

#### Solution（待评测实现）

```json
{
  "name": "rmsnorm_triton_v1",
  "definition": "rmsnorm_h4096",
  "author": "xxx",
  "spec": {
    "languages": ["triton"],
    "target_hardware": ["LOCAL"],
    "entry_point": "kernel.py::run",
    "destination_passing_style": true,
    "dependencies": ["torch", "triton"]
  },
  "sources": [
    {"path": "kernel.py", "content": "..."}
  ]
}
```

**支持语言**：

| 语言 | 入口格式 | 绑定方式 |
|------|----------|----------|
| `pytorch` | `file.py::func` | Python import |
| `triton` | `file.py::func` | Python import |
| `cute_dsl` | `file.py::func` | Python import |
| `cuda_cpp` | `file.cu::func` | torch extension |
| `cutlass` | `file.cpp::func` | torch extension |

### 3.2 函数接口规范

#### DPS 模式（Destination Passing Style）

**默认模式**，`destination_passing_style: true`

```python
# 参数顺序: inputs按definition顺序 + outputs按definition顺序
def run(input1, input2, ..., output1, output2, ...):
    # 就地写入输出，不返回
    output1[:] = compute_result(...)
```

#### 返回值模式

`destination_passing_style: false`

```python
# 参数顺序: 仅inputs
def run(input1, input2, ...):
    # 返回输出张量
    return output1, output2, ...
```

### 3.3 Trace 输出格式

```json
{
  "definition": "rmsnorm_h4096",
  "solution": "rmsnorm_triton_v1",
  "workload": {"uuid": "wkl_001", "axes": {...}},
  "evaluation": {
    "status": "PASSED",
    "device": "cuda:0",
    "correctness": {
      "max_absolute_error": 0.001,
      "max_relative_error": 0.0001,
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

**EvaluationStatus 枚举**：

- `PASSED` - 通过所有检查
- `INCORRECT_NUMERICAL` - 数值误差超出容差
- `INCORRECT_SHAPE` - 输出形状不匹配
- `INCORRECT_DTYPE` - 输出数据类型不匹配
- `RUNTIME_ERROR` - 运行时错误
- `TIMEOUT` - 超时
- `BUILD_ERROR` - 编译错误

## 4. 评测流程

```
┌─────────┐    ┌──────────┐    ┌────────────┐    ┌─────────────┐
│  Load   │ -> │ Generate │ -> │  Reference │ -> │   User      │
│  JSONs  │    │  Inputs  │    │    Run     │    │ Solution    │
└─────────┘    └──────────┘    └────────────┘    └─────────────┘
                                                    │
                       ┌────────────────────────────┼─────────────┐
                       │                            │             │
                       ▼                            ▼             ▼
                 ┌──────────┐               ┌──────────┐   ┌──────────┐
                 │ Correct  │               │ Incorrect│   │  Error   │
                 │   ✓      │               │   ✗      │   │   ⚠      │
                 └──────────┘               └──────────┘   └──────────┘
                       │                            │             │
                       ▼                            ▼             ▼
                 ┌──────────┐               ┌──────────┐   ┌──────────┐
                 │  Timing  │               │  Report  │   │  Report  │
                 │Benchmark │               │  Error   │   │  Error   │
                 └──────────┘               └──────────┘   └──────────┘
```

### 4.1 正确性检查流程

1. **输入生成**：根据 workload 的 input spec 生成/加载输入数据
2. **参考执行**：运行 reference 实现，获取 ground truth
3. **待测执行**：运行用户 solution
4. **形状检查**：验证输出张量形状
5. **类型检查**：验证输出张量 dtype
6. **数值比较**：使用 `allclose` 风格公式：
   ```
   |output - reference| <= atol + rtol * |reference|
   ```
7. **容差判定**：匹配比例 >= `required_matched_ratio`

### 4.2 计时流程

1. **预热**：执行 warmup 轮（默认10轮），触发JIT编译
2. **冷缓存**：每轮前清除L2缓存，避免缓存干扰
3. **多轮测量**：执行 rep 轮（默认50轮），记录每轮时间
4. **统计汇总**：返回 median/mean/all 统计结果

## 5. 核心模块详解

### 5.1 core/data/definition.py

**Definition** 类是算子的形式化定义，提供以下核心方法：

```python
# 获取解析后的轴值（含表达式计算）
def get_resolved_axes_values(var_axes_values: dict[str, int]) -> dict[str, int]

# 获取输入张量形状
def get_input_shapes(var_axes_values: dict) -> dict[str, Optional[tuple]]

# 获取输出张量形状
def get_output_shapes(var_axes_values: dict) -> dict[str, Optional[tuple]]

# 缓存的 torch dtype 列表
@property
def torch_input_dtypes -> list[torch.dtype]
@property
def torch_output_dtypes -> list[torch.dtype]
```

### 5.2 core/bench/correctness.py

**数值正确性校验**：

```python
# 设置随机种子（可复现性）
def set_seed(seed: int)

# 检查张量异常值（NaN/Inf/全零）
def check_tensor_sanity(sol, ref, allow_negative_inf=False) -> Optional[Correctness]

# 计算误差统计
# 返回: (correctness_metrics, exceeds_tolerance)
def compute_error_stats(output, reference, tolerance) -> Tuple[Correctness, bool]
```

**容差公式**：

```python
# 元素级容差边界
tol_bound = atol + rtol * |reference|
exceeds = |output - reference| > tol_bound

# 整体判定
matched_ratio = (exceeds == False).sum() / total_elements
pass = matched_ratio >= required_matched_ratio
```

### 5.3 core/bench/timing.py

**计时方法**：

```python
def time_runnable(
    fn,              # 待测函数
    inputs,          # 输入列表
    outputs,         # 输出列表（DPS模式）
    device,          # 设备
    warmup=10,       # 预热轮数
    rep=50,          # 测量轮数
    return_mode="median",  # 统计方式
) -> float | list[float]
```

**实现细节**：

- 使用 `ShiftingMemoryPoolAllocator` 保证每轮迭代有独特 data_ptr
- 通过 `torch.cuda.Event` 记录 GPU 时间
- 预热轮次清除L2缓存，模拟冷缓存场景

### 5.4 core/bench/io.py

**输入生成**：

```python
# 根据definition和workload生成输入
def gen_inputs(
    definition: Definition,
    workload: Workload,
    device: str,
    safe_tensors: dict,      # 预加载的safetensors
    custom_inputs_fn,        # 自定义输入生成函数
) -> list[Any]
```

**启发式输入生成**：

| 张量名称模式 | 生成策略 |
|-------------|---------|
| `*_weight`, `weight` | `torch.randn / sqrt(fan_in)` |
| `*_norm_weight` | `torch.ones` |
| `*_norm_bias` | `torch.zeros` |
| `cos`, `sin`, `rope_*` | `[-π, π]` 范围内随机角度 |
| `probs` (sampling op) | softmax归一化 |

**输出分配**：

```python
# 预分配输出张量（DPS模式）
def allocate_outputs(
    definition: Definition,
    resolved_axes: dict[str, int],
    device: str,
) -> list[torch.Tensor]
```

**ShiftingMemoryPoolAllocator**：

- 预分配略大于输入的内存池
- 每轮迭代 data_ptr 偏移 256 字节
- 保持 VRAM 占用接近 1× 输入大小
- 避免 cudaMalloc 干扰计时

### 5.5 driver/problem_packager.py

**ProblemPackager** 负责评测完整流程：

```python
class ProblemPackager:
    def __init__(self, definition, workloads, solution, output_dir)
    
    # 创建staging目录，写入所有必要文件
    def package() -> Path
    
    # 编译C++/CUDA代码（如需要）
    def compile(timeout: int)
    
    # 执行评测，返回traces
    def execute(timeout: int) -> Tuple[bool, list[dict]]
```

## 6. CLI 使用规范

### 6.1 单题评测

```bash
# 方式1: 通过problem目录
hpc-bench <problem_dir> --solution solution.json

# 方式2: 显式指定各文件
hpc-bench \
  --definition def.json \
  --workload workload.jsonl \
  --solution sol.json \
  -o results.jsonl \
  --json
```

### 6.2 批量评测

```bash
python scripts/run_dataset.py \
  data/benchmark \
  --category L1 L2 \
  --solution-name solution.json \
  -o ./out
```

### 6.3 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--compile-timeout` | 编译超时(秒) | 300 |
| `--timeout` | 执行超时(秒) | 300 |
| `--lock-clocks` | 要求锁GPU时钟 | False |
| `--keep-staging` | 保留staging目录 | False |
| `-v, --verbose` | 详细输出 | False |

## 7. 扩展指南

### 7.1 添加新的输入类型

1. 在 `core/data/workload.py` 添加新的 InputSpec 子类
2. 在 `core/bench/io.py` 的 `gen_inputs()` 添加处理逻辑

### 7.2 添加新的评测指标

1. 在 `core/data/trace.py` 的 `Evaluation` 添加字段
2. 在 `driver/problem_packager.py` 的 execute 中添加计算逻辑

### 7.3 添加新的语言支持

1. 在 `core/data/solution.py` 的 `SupportedLanguages` 添加枚举
2. 在 `driver/problem_packager.py` 添加编译/加载逻辑

### 7.4 自定义计时方法

```python
# 继承或替换 timing.py 中的 time_runnable
def custom_time_runnable(fn, inputs, outputs, device):
    # 实现自定义计时逻辑
    return latency_ms
```

## 8. 与 SOL-ExecBench 的兼容性

### 8.1 Schema 兼容

- `definition.json`：完全兼容
- `workload.jsonl`：完全兼容（注意：SOL-ExecBench 可能支持额外字段，本框架会忽略未知字段）
- `solution.json`：完全兼容核心字段

### 8.2 差异说明

| 特性 | SOL-ExecBench | hpc_bench |
|------|---------------|-----------|
| CUPTI计时 | 支持 | 未实现（使用CUDA events） |
| 反作弊检测 | 完整 | 未实现 |
| 锁时钟 | 支持 | 未实现 |
| 子进程隔离 | 完整 | 简化版（Python内执行） |
| C++编译 | 完整 | 待完善 |

### 8.3 迁移指南

从 SOL-ExecBench 迁移到 hpc_bench：

1. 数据文件无需修改
2. CLI 命令基本一致
3. 注意 C++ 解决方案可能需要额外配置

## 9. 参考示例

### 9.1 PyTorch Solution

```json
{
  "name": "rmsnorm_pytorch",
  "definition": "rmsnorm_h4096",
  "author": "test",
  "spec": {
    "languages": ["pytorch"],
    "target_hardware": ["LOCAL"],
    "entry_point": "kernel.py::run",
    "destination_passing_style": true
  },
  "sources": [{
    "path": "kernel.py",
    "content": "import torch\ndef run(input, weight, eps, output):\n    ..."
  }]
}
```

### 9.2 Triton Solution

```json
{
  "name": "rmsnorm_triton",
  "definition": "rmsnorm_h4096",
  "author": "test",
  "spec": {
    "languages": ["triton"],
    "target_hardware": ["LOCAL"],
    "entry_point": "kernel.py::run",
    "destination_passing_style": true,
    "dependencies": ["torch", "triton"]
  },
  "sources": [{
    "path": "kernel.py",
    "content": "import triton\n@triton.jit\ndef _kernel(...):\n    ...\ndef run(input, weight, eps, output):\n    _kernel[grid](...)"
  }]
}
```

## 10. 附录

### 10.1 数据类型映射

| Schema dtype | torch.dtype | 说明 |
|-------------|-------------|------|
| `float64` | `torch.float64` | FP64 |
| `float32` | `torch.float32` | FP32 |
| `float16` | `torch.float16` | FP16 |
| `bfloat16` | `torch.bfloat16` | BF16 |
| `float8_e4m3fn` | `torch.float8_e4m3fn` | E4M3 FP8 |
| `float8_e5m2` | `torch.float8_e5m2` | E5M2 FP8 |
| `float4_e2m1fn_x2` | `torch.float4_e2m1fn_x2` | E2M1 FP4 (pack2) |
| `int64` | `torch.int64` | INT64 |
| `int32` | `torch.int32` | INT32 |
| `int16` | `torch.int16` | INT16 |
| `int8` | `torch.int8` | INT8 |
| `bool` | `torch.bool` | BOOL |

### 10.2 错误代码速查

| 状态 | 含义 | 排查建议 |
|------|------|---------|
| `INCORRECT_NUMERICAL` | 数值误差大 | 检查算法、精度损失 |
| `INCORRECT_SHAPE` | 输出形状错 | 检查输出维度计算 |
| `INCORRECT_DTYPE` | 数据类型错 | 检查 `.to(dtype)` |
| `RUNTIME_ERROR` | 运行时错误 | 查看详细错误信息 |
| `TIMEOUT` | 超时 | 检查死循环或无限递归 |
| `BUILD_ERROR` | 编译错误 | 检查C++代码语法 |
