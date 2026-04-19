# hpc_bench 测试

## 测试结构

```
tests/
├── e2e/                      # 端到端测试
│   ├── __init__.py
│   └── test_rmsnorm_example.py  # RMSNorm 示例完整测试
├── __init__.py
├── README.md                 # 本文档
└── run_tests.sh              # 测试运行脚本
```

## 运行测试

### 方式一：使用脚本

```bash
cd /home/zhangxa/codes/hpc_bench
./tests/run_tests.sh
```

### 方式二：使用 pytest

```bash
cd /home/zhangxa/codes/hpc_bench
export PYTHONPATH=src:$PYTHONPATH

# 运行所有测试
python -m pytest tests/ -v

# 仅运行 e2e 测试
python -m pytest tests/e2e -v

# 跳过需要 CUDA 的测试
python -m pytest tests/e2e -v -m "not cuda"

# 仅运行特定测试
python -m pytest tests/e2e/test_rmsnorm_example.py::TestRMSNormExample::test_definition_loads -v
```

## 测试覆盖

### e2e 测试 (`tests/e2e/test_rmsnorm_example.py`)

| 测试方法 | 说明 |
|---------|------|
| `test_definition_loads` | 验证 definition.json 正确加载 |
| `test_workloads_load` | 验证 workload.jsonl 正确加载 |
| `test_solution_with_file_source_loads` | 验证文件模式的 solution 加载 |
| `test_kernel_file_exists_and_valid` | 验证 kernel.py 存在且语法正确 |
| `test_inline_solution_also_works` | 验证内联模式的 solution 仍可用 |
| `test_full_evaluation_passes` | 验证完整评测流程通过（需 CUDA）|
| `test_reference_implementation_runs` | 验证参考实现可执行 |
| `test_cli_loads_solution_with_file_source` | 验证 CLI 文件加载逻辑 |
| `test_cli_fails_gracefully_when_source_missing` | 验证错误处理 |

## 添加新测试

### 测试 Solution 加载模式

确保测试覆盖两种模式：

1. **内联模式**（content 在 solution.json 中）
2. **文件模式**（content 从独立文件加载）

示例：

```python
def test_both_solution_modes(self):
    """Test both inline and file-based solutions work."""
    # 内联模式
    inline_solution = {
        "sources": [{
            "path": "kernel.py",
            "content": "def run(x, y, out): out[:] = x + y"
        }]
    }
    Solution(**inline_solution)  # 应该成功

    # 文件模式
    file_solution = {
        "sources": [{"path": "kernel.py"}]  # 无 content
    }
    # 需要从文件系统加载 content
    solution_data = load_json_file("solution.json")
    for source in solution_data.get("sources", []):
        if source.get("content") is None:
            source_path = solution_dir / source["path"]
            source["content"] = source_path.read_text()
    Solution(**solution_data)  # 应该成功
```

### 测试错误处理

确保测试验证当文件缺失时的错误提示：

```python
def test_missing_source_file_raises_error(self, tmp_path: Path):
    """Test that missing source file raises clear error."""
    solution = {"sources": [{"path": "nonexistent.py"}]}
    (tmp_path / "solution.json").write_text(json.dumps(solution))

    with pytest.raises(FileNotFoundError) as exc:
        # CLI 加载逻辑
        for source in solution_data.get("sources", []):
            if source.get("content") is None:
                source_path = solution_dir / source["path"]
                if not source_path.exists():
                    raise FileNotFoundError(
                        f"Source file not found: {source_path}. "
                        f"Provide inline content or ensure file exists."
                    )

    assert "nonexistent.py" in str(exc.value)
```

## CI/CD 集成

### GitHub Actions 示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -e . pytest torch
      - run: pytest tests/e2e -v -m "not cuda"
```

## 测试最佳实践

1. **每个修复配一个测试** - 确保修改不引入回归
2. **使用临时目录** - 避免污染工作目录
3. **跳过 CUDA 测试** - 在没有 GPU 的环境中使用 `@pytest.mark.skipif`
4. **清晰的错误信息** - 断言失败时提供详细信息
