# Design: Sleep Mode Model Switching Feature

## Overview

验证基于 Yuanrong Actor 使用 vLLM external_launcher，通过 sleep_mode 复用 executor 进程来切换模型，从而节省冷启动 runtime 依赖的时延。

## Problem Statement

当前项目中，每次运行推理都需要重新创建 Yuanrong Actor，每次 Actor 初始化都会：
1. **Import runtime 依赖**：首次 `import torch`, `import torch_npu` 等需要加载大量动态库和初始化
2. 启动新的 executor 进程
3. 初始化 HCCL 通信 (`torch.distributed.init_process_group()`)
4. 初始化 vLLM engine 及其组件

这些冷启动操作引入了显著的时延。当需要在同一个 Actor 上切换不同模型进行推理时，如果能复用 executor 进程和 HCCL 通信状态，可以大幅降低模型切换的时延。

### 关键洞察

**Import runtime 时延是冷启动的重要组成部分：**
- 首次 `import torch`：~2-5 秒（加载 PyTorch C++ 扩展）
- 首次 `import torch_npu`：~3-8 秒（加载 Ascend NPU 相关库）
- 这些 import 只在进程首次运行时发生
- 进程复用时这部分时延可以完全避免

## Goal

量化使用 vLLM sleep_mode 复用 executor 相比传统方式节省的冷启动时延。

## Design

### Test Strategy

创建四个独立的测试脚本，分别测试不同的模型切换方式，并对比时延：

#### 0. Baseline: Fresh Process (`baseline_fresh_process.py`)

**方式：** 每次模型推理都启动全新的 Python 进程
- **完整冷启动**：import runtime + 进程启动 + HCCL 初始化 + vLLM engine 初始化
- 测量最完整的冷启动时延，包括首次 import torch/torch_npu 的开销
- 这是最真实的"每次都冷启动"场景

#### 1. Baseline: Recreate Actor (`baseline_recreate_actor.py`)

**方式：** 在同一进程中，每次切换模型都重新创建 Actor
- 避免了 import runtime 的时延（同一进程复用）
- 但仍需：启动新 executor 进程 + HCCL 初始化 + vLLM engine 初始化
- 测量创建 Actor 的时延

#### 2. Baseline: Recreate LLM (`baseline_recreate_llm.py`)

**方式：** 在同一个 Actor 内重新初始化 LLM
- Actor 进程保持存活，import runtime 已完成
- 每次切换模型重新创建 `LLM` 实例（不复用 executor）
- 避免了进程启动 + HCCL 初始化的开销
- 测量 vLLM engine 初始化时延

#### 3. Feature: Sleep Mode (`test_sleep_mode.py`)

**方式：** 使用 sleep_mode 复用 executor
- Actor + executor 进程保持存活，import runtime 已完成
- 使用 `llm.sleep()` 释放旧模型权重
- 使用 `llm.wake_up()` 唤醒后加载新模型
- **预期时延最低**：仅释放/加载模型权重

### Test Flow

每个测试脚本执行以下流程并记录时延：

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Import Runtime: 首次 import torch, torch_npu 等          │
│    - 仅 baseline_fresh_process.py 测量此项                   │
│    - 记录: import_runtime_latency_ms                          │
├─────────────────────────────────────────────────────────────┤
│ 2. Initialization: 加载模型 A                                │
│    - 创建 Actor/LLM                                          │
│    - 记录: init_latency_ms                                    │
├─────────────────────────────────────────────────────────────┤
│ 3. Warmup Inference: 模型 A 上运行几次推理                      │
│    - 记录: warmup_latency_ms (平均)                            │
├─────────────────────────────────────────────────────────────┤
│ 4. Model Switch: 切换到模型 B                                  │
│    - 释放旧模型 / 创建新实例 / sleep-wake                       │
│    - 记录: switch_latency_ms                                   │
├─────────────────────────────────────────────────────────────┤
│ 5. Post-switch Inference: 模型 B 上运行几次推理                  │
│    - 记录: post_switch_latency_ms (平均)                        │
├─────────────────────────────────────────────────────────────┤
│ 6. Second Switch: 切换回模型 A (可选，验证多次切换)               │
│    - 记录: second_switch_latency_ms                            │
└─────────────────────────────────────────────────────────────┘
```

### Output Format

每个脚本输出结构化的时延统计：

```json
{
  "test_type": "sleep_mode",
  "model_a": "/workspace/models/qwen2.5_7B_Instruct",
  "model_b": "/workspace/models/Qwen2.5-VL-7B-Instruct",
  "tp_size": 2,
  "import_runtime_latency_ms": 5234.56,  // 仅 fresh_process 测试
  "init_latency_ms": 12345.67,
  "warmup_latency_ms_avg": 123.45,
  "switch_latency_ms": 4567.89,
  "post_switch_latency_ms_avg": 98.76,
  "second_switch_latency_ms": 4321.12
}
```

### Architecture

```
yr_vllm_launcher/
├── main.py                    # 现有示例代码
├── vllm_actor.py              # 现有 Actor 定义
├── baseline_fresh_process.py   # 新增：基准测试0（完整冷启动）
├── baseline_recreate_actor.py  # 新增：基准测试1
├── baseline_recreate_llm.py   # 新增：基准测试2
├── test_sleep_mode.py          # 新增：sleep_mode 测试
└── compare_results.py          # 新增：对比脚本
```

### Components

#### vllm_actor.py 修改

添加方法支持 sleep_mode：

```python
@yr.instance
class VLLMEngine:
    # 现有代码...

    def sleep(self, level: int = 1):
        """Sleep the vLLM engine to free model weights."""
        self.llm.sleep(level=level)

    def wake_up(self):
        """Wake up the vLLM engine."""
        self.llm.wake_up()

    def switch_model(self, new_model_path: str):
        """Switch to a new model using sleep mode."""
        self.sleep(level=1)
        # Create new LLM with new model path
        self.llm = LLM(
            model=new_model_path,
            tensor_parallel_size=self._tp,
            distributed_executor_backend="external_launcher",
            # ... other params
        )
```

#### baseline_fresh_process.py

- 启动子进程 A → import runtime + 创建 Actor + LLM(A) + 推理 → 退出
- 启动子进程 B → import runtime + 创建 Actor + LLM(B) + 推理 → 退出
- 记录完整冷启动时延（包括 import runtime）

#### baseline_recreate_actor.py

- 创建 Actor A → 推理 → 终销毁 Actor A
- 创建 Actor B → 推理 → 终销毁 Actor B
- 记录完整时延（不含 import runtime，因为同一进程复用）

#### baseline_recreate_llm.py

- 创建 Actor
- 创建 LLM(A) → 推理 → del LLM(A)
- 创建 LLM(B) → 推理 → del LLM(B)
- 终销毁 Actor

#### test_sleep_mode.py

- 创建 Actor + LLM(A) → 推理
- `llm.sleep()` → `llm.wake_up()` → 创建 LLM(B) → 推理
- `llm.sleep()` → `llm.wake_up()` → 创建 LLM(A) → 推理

#### compare_results.py

- 运行三个测试脚本
- 汇总结果
- 输出对比表格
- 计算节省百分比

### Error Handling

- 模型加载失败：记录错误并跳过该模型
- HCCL 初始化失败：退出并记录错误
- 内存不足：释放资源并尝试清理
- 时延记录异常：标记为无效

### Testing

1. **单元测试：** 测试新增的 `sleep()` 和 `wake_up()` 方法
2. **集成测试：** 运行完整测试流程验证各脚本输出格式正确
3. **性能测试：** 在真实环境中运行对比测试

### Configuration

通过环境变量或命令行参数配置：
- 模型路径 (MODEL_A, MODEL_B)
- Tensor parallel size (TP_SIZE)
- 推理次数 (NUM_INFERENCE_RUNS)
- 日志级别 (LOG_LEVEL)

## Dependencies

- vLLM-ascend (external_launcher 模式)
- Yuanrong Actor framework
- PyTorch with NPU (Ascend) support
- Python 3.8+

## Implementation Details

### Import Runtime 时延测量方法

在 `baseline_fresh_process.py` 中，使用 `subprocess` 启动新进程并测量总时延：

```python
import subprocess
import time
import json

def run_fresh_process(model_path, tp_size):
    """启动全新进程运行推理，测量包含 import runtime 的总时延"""
    start_time = time.time()

    # 启动子进程
    proc = subprocess.Popen([
        sys.executable,
        "_fresh_process_worker.py",  # 独立的工作进程脚本
        "--model", model_path,
        "--tp", str(tp_size)
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = proc.communicate()
    total_latency_ms = (time.time() - start_time) * 1000

    # 从子进程输出中解析推理时延
    result = json.loads(stdout)
    result["import_runtime_latency_ms"] = total_latency_ms - result["init_latency_ms"]

    return result
```

### `_fresh_process_worker.py` (新增辅助脚本)

这是被 `baseline_fresh_process.py` 调用的独立工作进程，其启动时会被测量：

```python
#!/usr/bin/env python3
# 独立工作进程 - 测量包含 import runtime 的冷启动时延

import os
os.environ["VLLM_PLATFORM"] = "ascend"

# 这些 import 会被测量时延！
import torch
import torch_npu
import torch.distributed
import yr
import vllm
from vllm import LLM, SamplingParams

import argparse
import json
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=2)
    args = parser.parse_args()

    # 测量初始化时延（不含 import，因为已经完成）
    start = time.time()

    # ... 创建 Actor 和 LLM 的代码 ...

    init_latency_ms = (time.time() - start) * 1000

    # 返回结果
    result = {
        "init_latency_ms": init_latency_ms,
        # ... 其他测量值 ...
    }
    print(json.dumps(result))

if __name__ == "__main__":
    main()
```

## Success Criteria

1. 四个测试脚本都能正常运行并输出结构化结果
2. `baseline_fresh_process.py` 能准确测量 import_runtime_latency_ms
3. `test_sleep_mode.py` 的 switch_latency 显著低于其他 baseline
4. 对比脚本能准确展示：
   - 完整冷启动时延（含 import）
   - 相比 fresh process 节省的百分比
   - 相比 recreate actor/LLM 节省的百分比
5. 测试结果可复现（多次运行结果稳定）

### 预期时延关系

```
fresh_process_switch > recreate_actor_switch > recreate_llm_switch > sleep_mode_switch
```

其中：
- `fresh_process_switch` 包含：import runtime + 进程启动 + HCCL + vLLM init
- `recreate_actor_switch` 包含：进程启动 + HCCL + vLLM init
- `recreate_llm_switch` 包含：vLLM init
- `sleep_mode_switch` 包含：释放/加载模型权重

## Future Enhancements

- 支持更多模型切换方式（如手动加载权重）
- 支持多节点测试
- 添加内存使用监控
- 支持自动选择最优切换策略
