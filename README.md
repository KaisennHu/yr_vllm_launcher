# Yuanrong + vLLM External Launcher Integration

在 Yuanrong Actor 中使用 vLLM 的 `external_launcher` 模式进行分布式推理。

## 快速开始

```bash
python main.py
```

这将：
1. 创建 2 个 Yuanrong Actor（TP=2）
2. 使用 `gpt2` 模型（无需下载）
3. 执行推理并打印结果

## 代码示例

```python
import yr
from yr.fcc import create_function_group, FunctionGroupOptions
import vllm_actor


def main():
    yr.init(yr.Config())

    # 创建 2 个 Actor，TP=2
    opts = FunctionGroupOptions(
        cpu=2000, memory=4096,
        resources={"NPU/Ascend910B4/count": 1},
        scheduling_affinity_each_bundle_size=2,
    )

    engines = create_function_group(
        func=vllm_actor.VLLMEngine,
        args=("gpt2", 2),
        group_size=2,
        options=opts,
    )

    # 推理
    result = engines[0].generate("Hello, world")
    print(f"Result: {result}")

    yr.finalize()
```

## 使用不同模型

```python
# 使用本地模型路径
engines = create_function_group(
    func=vllm_actor.VLLMEngine,
    args=("/path/to/local/model", 4),  # TP=4
    group_size=4,
    options=opts,
)
```

## HCCL 通信机制

**初始化责任划分：**
- **vLLM-ascend**: 负责调用 `torch.distributed.init_process_group()` 初始化 HCCL
- **Yuanrong**: 通过环境变量配置 HCCL，不直接初始化

**流程：**
```
Yuanrong Actor 启动
    → Yuanrong 设置环境变量
    → vLLM 调用 torch.distributed.init_process_group()
    → PyTorch 内部初始化 HCCL
```

## 环境变量

vLLM external_launcher 需要的环境变量：
- `RANK`: 全局 rank
- `LOCAL_RANK`: 本地 rank
- `WORLD_SIZE`: 总进程数
- `MASTER_ADDR`: 主节点地址
- `MASTER_PORT`: 通信端口（默认 29500）
- `VLLM_ENABLE_V1_MULTIPROCESSING`: 必须设为 "0"
- `HCCL_IF_BASE_PORT`: HCCL 基础端口（默认 50000）

这些变量由 `vllm_actor.py` 自动设置（部分从 Yuanrong 继承）。

## 模型切换

本项目包含模型切换性能测试，通过重新创建 LLM 实例来切换模型。

### 切换原理

模型切换通过以下流程实现：
1. 删除旧的 LLM 实例
2. 调用 `destroy_model_parallel()` 清理 vLLM 内部状态
3. 调用 `gc.collect()` 和 `torch.npu.empty_cache()` 释放 NPU 内存
4. 创建新的 LLM 实例，加载新模型权重

这种方法完全释放旧模型占用的 NPU 内存，然后创建新的 LLM 实例加载新模型。

### 测试脚本

| 脚本 | 说明 |
|-------|------|
| `tests/cold_start.py` | 完全冷启动：创建新 Actor + 初始化 vLLM |
| `tests/test_sleep_mode.py` | 模型切换：复用 Actor，切换模型 |
| `tests/compare_results.py` | 对比测试结果 |

### 运行测试

```bash
# 冷启动测试
python3 tests/cold_start.py

# 模型切换测试
python3 tests/test_sleep_mode.py

# 对比结果
python3 tests/compare_results.py
```

### TP=2 并行行为

使用 TP=2 时，模型权重是分片加载的：
- Rank 0 加载模型权重的第一部分
- Rank 1 加载模型权重的第二部分
- 两个 rank 通过 HCCL 通信协作完成推理

因此切换模型时，**两个 actor 都需要调用 switch_model**，否则无法完成模型切换。
