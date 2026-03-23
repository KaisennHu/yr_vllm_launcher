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
