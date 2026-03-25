# Testing Guide

## 前置条件

- Yuanrong Actor framework 已安装
- vLLM-ascend 支持 external_launcher 模式
- PyTorch 支持 NPU (Ascend)
- 模型文件已放置在配置的路径

## 测试配置

如需修改模型路径，编辑测试脚本：

```python
model_a = "/workspace/models/qwen2.5_7B_Instruct"
model_b = "/workspace/models/Qwen2.5-VL-7B-Instruct"
tp_size = 2
```

## 运行测试

### 单独运行测试

每个测试会输出 JSON 结果文件到项目根目录：

```bash
python3 tests/cold_start.py          # -> results_cold_start.json
python3 tests/test_sleep_mode.py      # -> results_sleep_mode.json
```

### 对比结果

```bash
python3 tests/compare_results.py
```

这将显示：
- 初始化延迟（创建 Actor）
- 模型切换延迟
- 多次切换的方差

## 结果解释

### Actor 创建延迟 (Init Latency)

所有测试首次都会创建新 Actor，因此 init_latency_ms 应该接近。

### 模型切换延迟 (Switch Latency)

模型切换流程：
- 复用 Actor 进程
- 删除旧 LLM 实例，调用 `destroy_model_parallel()` 清理 vLLM 内部状态
- 调用 `gc.collect()` 和 `torch.npu.empty_cache()` 释放 NPU 内存
- 创建新的 LLM 实例加载新模型

### 模型切换实现细节

模型切换使用以下流程：

1. 删除旧 LLM 实例
2. 调用 `destroy_model_parallel()` 清理 vLLM 内部状态
3. 调用 `gc.collect()` 和 `torch.npu.empty_cache()` 释放 NPU 内存
4. 创建新的 LLM 实例，加载新模型权重

这种方法完全释放旧模型占用的 NPU 内存，然后创建新的 LLM 实例。

### TP=2 并行行为

使用 TP=2 时，模型权重是分片加载的：
- Rank 0 加载模型权重的第一部分
- Rank 1 加载模型权重的第二部分
- 两个 rank 通过 HCCL 通信协作完成推理

因此切换模型时，**两个 actor 都需要调用 switch_model**，否则无法完成模型切换。
