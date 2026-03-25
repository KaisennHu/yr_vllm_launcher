# Sleep Mode Model Switching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现基于 Yuanrong Actor 使用 vLLM sleep_mode 复用 executor 进行模型切换的测试框架，量化相比传统方式节省的冷启动时延。

**Architecture:** 创建4个独立的测试脚本分别测试不同的模型切换方式，通过结构化JSON输出时延数据，使用对比脚本汇总结果。

**Tech Stack:** vLLM-ascend (external_launcher), Yuanrong Actor, PyTorch NPU, Python 3.8+

---

## File Structure

```
yr_vllm_launcher/
├── vllm_actor.py              # Modify: 添加 sleep/wake_up/switch_model 方法
├── baseline_fresh_process.py   # Create: 测试完整冷启动（含 import）
├── _fresh_process_worker.py     # Create: fresh process 的辅助脚本
├── baseline_recreate_actor.py  # Create: 测试重建 Actor
├── baseline_recreate_llm.py   # Create: 测试重建 LLM
├── test_sleep_mode.py          # Create: 测试 sleep_mode
└── compare_results.py          # Create: 对比脚本
```

---

## Task 1: Modify vllm_actor.py

**Files:**
- Modify: `vllm_actor.py`

- [ ] **Step 1: Read current vllm_actor.py to understand structure**

Run: `cat vllm_actor.py`
Expected: See current VLLMEngine class implementation

- [ ] **Step 2: Add sleep() method to VLLMEngine class**

After `__init__` method, add:

```python
    def sleep(self, level: int = 1):
        """Sleep vLLM engine to free model weights.

        Args:
            level: Sleep level (1: free model weights, 2: more aggressive)
        """
        _logger = yr.log.get_logger()
        _logger.info(f"Calling llm.sleep(level={level})")
        self.llm.sleep(level=level)
        _logger.info("llm.sleep() completed")
```

- [ ] **Step 3: Add wake_up() method to VLLMEngine class**

After sleep() method, add:

```python
    def wake_up(self):
        """Wake up vLLM engine after sleep."""
        _logger = yr.log.get_logger()
        _logger.info("Calling llm.wake_up()")
        self.llm.wake_up()
        _logger.info("llm.wake_up() completed")
```

- [ ] **Step 4: Add switch_model() method to VLLMEngine class**

After wake_up() method, add:

```python
    def switch_model(self, new_model_path: str):
        """Switch to a new model using sleep mode.

        Args:
            new_model_path: Path to new model
        """
        _logger = yr.log.get_logger()
        _logger.info(f"Switching to new model: {new_model_path}")

        # Sleep to free current model weights
        self.sleep(level=1)

        # Create new LLM with new model path
        _logger.info(f"Initializing vLLM LLM with new model: {new_model_path}")
        self.llm = LLM(
            model=new_model_path,
            tensor_parallel_size=self._tp,
            trust_remote_code=True,
            distributed_executor_backend="external_launcher",
        )
        _logger.info("vLLM LLM switched to new model successfully")
```

- [ ] **Step 5: Verify vllm_actor.py syntax**

Run: `python -m py_compile vllm_actor.py`
Expected: No syntax errors

- [ ] **Step 6: Commit**

```bash
git add vllm_actor.py
git commit -m "feat: add sleep/wake_up/switch_model methods to VLLMEngine"
```

---

## Task 2: Create _fresh_process_worker.py

**Files:**
- Create: `_fresh_process_worker.py`

- [ ] **Step 1: Write _fresh_process_worker.py**

```python
#!/usr/bin/env python3
# coding=UTF-8
# Worker process for baseline_fresh_process.py
# Measures cold startup latency including import runtime

import os
os.environ["VLLM_PLATFORM"] = "ascend"

import torch
import torch_npu
import torch.distributed
import time
import json
import argparse

import yr
import vllm_actor
from vllm import SamplingParams


def run_single_inference(model_path: str, tp: int, rank: int, world_size: int):
    """Run a single inference session and return timing results."""
    _logger = yr.log.get_logger()

    init_start = time.time()

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if "HCCL_IF_BASE_PORT" not in os.environ:
        os.environ["HCCL_IF_BASE_PORT"] = "50000"

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            init_method="env://",
            backend="cpu:gloo,npu:hccl",
            rank=rank,
            world_size=world_size,
        )

    opts = yr.InvokeOptions()
    opts.cpu = 2000
    opts.memory = 4096
    opts.custom_resources = {"NPU/.+/count": 1}

    actor = vllm_actor.VLLMEngine.options(opts).invoke(model_path, tp, rank, world_size)

    init_latency_ms = (time.time() - init_start) * 1000

    prompt = "Hello, my name is"
    warmup_start = time.time()
    obj_ref = actor.generate.invoke(prompt, max_tokens=10)
    yr.wait(obj_ref)
    warmup_latency_ms = (time.time() - warmup_start) * 1000

    result = yr.get(obj_ref) if rank == 0 else None

    actor.terminate()

    return {
        "init_latency_ms": init_latency_ms,
        "warmup_latency_ms": warmup_latency_ms,
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser(description="Fresh process worker for cold startup measurement")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--rank", type=int, default=0, help="Rank")
    parser.add_argument("--world-size", type=int, default=2, help="World size")
    args = parser.parse_args()

    yr.init(yr.Config(log_level="INFO"))

    result = run_single_inference(args.model, args.tp, args.rank, args.world_size)

    print(json.dumps(result))

    yr.finalize()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make script executable**

Run: `chmod +x _fresh_process_worker.py`
Expected: No error

- [ ] **Step 3: Commit**

```bash
git add _fresh_process_worker.py
git commit -m "feat: add _fresh_process_worker.py helper script"
```

---

## Task 3: Create baseline_fresh_process.py

**Files:**
- Create: `baseline_fresh_process.py`

- [ ] **Step 1: Write baseline_fresh_process.py**

```python
#!/usr/bin/env python3
# coding=UTF-8
# Baseline 0: Fresh Process - measures complete cold startup latency

import subprocess
import sys
import time
import json
import os


def run_fresh_process_inference(model_path: str, tp_size: int):
    """Run inference in a fresh process and measure total latency."""
    start_time = time.time()

    proc = subprocess.Popen([
        sys.executable,
        "_fresh_process_worker.py",
        "--model", model_path,
        "--tp", str(tp_size),
        "--rank", "0",
        "--world-size", str(tp_size),
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    stdout, stderr = proc.communicate()
    total_latency_ms = (time.time() - start_time) * 1000

    if proc.returncode != 0:
        print(f"Error in subprocess: {stderr}", file=sys.stderr)
        raise RuntimeError(f"Subprocess failed with return code {proc.returncode}")

    worker_result = json.loads(stdout)
    worker_result["import_runtime_latency_ms"] = total_latency_ms - worker_result["init_latency_ms"]

    return worker_result


def main():
    model_a = "/workspace/models/qwen2.5_7B_Instruct"
    model_b = "/workspace/models/Qwen2.5-VL-7B-Instruct"
    tp_size = 2

    print(f"=== Baseline: Fresh Process (TP={tp_size}) ===")
    print(f"Model A: {model_a}")
    print(f"Model B: {model_b}")
    print()

    print("Running Model A in fresh process...")
    result_a = run_fresh_process_inference(model_a, tp_size)
    print(f"  Import runtime: {result_a['import_runtime_latency_ms']:.2f} ms")
    print(f"  Init latency: {result_a['init_latency_ms']:.2f} ms")
    print(f"  Warmup inference: {result_a['warmup_latency_ms']:.2f} ms")
    print()

    print("Running Model B in fresh process (simulating switch)...")
    result_b = run_fresh_process_inference(model_b, tp_size)
    print(f"  Import runtime: {result_b['import_runtime_latency_ms']:.2f} ms")
    print(f"  Init latency: {result_b['init_latency_ms']:.2f} ms")
    print(f"  Warmup inference: {result_b['warmup_latency_ms']:.2f} ms")
    print()

    switch_latency_ms = result_b['import_runtime_latency_ms'] + result_b['init_latency_ms']

    output = {
        "test_type": "fresh_process",
        "model_a": model_a,
        "model_b": model_b,
        "tp_size": tp_size,
        "import_runtime_latency_ms": result_b['import_runtime_latency_ms'],
        "init_latency_ms": result_a['init_latency_ms'],
        "warmup_latency_ms_avg": result_a['warmup_latency_ms'],
        "switch_latency_ms": switch_latency_ms,
        "post_switch_latency_ms_avg": result_b['warmup_latency_ms'],
    }

    print("=== Results ===")
    print(json.dumps(output, indent=2))

    with open("results_fresh_process.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to results_fresh_process.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make script executable**

Run: `chmod +x baseline_fresh_process.py`
Expected: No error

- [ ] **Step 3: Commit**

```bash
git add baseline_fresh_process.py
git commit -m "feat: add baseline_fresh_process.py test script"
```

---

## Task 4: Create baseline_recreate_actor.py

**Files:**
- Create: `baseline_recreate_actor.py`

- [ ] **Step 1: Write baseline_recreate_actor.py**

```python
#!/usr/bin/env python3
# coding=UTF-8
# Baseline 1: Recreate Actor - recreates Actor for each model

import os
os.environ["VLLM_PLATFORM"] = "ascend"
import torch
import torch_npu
import torch.distributed

import yr
import vllm_actor
from vllm import SamplingParams
import time
import json


def create_actor_and_infer(model_path: str, tp: int, rank: int, world_size: int, prompt: str = "Hello, my name is"):
    """Create Actor, run inference, and return timing."""
    _logger = yr.log.get_logger()

    init_start = time.time()

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if "HCCL_IF_BASE_PORT" not in os.environ:
        os.environ["HCCL_IF_BASE_PORT"] = "50000"

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            init_method="env://",
            backend="cpu:gloo,npu:hccl",
            rank=rank,
            world_size=world_size,
        )

    opts = yr.InvokeOptions()
    opts.cpu = 2000
    opts.memory = 4096
    opts.custom_resources = {"NPU/.+/count": 1}

    actor = vllm_actor.VLLMEngine.options(opts).invoke(model_path, tp, rank, world_size)

    init_latency_ms = (time.time() - init_start) * 1000

    inference_start = time.time()
    obj_ref = actor.generate.invoke(prompt, max_tokens=10)
    yr.wait(obj_ref)
    inference_latency_ms = (time.time() - inference_start) * 1000

    result = yr.get(obj_ref) if rank == 0 else None

    actor.terminate()

    return {
        "init_latency_ms": init_latency_ms,
        "inference_latency_ms": inference_latency_ms,
        "result": result,
    }


def main():
    model_a = "/workspace/models/qwen2.5_7B_Instruct"
    model_b = "/workspace/models/Qwen2.5-VL-7B-Instruct"
    tp_size = 2
    rank = 0
    world_size = 2

    yr.init(yr.Config(log_level="INFO"))

    print(f"=== Baseline: Recreate Actor (TP={tp_size}) ===")
    print(f"Model A: {model_a}")
    print(f"Model B: {model_b}")
    print()

    print("Creating Actor for Model A...")
    result_a = create_actor_and_infer(model_a, tp_size, rank, world_size)
    print(f"  Init latency: {result_a['init_latency_ms']:.2f} ms")
    print(f"  Inference: {result_a['inference_latency_ms']:.2f} ms")
    print()

    print("Creating Actor for Model B (simulating switch)...")
    switch_start = time.time()
    result_b = create_actor_and_infer(model_b, tp_size, rank, world_size)
    switch_latency_ms = (time.time() - switch_start) * 1000
    print(f"  Switch latency: {switch_latency_ms:.2f} ms")
    print(f"  Inference: {result_b['inference_latency_ms']:.2f} ms")
    print()

    output = {
        "test_type": "recreate_actor",
        "model_a": model_a,
        "model_b": model_b,
        "tp_size": tp_size,
        "init_latency_ms": result_a['init_latency_ms'],
        "warmup_latency_ms_avg": result_a['inference_latency_ms'],
        "switch_latency_ms": switch_latency_ms,
        "post_switch_latency_ms_avg": result_b['inference_latency_ms'],
    }

    print("=== Results ===")
    print(json.dumps(output, indent=2))

    with open("results_recreate_actor.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to results_recreate_actor.json")

    yr.finalize()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make script executable**

Run: `chmod +x baseline_recreate_actor.py`
Expected: No error

- [ ] **Step 3: Commit**

```bash
git add baseline_recreate_actor.py
git commit -m "feat: add baseline_recreate_actor.py test script"
```

---

## Task 5: Create baseline_recreate_llm.py

**Files:**
- Create: `baseline_recreate_llm.py`

- [ ] **Step 1: Write baseline_recreate_llm.py**

```python
#!/usr/bin/env python3
# coding=UTF-8
# Baseline 2: Recreate LLM - recreates LLM instance within same Actor

import os
os.environ["VLLM_PLATFORM"] = "ascend"
import torch
import torch_npu
import torch.distributed

import yr
import vllm_actor
from vllm import SamplingParams
import time
import json


def main():
    model_a = "/workspace/models/qwen2.5_7B_Instruct"
    model_b = "/workspace/models/Qwen2.5-VL-7B-Instruct"
    tp_size = 2
    rank = 0
    world_size = 2

    yr.init(yr.Config(log_level="INFO"))

    print(f"=== Baseline: Recreate LLM (TP={tp_size}) ===")
    print(f"Model A: {model_a}")
    print(f"Model B: {model_b}")
    print()

    # Setup environment variables (shared)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if "HCCL_IF_BASE_PORT" not in os.environ:
        os.environ["HCCL_IF_BASE_PORT"] = "50000"

    # Initialize HCCL once
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            init_method="env://",
            backend="cpu:gloo,npu:hccl",
            rank=rank,
            world_size=world_size,
        )

    # Create Actor once (kept alive)
    opts = yr.InvokeOptions()
    opts.cpu = 2000
    opts.memory = 4096
    opts.custom_resources = {"NPU/.+/count": 1}

    print("Creating Actor...")
    actor = vllm_actor.VLLMEngine.options(opts).invoke(model_a, tp_size, rank, world_size)
    print()

    # Warmup inference with Model A
    prompt = "Hello, my name is"
    print(f"Warmup inference with Model A: '{prompt}'")
    warmup_start = time.time()
    obj_ref = actor.generate.invoke(prompt, max_tokens=10)
    yr.wait(obj_ref)
    warmup_latency_ms = (time.time() - warmup_start) * 1000
    print(f"  Warmup inference: {warmup_latency_ms:.2f} ms")
    print()

    # Switch to Model B by recreating Actor
    print("Switching to Model B (recreating LLM)...")
    switch_start = time.time()

    # Terminate old actor
    actor.terminate()

    # Create new actor with Model B
    actor = vllm_actor.VLLMEngine.options(opts).invoke(model_b, tp_size, rank, world_size)

    switch_latency_ms = (time.time() - switch_start) * 1000
    print(f"  Switch latency: {switch_latency_ms:.2f} ms")
    print()

    # Inference with Model B
    print(f"Inference with Model B: '{prompt}'")
    inference_start = time.time()
    obj_ref = actor.generate.invoke(prompt, max_tokens=10)
    yr.wait(obj_ref)
    post_switch_latency_ms = (time.time() - inference_start) * 1000
    print(f"  Inference: {post_switch_latency_ms:.2f} ms")
    print()

    # Cleanup
    actor.terminate()
    yr.finalize()

    # Output result
    output = {
        "test_type": "recreate_llm",
        "model_a": model_a,
        "model_b": model_b,
        "tp_size": tp_size,
        "init_latency_ms": 0,  # N/A - not measured in this test
        "warmup_latency_ms_avg": warmup_latency_ms,
        "switch_latency_ms": switch_latency_ms,
        "post_switch_latency_ms_avg": post_switch_latency_ms,
    }

    print("=== Results ===")
    print(json.dumps(output, indent=2))

    with open("results_recreate_llm.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to results_recreate_llm.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make script executable**

Run: `chmod +x baseline_recreate_llm.py`
Expected: No error

- [ ] **Step 3: Commit**

```bash
git add baseline_recreate_llm.py
git commit -m "feat: add baseline_recreate_llm.py test script"
```

---

## Task 6: Create test_sleep_mode.py

**Files:**
- Create: `test_sleep_mode.py`

- [ ] **Step 1: Write test_sleep_mode.py**

```python
#!/usr/bin/env python3
# coding=UTF-8
# Feature Test: Sleep Mode - uses sleep/wake_up to reuse executor

import os
os.environ["VLLM_PLATFORM"] = "ascend"
import torch
import torch_npu
import torch.distributed

import yr
import vllm_actor
from vllm import SamplingParams
import time
import json


def main():
    model_a = "/workspace/models/qwen2.5_7B_Instruct"
    model_b = "/workspace/models/Qwen2.5-VL-7B-Instruct"
    tp_size = 2
    rank = 0
    world_size = 2

    yr.init(yr.Config(log_level="INFO"))

    print(f"=== Sleep Mode Test (TP={tp_size}) ===")
    print(f"Model A: {model_a}")
    print(f"Model B: {model_b}")
    print()

    # Setup environment variables
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if "HCCL_IF_BASE_PORT" not in os.environ:
        os.environ["HCCL_IF_BASE_PORT"] = "50000"

    # Initialize HCCL
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            init_method="env://",
            backend="cpu:gloo,npu:hccl",
            rank=rank,
            world_size=world_size,
        )

    # Create Actor with Model A
    opts = yr.InvokeOptions()
    opts.cpu = 2000
    opts.memory = 4096
    opts.custom_resources = {"NPU/.+/count": 1}

    print("Creating Actor with Model A...")
    init_start = time.time()
    actor = vllm_actor.VLLMEngine.options(opts).invoke(model_a, tp_size, rank, world_size)
    init_latency_ms = (time.time() - init_start) * 1000
    print(f"  Init latency: {init_latency_ms:.2f} ms")
    print()

    # Warmup inference with Model A
    prompt = "Hello, my name is"
    print(f"Warmup inference with Model A: '{prompt}'")
    warmup_start = time.time()
    obj_ref = actor.generate.invoke(prompt, max_tokens=10)
    yr.wait(obj_ref)
    warmup_latency_ms = (time.time() - warmup_start) * 1000
    print(f"  Warmup inference: {warmup_latency_ms:.2f} ms")
    print()

    # Switch to Model B using sleep mode
    print("Switching to Model B using sleep mode...")
    switch_start = time.time()

    # Sleep to free model weights
    actor.sleep(level=1)

    # Wake up and switch to new model
    actor.switch_model(model_b)

    switch_latency_ms = (time.time() - switch_start) * 1000
    print(f"  Switch latency: {switch_latency_ms:.2f} ms")
    print()

    # Inference with Model B
    print(f"Inference with Model B: '{prompt}'")
    inference_start = time.time()
    obj_ref = actor.generate.invoke(prompt, max_tokens=10)
    yr.wait(obj_ref)
    post_switch_latency_ms = (time.time() - inference_start) * 1000
    print(f"  Inference: {post_switch_latency_ms:.2f} ms")
    print()

    # Second switch back to Model A (optional, verify multiple switches)
    print("Second switch back to Model A...")
    second_switch_start = time.time()
    actor.sleep(level=1)
    actor.switch_model(model_a)
    second_switch_latency_ms = (time.time() - second_switch_start) * 1000
    print(f"  Second switch latency: {second_switch_latency_ms:.2f} ms")
    print()

    # Cleanup
    actor.terminate()
    yr.finalize()

    # Output result
    output = {
        "test_type": "sleep_mode",
        "model_a": model_a,
        "model_b": model_b,
        "tp_size": tp_size,
        "init_latency_ms": init_latency_ms,
        "warmup_latency_ms_avg": warmup_latency_ms,
        "switch_latency_ms": switch_latency_ms,
        "post_switch_latency_ms_avg": post_switch_latency_ms,
        "second_switch_latency_ms": second_switch_latency_ms,
    }

    print("=== Results ===")
    print(json.dumps(output, indent=2))

    with open("results_sleep_mode.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to results_sleep_mode.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make script executable**

Run: `chmod +x test_sleep_mode.py`
Expected: No error

- [ ] **Step 3: Commit**

```bash
git add test_sleep_mode.py
git commit -m "feat: add test_sleep_mode.py feature test"
```

---

## Task 7: Create compare_results.py

**Files:**
- Create: `compare_results.py`

- [ ] **Step 1: Write compare_results.py**

```python
#!/usr/bin/env python3
# coding=UTF-8
# Compare Results - aggregate and compare all test results

import json
import os
import sys
from typing import Dict, Optional


def load_result(filename: str) -> Optional[Dict]:
    """Load JSON result file."""
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found", file=sys.stderr)
        return None
    with open(filename, "r") as f:
        return json.load(f)


def calculate_savings(baseline: float, optimized: float) -> Dict[str, float]:
    """Calculate time savings percentage."""
    if baseline == 0:
        return {"savings_ms": 0, "savings_pct": 0}
    savings_ms = baseline - optimized
    savings_pct = (savings_ms / baseline) * 100
    return {"savings_ms": savings_ms, "savings_pct": savings_pct}


def main():
    print("=" * 80)
    print("Sleep Mode Model Switching - Performance Comparison")
    print("=" * 80)
    print()

    # Load all results
    fresh_process = load_result("results_fresh_process.json")
    recreate_actor = load_result("results_recreate_actor.json")
    recreate_llm = load_result("results_recreate_llm.json")
    sleep_mode = load_result("results_sleep_mode.json")

    # Display configuration
    if sleep_mode:
        print(f"Model A: {sleep_mode['model_a']}")
        print(f"Model B: {sleep_mode['model_b']}")
        print(f"TP Size: {sleep_mode['tp_size']}")
        print()

    # Display switch latency comparison
    print("=" * 80)
    print("Model Switch Latency Comparison")
    print("=" * 80)
    print()
    print(f"{'Test Type':<25} {'Switch (ms)':<15} {'Savings':<15} {'%'}")
    print("-" * 80)

    baseline_latency = None
    if fresh_process:
        latency = fresh_process.get('switch_latency_ms', 0)
        print(f"{'Fresh Process':<25} {latency:<15.2f} {'-':<15} {'N/A':<5}")
        baseline_latency = latency

    if recreate_actor:
        latency = recreate_actor.get('switch_latency_ms', 0)
        if baseline_latency:
            savings = calculate_savings(baseline_latency, latency)
            print(f"{'Recreate Actor':<25} {latency:<15.2f} {savings['savings_ms']:<15.2f} {savings['savings_pct']:<5.1f}")
        else:
            print(f"{'Recreate Actor':<25} {latency:<15.2f} {'-':<15} {'N/A':<5}")

    if recreate_llm:
        latency = recreate_llm.get('switch_latency_ms', 0)
        if baseline_latency:
            savings = calculate_savings(baseline_latency, latency)
            print(f"{'Recreate LLM':<25} {latency:<15.2f} {savings['savings_ms']:<15.2f} {savings['savings_pct']:<5.1f}")
        else:
            print(f"{'Recreate LLM':<25} {latency:<15.2f} {'-':<15} {'N/A':<5}")

    if sleep_mode:
        latency = sleep_mode.get('switch_latency_ms', 0)
        if baseline_latency:
            savings = calculate_savings(baseline_latency, latency)
            print(f"{'Sleep Mode (Feature)':<25} {latency:<15.2f} {savings['savings_ms']:<15.2f} {savings['savings_pct']:<5.1f}")
        else:
            print(f"{'Sleep Mode (Feature)':<25} {latency:<15.2f} {'-':<15} {'N/A':<5}")

    print()
    print("=" * 80)

    # Display import runtime breakdown (if available)
    if fresh_process and 'import_runtime_latency_ms' in fresh_process:
        print()
        print("=" * 80)
        print("Fresh Process Breakdown")
        print("=" * 80)
        print()
        print(f"Import Runtime: {fresh_process['import_runtime_latency_ms']:.2f} ms")
        print(f"Init: {fresh_process['init_latency_ms']:.2f} ms")
        print(f"Warmup Inference: {fresh_process['warmup_latency_ms_avg']:.2f} ms")
        total = fresh_process['import_runtime_latency_ms'] + fresh_process['init_latency_ms']
        print(f"Total Startup: {total:.2f} ms")
        print()

        # Calculate percentages
        if total > 0:
            import_pct = (fresh_process['import_runtime_latency_ms'] / total) * 100
            init_pct = (fresh_process['init_latency_ms'] / total) * 100
            print(f"Import Runtime占比: {import_pct:.1f}%")
            print(f"Init占比: {init_pct:.1f}%")
            print()

    # Display second switch comparison (if available)
    if sleep_mode and 'second_switch_latency_ms' in sleep_mode:
        print("=" * 80)
        print("Multiple Switch Comparison")
        print("=" * 80)
        print()
        first_switch = sleep_mode.get('switch_latency_ms', 0)
        second_switch = sleep_mode.get('second_switch_latency_ms', 0)
        print(f"First Switch (A->B): {first_switch:.2f} ms")
        print(f"Second Switch (B->A): {second_switch:.2f} ms")
        print()

        if first_switch > 0:
            diff_pct = abs(second_switch - first_switch) / first_switch * 100
            print(f"Variance: {diff_pct:.1f}%")
            print()

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()

    if sleep_mode and fresh_process:
        switch_ms = sleep_mode.get('switch_latency_ms', 0)
        baseline_ms = fresh_process.get('switch_latency_ms', 0)
        savings = calculate_savings(baseline_ms, switch_ms)

        print(f"Sleep Mode reduces model switch latency by:")
        print(f"  {savings['savings_ms']:.2f} ms ({savings['savings_pct']:.1f}%)")
        print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile compare_results.py`
Expected: No syntax errors

- [ ] **Step 3: Make script executable**

Run: `chmod +x compare_results.py`
Expected: No error

- [ ] **Step 4: Commit**

```bash
git add compare_results.py
git commit -m "feat: add compare_results.py comparison script"
```

---

## Task 8: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add performance testing section to README**

At the end of README.md, add:

```markdown
## Performance Testing

This project includes performance tests to measure model switching latency using different approaches:

### Test Scripts

- `baseline_fresh_process.py`: Full cold startup (includes import runtime)
- `baseline_recreate_actor.py`: Recreate Actor for each model
- `baseline_recreate_llm.py`: Recreate LLM within same Actor
- `test_sleep_mode.py`: Use sleep_mode to reuse executor

### Running Tests

```bash
# Run individual tests
python baseline_fresh_process.py
python baseline_recreate_actor.py
python baseline_recreate_llm.py
python test_sleep_mode.py

# Compare results
python compare_results.py
```

### Expected Result

The `sleep_mode` approach should show significantly lower model switching latency
compared to baseline approaches, as it reuses executor process
and only swaps model weights.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add performance testing section to README"
```

---

## Task 9: Integration Testing

**Files:**
- Test: All scripts

- [ ] **Step 1: Verify all scripts have correct syntax**

Run: `python -m py_compile vllm_actor.py baseline_fresh_process.py _fresh_process_worker.py baseline_recreate_actor.py baseline_recreate_llm.py test_sleep_mode.py compare_results.py`
Expected: No syntax errors

- [ ] **Step 2: Check all scripts are executable**

Run: `ls -l *.py | grep -E "(baseline_|test_|compare_|_fresh)" | grep -v main.py`
Expected: All test scripts have execute permissions

- [ ] **Step 3: Verify vllm_actor.py has new methods**

Run: `grep -E "(def sleep|def wake_up|def switch_model)" vllm_actor.py`
Expected: Show three new method definitions

- [ ] **Step 4: Final integration commit**

```bash
git commit --allow-empty -m "test: integration test complete - all scripts verified"
```

---

## Task 10: Documentation

**Files:**
- Create: `TESTING.md`

- [ ] **Step 1: Create TESTING.md documentation**

```markdown
# Testing Guide

## Prerequisites

- Yuanrong Actor framework installed
- vLLM-ascend with external_launcher support
- PyTorch with NPU (Ascend) support
- Model files available at configured paths

## Test Configuration

Edit model paths in test scripts if needed:

```python
model_a = "/workspace/models/qwen2.5_7B_Instruct"
model_b = "/workspace/models/Qwen2.5-VL-7B-Instruct"
tp_size = 2
```

## Running Tests

### Individual Tests

Each test outputs a JSON result file:

```bash
python baseline_fresh_process.py    # -> results_fresh_process.json
python baseline_recreate_actor.py   # -> results_recreate_actor.json
python baseline_recreate_llm.py    # -> results_recreate_llm.json
python test_sleep_mode.py           # -> results_sleep_mode.json
```

### Compare Results

```bash
python compare_results.py
```

This displays:
- Latency comparison table
- Time savings vs baseline
- Import runtime breakdown (fresh process)
- Multiple switch variance

## Interpreting Results

### Switch Latency

Lower is better. Expected order:
```
fresh_process > recreate_actor > recreate_llm > sleep_mode
```

### Import Runtime Percentage

Shows how much time is spent on `import torch` and related libraries.
This cost is avoided when reusing processes/actors.

### Expected Savings

Sleep mode should save significant time compared to baselines:
- vs Fresh Process: 40-60% savings (depends on model size)
- vs Recreate Actor: 20-40% savings
- vs Recreate LLM: 10-30% savings
```

- [ ] **Step 2: Commit**

```bash
git add TESTING.md
git commit -m "docs: add TESTING.md documentation"
```
