#!/usr/bin/env python3
# coding=UTF-8
# Test: Model switching with vLLM MP backend using Yuanrong Actor
#
# Usage:
#   cd /home/hhc/yr_wp/yr_vllm_launcher/tests
#   python model_switch_mp_actors.py
#
# Description:
#   Tests model switching (Model A <-> Model B) with vLLM MP backend.
#   - Uses Yuanrong Actor for process isolation (different HCCL ports)
#   - Uses distributed_executor_backend="mp" + enable_sleep_mode=True
#   - Each Actor is a separate process with unique HCCL port
#   - Switching via Actor RPC: sleep() and wake_up()
#   - Measures and logs: wake_up time, inference time, NPU memory

import os
import time
import torch

import yr
import yr.log
from vllm import LLM, SamplingParams


def get_npu_memory() -> dict:
    """Get NPU memory usage using torch.npu API."""
    try:
        free, total = torch.npu.mem_get_info()
        allocated = total - free
        return {"allocated_gb": allocated / 1024**3, "free_gb": free / 1024**3, "total_gb": total / 1024**3}
    except Exception as e:
        print(f"Warning: failed to get NPU memory info: {e}")
        return {"allocated_gb": 0.0, "free_gb": 0.0, "total_gb": 0.0}


@yr.instance
class VLLMEngineMP:
    """Yuanrong Actor wrapping vLLM with MP backend + sleep mode."""

    def __init__(
        self,
        model_path: str,
        tp: int = 2,
        hccl_port: int = 50000,
    ):
        """
        Initialize vLLM Actor with MP backend.

        Args:
            model_path: Model path
            tp: Tensor parallel size
            hccl_port: Unique HCCL port for this Actor
        """
        _logger = yr.log.get_logger()
        _logger.info(
            f"VLLMEngineMP __init__: model={model_path}, tp={tp}, hccl_port={hccl_port}"
        )

        self._model_path = model_path
        self._tp = tp
        self._hccl_port = hccl_port

        # Setup environment variables for MP backend
        os.environ["HCCL_IF_BASE_PORT"] = str(hccl_port)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Create LLM with MP backend and sleep mode
        _logger.info(f"Creating LLM with MP backend: model={model_path}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp,
            trust_remote_code=True,
            enable_sleep_mode=True,
            distributed_executor_backend="mp",
        )
        _logger.info("LLM created successfully")

        # Initially sleep to release memory
        _logger.info("Putting model to sleep (level=2)")
        self.llm.sleep(level=2)
        _logger.info("Model sleeping, memory released")

    def wake_up(self):
        """Wake up model, load weights from disk to NPU."""
        _logger = yr.log.get_logger()
        _logger.info(f"Waking up model: {self._model_path}")
        self.llm.wake_up()
        _logger.info("Model woke up successfully")

    def sleep(self, level: int = 2):
        """Sleep model, release NPU and CPU memory."""
        _logger = yr.log.get_logger()
        _logger.info(f"Sleeping model: {self._model_path}, level={level}")
        self.llm.sleep(level=level)
        _logger.info("Model sleeping, memory released")

    def generate(self, prompt: str, max_tokens: int = 10) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        _logger = yr.log.get_logger()
        _logger.info(
            f"generate called: prompt='{prompt}', max_tokens={max_tokens}"
        )
        sampling_params = SamplingParams(
            temperature=0.8, top_p=0.95, max_tokens=max_tokens, logprobs=1
        )
        outputs = self.llm.generate([prompt], sampling_params)
        result = outputs[0].outputs[0].text
        _logger.info(f"Generated: {result[:100]}...")
        return result

    def ready(self) -> bool:
        """Return True if actor is ready."""
        return True

    def cleanup(self):
        """Cleanup resources."""
        _logger = yr.log.get_logger()
        _logger.info(f"Cleaning up model: {self._model_path}")
        try:
            self.llm.sleep(level=2)
            del self.llm
        except Exception as e:
            _logger.warning(f"Cleanup error: {e}")
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
        _logger.info("Cleanup completed")


def main():
    """Main test function."""
    # 初始化 Yuanrong
    yr.init(yr.Config(log_level="INFO"))

    model_a = "/workspace/models/qwen2.5_7B_Instruct"
    model_b = "/workspace/models/Qwen2.5-VL-7B-Instruct"

    print(f"=== Model Switch Benchmark: Yuanrong Actor + MP Backend + Sleep Mode ===")
    print(f"Model A: {model_a}")
    print(f"Model B: {model_b}")

    # Print initial NPU memory
    initial_mem = get_npu_memory()
    print(f"\nInitial NPU memory: {initial_mem['allocated_gb']:.2f}GB (free: {initial_mem['free_gb']:.2f}GB)")

    # Create two Actor instances with different HCCL ports
    # 顺序创建：actor_a 先初始化并 sleep，再创建 actor_b
    print("\n=== Creating Actor A ===")
    actor_a = VLLMEngineMP.invoke(
        model_path=model_a,
        tp=2,
        hccl_port=50000,
    )
    actor_a.ready.invoke().wait()
    print("Actor A ready (in sleep mode)")

    mem_after_a = get_npu_memory()
    print(f"NPU memory after Actor A: {mem_after_a['allocated_gb']:.2f}GB (free: {mem_after_a['free_gb']:.2f}GB)")

    print("\n=== Creating Actor B ===")
    actor_b = VLLMEngineMP.invoke(
        model_path=model_b,
        tp=2,
        hccl_port=50010,
    )
    actor_b.ready.invoke().wait()
    print("Actor B ready (in sleep mode)")

    mem_after_init = get_npu_memory()
    print(f"NPU memory after init: {mem_after_init['allocated_gb']:.2f}GB (free: {mem_after_init['free_gb']:.2f}GB)")

    # === First: Activate Model A ===
    print("\n=== Switching to Model A ===")
    start_time = time.time()
    actor_a.wake_up.invoke().wait()
    switch_time = time.time() - start_time
    mem_a = get_npu_memory()
    print(f"Model A woke up in {switch_time:.2f}s, NPU: {mem_a['allocated_gb']:.2f}GB")

    # Run inference on Model A
    print("\n=== Running inference on Model A ===")
    infer_start = time.time()
    result_a = yr.get(actor_a.generate.invoke("Hello, my name is", 10))
    infer_time = time.time() - infer_start
    print(f"Inference completed in {infer_time:.2f}s")
    print(f"Result: {result_a}")

    # === Switch to Model B (A sleeps, B wakes up) ===
    print("\n=== Switching to Model B ===")
    switch_start = time.time()
    actor_a.sleep.invoke(level=2).wait()
    actor_b.wake_up.invoke().wait()
    switch_time = time.time() - switch_start
    mem_b = get_npu_memory()
    print(f"Switch completed in {switch_time:.2f}s, NPU: {mem_b['allocated_gb']:.2f}GB")

    # Run inference on Model B
    print("\n=== Running inference on Model B ===")
    infer_start = time.time()
    result_b = yr.get(actor_b.generate.invoke("Hello, my name is", 10))
    infer_time = time.time() - infer_start
    print(f"Inference completed in {infer_time:.2f}s")
    print(f"Result: {result_b}")

    # === Switch back to Model A (B sleeps, A wakes up) ===
    # (跳过第二次切换，只测试一次切换)

    # Cleanup
    print("\n=== Cleanup ===")
    actor_a.cleanup.invoke().wait()
    actor_b.cleanup.invoke().wait()

    # Print final NPU memory
    final_mem = get_npu_memory()
    print(f"\nFinal NPU memory: {final_mem['allocated_gb']:.2f}GB (free: {final_mem['free_gb']:.2f}GB)")

    print("\n=== Test completed successfully! ===")


    yr.finalize()


if __name__ == "__main__":
    main()
