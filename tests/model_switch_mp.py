#!/usr/bin/env python3
# coding=UTF-8
# Test: Model switching with vLLM multiprocess backend using sleep mode
#
# Usage:
#   cd /home/hhc/yr_wp/yr_vllm_launcher/tests
#   python model_switch_mp.py
#
# Description:
#   Tests model switching (Model A <-> Model B) with vLLM mp backend.
#   - Uses distributed_executor_backend="mp"
#   - Uses enable_sleep_mode=True for fast switching
#   - Both models are loaded initially, then sleep/wake_up is used for switching
#   - Each LLM instance gets unique HCCL port to avoid conflicts
#   - Measures and logs: wake_up time, inference time, NPU memory

import os
import time
import torch
from vllm import LLM, SamplingParams
os.environ["HCCL_HOST_SOCKET_PORT_RANGE"] = "60050-60100"
os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "61050-61100"
os.environ["VLLM_ASCEND_ENABLE_NZ"] = "0"


def get_npu_memory() -> dict:
    """Get NPU memory usage using torch.npu API."""
    try:
        free, total = torch.npu.mem_get_info()
        allocated = total - free
        return {"allocated_gb": allocated / 1024**3, "free_gb": free / 1024**3, "total_gb": total / 1024**3}
    except Exception as e:
        print(f"Warning: failed to get NPU memory info: {e}")
        return {"allocated_gb": 0.0, "free_gb": 0.0, "total_gb": 0.0}


class SleepModeModelSwitch:
    """Model switch benchmark using vLLM sleep mode."""

    def __init__(self, tensor_parallel_size=4, base_hccl_port=50000):
        self.tensor_parallel_size = tensor_parallel_size
        self._base_hccl_port = base_hccl_port  # Base HCCL port
        self.models = {}  # model_path -> LLM instance
        self.active_model = None

    def load_models(self, model_paths: list):
        """Load all models with sleep mode enabled."""
        print(f"\n=== Loading {len(model_paths)} models with sleep mode ===")
        start_time = time.time()

        for idx, model_path in enumerate(model_paths):
            # 为每个模型设置唯一的 HCCL 端口
            hccl_port = self._base_hccl_port + (idx * 10)
            os.environ["HCCL_IF_BASE_PORT"] = str(hccl_port)

            print(f"Loading model: {model_path} (HCCL port: {hccl_port})")
            llm = LLM(
                model=model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=True,
                enable_sleep_mode=True,
                distributed_executor_backend="mp",
                gpu_memory_utilization = 0.7,
                kv_cache_memory_bytes=1 * 1024**3,
                max_model_len=2048,
                max_num_seqs=2,
            )
            llm.sleep(level=2)
            self.models[model_path] = llm

        elapsed = time.time() - start_time
        mem = get_npu_memory()
        print(f"All models loaded in {elapsed:.2f}s, NPU: {mem['allocated_gb']:.2f}GB")

    def switch_to_model(self, model_path: str):
        """Switch to target model using sleep/wake_up."""
        print(f"\n=== Switching to model: {model_path} ===")
        start_time = time.time()

        # Put current model to sleep (level=2 releases both NPU and CPU memory)
        if self.active_model and self.active_model != model_path:
            print(f"Sleeping current model: {self.active_model}")
            self.models[self.active_model].sleep(level=2)

        # Wake up target model
        print(f"Waking up model: {model_path}")
        self.models[model_path].wake_up()

        elapsed = time.time() - start_time
        mem = get_npu_memory()
        print(f"Switch completed in {elapsed:.2f}s, NPU: {mem['allocated_gb']:.2f}GB")

        self.active_model = model_path
        return self.models[model_path]

    def run_inference(self, prompt: str, max_tokens: int = 10) -> str:
        """Run inference on active model."""
        if not self.active_model:
            raise RuntimeError("No active model")

        print(f"\n=== Running inference ===")
        start_time = time.time()

        sampling_params = SamplingParams(
            temperature=0.8, top_p=0.95, max_tokens=max_tokens, logprobs=1
        )
        outputs = self.models[self.active_model].generate([prompt], sampling_params)
        result = outputs[0].outputs[0].text

        elapsed = time.time() - start_time
        print(f"Inference completed in {elapsed:.2f}s")
        print(f"Prompt: {prompt}")
        print(f"Result: {result}")
        return result

    def cleanup(self):
        """Clean up all models."""
        print("\n=== Cleanup ===")
        for model_path, llm in self.models.items():
            try:
                llm.sleep(level=2)
                del llm
            except Exception as e:
                print(f"Error cleaning up {model_path}: {e}")
        self.models.clear()

        # Force NPU cleanup
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
        mem = get_npu_memory()
        print(f"Cleanup done, NPU: {mem['allocated_gb']:.2f}GB")


def main():
    """Main test function."""
    # Environment setup
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"

    # Model paths
    model_a = "/workspace/models/qwen2.5_7B_Instruct"
    model_b = "/workspace/models/Qwen2.5-VL-7B-Instruct"

    print(f"=== Model Switch Benchmark: MP Backend + Sleep Mode ===")
    print(f"Model A: {model_a}")
    print(f"Model B: {model_b}")

    # Print initial NPU memory
    initial_mem = get_npu_memory()
    print(f"\nInitial NPU memory: {initial_mem['allocated_gb']:.2f}GB (free: {initial_mem['free_gb']:.2f}GB)")

    # Create benchmark and load both models
    # Each model will use a different HCCL port (50000, 50010, ...)
    benchmark = SleepModeModelSwitch(tensor_parallel_size=4, base_hccl_port=50000)
    benchmark.load_models([model_a, model_b])

    # First activation: wake up Model A
    benchmark.switch_to_model(model_a)
    benchmark.run_inference("Hello, my name is", max_tokens=10)

    # Switch to Model B (A sleeps, B wakes up)
    benchmark.switch_to_model(model_b)
    benchmark.run_inference("Hello, my name is", max_tokens=10)

    # Switch back to Model A (B sleeps, A wakes up)
    benchmark.switch_to_model(model_a)
    benchmark.run_inference("Hello, my name is", max_tokens=10)

    # Cleanup
    benchmark.cleanup()

    # Print final NPU memory
    final_mem = get_npu_memory()
    print(f"\nFinal NPU memory: {final_mem['allocated_gb']:.2f}GB (free: {final_mem['free_gb']:.2f}GB)")

    print("\n=== Test completed successfully! ===")


if __name__ == "__main__":
    main()
