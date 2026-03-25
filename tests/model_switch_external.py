#!/usr/bin/env python3
# coding=UTF-8
# Test: Model switching with vLLM external_launcher backend
#
# Usage:
#   cd /home/hhc/yr_wp/yr_vllm_launcher/tests
#   torchrun --nproc-per-node=2 model_switch_external.py
#
# Description:
#   Tests model switching (Model A <-> Model B) with vLLM external_launcher backend.
#   - Uses distributed_executor_backend="external_launcher"
#   - Cannot have multiple LLM instances - must destroy and recreate
#   - Uses different GLOO/HCCL ports for each model to avoid conflicts
#   - Measures and logs: load time, inference time, NPU memory

import os
import time
import gc
import datetime
import torch
import torch.distributed
from torch.distributed import TCPStore
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)


def get_npu_memory() -> dict:
    """Get NPU memory usage using torch.npu API."""
    try:
        free, total = torch.npu.mem_get_info()
        allocated = total - free
        return {"allocated_gb": allocated / 1024**3, "free_gb": free / 1024**3, "total_gb": total / 1024**3}
    except Exception as e:
        print(f"Warning: failed to get NPU memory info: {e}")
        return {"allocated_gb": 0.0, "free_gb": 0.0, "total_gb": 0.0}


def init_distributed(port: int, rank: int, world_size: int) -> TCPStore:
    """Initialize torch.distributed with TCPStore."""
    print(f"[Rank {rank}] Initializing distributed on port {port}...")

    store = TCPStore(
        host_name="127.0.0.1",
        port=port,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=datetime.timedelta(seconds=30),
    )

    torch.distributed.init_process_group(
        backend="cpu:gloo,npu:hccl",
        store=store,
        rank=rank,
        world_size=world_size,
    )

    print(f"[Rank {rank}] Distributed initialized")
    return store


def cleanup_model(llm, store, rank):
    """
    Cleanup model and distributed resources completely.

    Follows same order as vllm_actor.py:cleanup():
    1. Delete TCPStore
    2. Delete LLM instance
    3. destroy_model_parallel()
    4. destroy_distributed_environment()
    5. destroy_process_group()
    6. gc.collect()
    7. torch.npu.empty_cache()
    """
    print(f"[Rank {rank}] === Cleanup ===")

    # 1. Delete TCPStore
    if store is not None:
        del store

    # 2. Delete LLM instance
    if llm is not None:
        del llm

    # 3. Destroy vLLM model parallel state
    try:
        destroy_model_parallel()
    except Exception as e:
        print(f"[Rank {rank}] destroy_model_parallel failed: {e}")

    # 4. Destroy vLLM distributed environment
    try:
        destroy_distributed_environment()
    except Exception as e:
        print(f"[Rank {rank}] destroy_distributed_environment failed: {e}")

    # 5. Destroy torch.distributed process group
    if torch.distributed.is_initialized():
        try:
            torch.distributed.barrier()
        except:
            pass
        torch.distributed.destroy_process_group()

    # 6. Force garbage collection
    gc.collect()

    # 7. Clear NPU cache
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()

    mem = get_npu_memory()
    print(f"[Rank {rank}] Cleanup done, NPU: {mem['allocated_gb']:.2f}GB")


def load_model(model_path: str, tensor_parallel_size: int, gloo_port: int, hccl_port: int, rank: int, world_size: int) -> tuple:
    """Load a single model with specified ports."""
    print(f"\n[Rank {rank}] === Loading model: {model_path} ===")
    print(f"[Rank {rank}] Ports: GLOO={gloo_port}, HCCL={hccl_port}")
    start_time = time.time()

    # Update HCCL port for this model
    os.environ["HCCL_IF_BASE_PORT"] = str(hccl_port)

    # Initialize distributed
    store = init_distributed(gloo_port, rank, world_size)

    # Create LLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        distributed_executor_backend="external_launcher",
    )

    elapsed = time.time() - start_time
    mem = get_npu_memory()
    print(f"[Rank {rank}] Model loaded in {elapsed:.2f}s, NPU: {mem['allocated_gb']:.2f}GB")

    return llm, store


def run_inference(llm, prompt: str, max_tokens: int = 10, rank: int = 0) -> str:
    """Run inference on given model."""
    print(f"\n[Rank {rank}] === Running inference ===")
    start_time = time.time()

    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=max_tokens, logprobs=1
    )
    outputs = llm.generate([prompt], sampling_params)
    result = outputs[0].outputs[0].text

    elapsed = time.time() - start_time
    print(f"[Rank {rank}] Inference completed in {elapsed:.2f}s")

    # Only rank 0 prints results
    if rank == 0:
        print(f"Prompt: {prompt}")
        print(f"Result: {result}")

    return result


def main():
    """Main test function."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    print(f"[Rank {rank}] === Model Switch Benchmark: External_Launcher (Destroy & Recreate) ===")

    # Environment setup
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"

    # Model paths
    model_a = "/workspace/models/qwen2.5_7B_Instruct"
    model_b = "/workspace/models/Qwen2.5-VL-7B-Instruct"

    # Port increment for each model load to avoid conflicts
    port_offset = 0
    base_gloo_port = 29510
    base_hccl_port = 50000

    # Print initial NPU memory
    initial_mem = get_npu_memory()
    print(f"[Rank {rank}] Initial NPU memory: {initial_mem['allocated_gb']:.2f}GB (free: {initial_mem['free_gb']:.2f}GB)")

    # Load Model A (ports 29510, 50000)
    llm_a, store_a = load_model(model_a, tensor_parallel_size=2,
                                  gloo_port=base_gloo_port + port_offset,
                                  hccl_port=base_hccl_port + port_offset,
                                  rank=rank, world_size=world_size)
    run_inference(llm_a, "Hello, my name is", max_tokens=10, rank=rank)

    # Cleanup Model A
    cleanup_model(llm_a, store_a, rank)
    port_offset += 10

    # Load Model B (ports 29520, 50010)
    llm_b, store_b = load_model(model_b, tensor_parallel_size=2,
                                  gloo_port=base_gloo_port + port_offset,
                                  hccl_port=base_hccl_port + port_offset,
                                  rank=rank, world_size=world_size)
    run_inference(llm_b, "Hello, my name is", max_tokens=10, rank=rank)

    # Cleanup Model B
    cleanup_model(llm_b, store_b, rank)
    port_offset += 10

    # Load Model A again (ports 29530, 50020)
    llm_a2, store_a2 = load_model(model_a, tensor_parallel_size=2,
                                   gloo_port=base_gloo_port + port_offset,
                                   hccl_port=base_hccl_port + port_offset,
                                   rank=rank, world_size=world_size)
    run_inference(llm_a2, "Hello, my name is", max_tokens=10, rank=rank)

    # Cleanup final model
    cleanup_model(llm_a2, store_a2, rank)

    # Print final NPU memory
    final_mem = get_npu_memory()
    print(f"\n[Rank {rank}] Final NPU memory: {final_mem['allocated_gb']:.2f}GB (free: {final_mem['free_gb']:.2f}GB)")

    if rank == 0:
        print("\n=== Test completed successfully! ===")


if __name__ == "__main__":
    main()
