#!/usr/bin/env python3
# coding=UTF-8
# Test: Model switching with torch torchrun + external_launcher
# Usage: torchrun --nproc-per-node=2 torch_run_test.py

import os
import time
import datetime
import torch
import torch_npu
import torch.distributed
import gc
import sys  # For sys.modules cleanup

from vllm import LLM, SamplingParams
from torch.distributed import TCPStore
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

# Import for CaMemAllocator cleanup (vllm-ascend specific)
try:
    from vllm_ascend.device_allocator.camem import CaMemAllocator
except ImportError:
    # Fallback if vllm_ascend is not available
    CaMemAllocator = None


def run_inference(llm, prompts, sampling_params, label):
    """Run inference and print results."""
    print(f"\n=== {label} ===")
    print(f"NPU Memory: {torch.npu.memory_allocated() / 1024**3:.2f} GB allocated")

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    print(f"Done {label}")


def cleanup_llm_instance(llm, change_hccl_port=False, new_hccl_port=50001):
    """
    Comprehensive cleanup of LLM instance and release NPU memory.

    Args:
        llm: LLM instance to cleanup
        change_hccl_port: Whether to change HCCL_IF_BASE_PORT
        new_hccl_port: New HCCL port value if change_hccl_port is True
    """
    rank = int(os.getenv("RANK", "0"))

    # 1. Sleep to LLM instance to release GPU memory
    if llm is not None:
        print(f"[Rank {rank}] Putting LLM to sleep...")
        llm.sleep(level=2)

    # 2. Delete LLM instance
    print(f"[Rank {rank}] Deleting LLM instance...")
    if llm is not None:
        del llm
    llm = None

    # 3. Clean up CaMemAllocator singleton (CRITICAL for NPU memory release)
    print(f"[Rank {rank}] Cleaning up CaMemAllocator singleton...")
    if CaMemAllocator is not None:
        try:
            # Get singleton instance
            allocator = CaMemAllocator.get_instance()

            # Clear all allocation data
            if hasattr(allocator, 'pointer_to_data'):
                before_count = len(allocator.pointer_to_data)
                allocator.pointer_to_data.clear()
                print(f"[Rank {rank}] Cleared {before_count} allocation data entries")

            # Clear all allocator and pool references
            if hasattr(allocator, 'allocator_and_pools'):
                before_count = len(allocator.allocator_and_pools)
                allocator.allocator_and_pools.clear()
                print(f"[Rank {rank}] Cleared {before_count} allocator/pool references")

            # Clear singleton reference itself
            CaMemAllocator.instance = None
            print(f"[Rank {rank}] CaMemAllocator singleton cleared")
        except Exception as e:
            print(f"[Rank {rank}] Error cleaning CaMemAllocator: {e}")

    # 4. Force garbage collection before distributed cleanup
    print(f"[Rank {rank}] Running garbage collection...")
    gc.collect()

    # 5. Clean up vLLM model parallel state (MUST be before destroy_process_group)
    print(f"[Rank {rank}] Destroying model parallel state...")
    try:
        from vllm_ascend.distributed.parallel_state import destroy_ascend_model_parallel
        destroy_ascend_model_parallel()
    except Exception as e:
        print(f"[Rank {rank}] Error destroying ascend model parallel: {e}")

    # 6. Destroy vLLM distributed environment (MUST be before destroy_process_group)
    print(f"[Rank {rank}] Destroying distributed environment...")
    try:
        destroy_distributed_environment()
    except Exception as e:
        print(f"[Rank {rank}] Error destroying distributed environment: {e}")

    # 7. Destroy torch.distributed process group
    print(f"[Rank {rank}] Destroying torch.distributed process group...")
    if torch.distributed.is_initialized():
        try:
            torch.distributed.barrier()
        except:
            pass
        torch.distributed.destroy_process_group()

    # 8. Clean up torch_n.dpu device state
    print(f"[Rank {rank}] Cleaning up torch_npu device state...")
    try:
        if torch.npu.is_available():
            torch.npu.synchronize()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
    except Exception as e:
        print(f"[Rank {rank}] Warning: error cleaning torch_npu state: {e}")

    # 9. Force garbage collection after PG destroy (to trigger PyHcclCommunicator cleanup)
    print(f"[Rank {rank}] Running garbage collection after PG destroy...")
    gc.collect()

    # 10. Clean up torch.ops.vllm related global state (invasive cleanup)
    print(f"[Rank {rank}] Cleaning up torch.ops.vllm state...")
    try:
        # Clean up torch.distributed internal global variables
        import torch.distributed as dist_module
        if hasattr(dist_module, '_group_map'):
            _group_map = dist_module._group_map
            if _group_map:
                print(f"[Rank {rank}] Clearing torch.distributed._group_map ({len(_group_map)} groups)")
                _group_map.clear()
    except Exception as e:
        print(f"[Rank {rank}] Warning: error cleaning torch.distributed state: {e}")

    # 11. Delete vllm_ascend.ops.rotary_embedding module from sys.modules (CRITICAL FIX)
    print(f"[Rank {rank}] Removing rotary_embedding module from sys.modules...")
    try:
        # Delete main module
        if 'vllm_ascend.ops.rotary_embedding' in sys.modules:
            del sys.modules['vllm_ascend.ops.rotary_embedding']
            print(f"[Rank {rank}] Removed vllm_ascend.ops.rotary_embedding from sys.modules")

        # Delete possible triton rope module (if used)
        if 'vllm_ascend.ops.triton.rope' in sys.modules:
            del sys.modules['vllm_ascend.ops.triton.rope']
            print(f"[Rank {rank}] Removed vllm_ascend.ops.triton.rope from sys.modules")

        # Delete parent package to force re-import
        if 'vllm_ascend.ops' in sys.modules:
            del sys.modules['vllm_ascend.ops']
            print(f"[Rank {rank}] Removed vllm_ascend.ops from sys.modules")

    except Exception as e:
        print(f"[Rank {rank}] Warning: error removing rotary_embedding module: {e}")

    # 12. Extra delay to ensure HCCL C-level cleanup completes
    print(f"[Rank {rank}] Waiting for HCCL cleanup...")
    time.sleep(2)

    # 13. Another garbage collection pass
    gc.collect()

    # 14. Change HCCL_IF_BASE_PORT if requested
    if change_hccl_port:
        old_port = os.environ.get('HCCL_IF_BASE_PORT')
        print(f"[Rank {rank}] Changing HCCL_IF_BASE_PORT from {old_port} to {new_hccl_port}")
        os.environ["HCCL_IF_BASE_PORT"] = str(new_hccl_port)

    # 15. Final garbage collection (two passes for cyclical references)
    gc.collect()
    gc.collect()

    print(f"[Rank {rank}] Cleanup complete. NPU Memory: {torch.npu.memory_allocated() / 1024**3:.2f} GB")


def switch_model(
    old_llm,
    new_model_path,
    new_port,
    tensor_parallel_size=2,
    trust_remote_code=True,
):
    """
    Switch to a new model by destroying old LLM and distributed environment,
    then reinitializing with a new TCPStore on a different port.

    Args:
        old_llm: Existing LLM instance to delete
        new_model_path: Path to new model
        new_port: Port for new TCPStore (must be different from first initialization)
        tensor_parallel_size: Tensor parallel size (TP=2)
        trust_remote_code: Whether to trust remote code

    Returns:
        New LLM instance

    Note:
        - We create a NEW TCPStore instance on a different port to avoid GLOO
          backend global state issues.
        - We clear CaMemAllocator singleton to release NPU memory.
        - We change HCCL_IF_BASE_PORT to avoid HCCL port conflicts.
    """
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    print(f"\n=== Switching to model: {new_model_path} ===")
    print(f"Rank: {rank}, World size: {world_size}")

    # Comprehensive cleanup of old LLM instance
    # Use new HCCL port based on new_port (e.g., 29511 -> 50001)
    new_hccl_port = 50000 + (new_port % 1000)  # Map TCPStore port to HCCL port
    cleanup_llm_instance(old_llm, change_hccl_port=True, new_hccl_port=new_hccl_port)

    # Small delay to ensure cleanup completes
    time.sleep(2)

    # Reinitialize torch.distributed process group with new TCPStore
    print("Reinitializing torch.distributed process group with new TCPStore...")
    print(f"Creating new TCPStore on port {new_port}...")
    store = TCPStore(
        host_name="127.0.0.1",
        port=new_port,
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

    # Create new LLM instance
    print(f"Creating new LLM instance with model: {new_model_path}")
    new_llm = LLM(
        model=new_model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
        distributed_executor_backend="external_launcher",
        enable_sleep_mode=True
    )

    print(f"NPU Memory after new LLM: {torch.npu.memory_allocated() / 1024**3:.2f} GB")
    print("Model switched successfully!")

    return new_llm


def main():
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    tensor_parallel_size = 2  # TP=2

    print(f"Rank: {rank}, World size: {world_size}, TP: {tensor_parallel_size}")

    # Model paths
    model_a = "/workspace/models/qwen2.5_7B_Instruct"
    model_b = "/workspace/models/Qwen2.5-VL-7B-Instruct"

    prompts = ["Hello, my name is"]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=10, logprobs=1)

    # Set environment variables for vLLM and HCCL
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"
    # Always set HCCL_IF_BASE_PORT to ensure consistent initial state
    os.environ["HCCL_IF_BASE_PORT"] = "50000"

    # Initialize torch.distributed process group with explicit TCPStore
    print(f"\n=== Initializing torch.distributed ===")
    print(f"Creating TCPStore on port 29510...")
    store = TCPStore(
        host_name="127.0.0.1",
        port=29510,
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
    print("torch.distributed initialized successfully")

    # Create first LLM with model A (TP=2)
    print(f"\n=== Initializing Model A: {model_a} ===")

    llm = LLM(
        model=model_a,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        distributed_executor_backend="external_launcher",
        enable_sleep_mode=True
    )

    # Run inference with Model A
    run_inference(llm, prompts, sampling_params, "Inference with Model A")

    # Switch to Model B (use port 29511 for new TCPStore)
    new_port = 29511
    del store
    llm = switch_model(llm, model_b, new_port, tensor_parallel_size)

    # Run inference with Model B
    run_inference(llm, prompts, sampling_params, "Inference with Model B")

    print("\n=== Test completed successfully! ===")


if __name__ == "__main__":
    main()
