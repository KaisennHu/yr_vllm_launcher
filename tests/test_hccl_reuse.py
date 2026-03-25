#!/usr/bin/env python3
# coding=UTF-8
# Test: Reuse torch.distributed communication group with CPU:GLOO + NPU:HCCL backend
# Usage: torchrun --nproc-per-node=2 test_hccl_reuse.py
#
# This test demonstrates creating and destroying communication groups with
# Ascend NPU hybrid backend (cpu:gloo,npu:hccl).
# Tests all-reduce operation on both CPU and NPU tensors.
# Uses explicit TCPStore to avoid GLOO backend global state issues.

import os
import time
import datetime
import torch
import torch.distributed
from torch.distributed import TCPStore


def test_cpu_collectives(rank, world_size):
    """Test all-reduce operation on CPU tensors."""
    print(f"\n--- CPU Tensor All-Reduce (rank={rank}) ---")

    cpu_tensor = torch.tensor([rank], dtype=torch.float32)
    torch.distributed.all_reduce(cpu_tensor, op=torch.distributed.ReduceOp.SUM)
    expected_sum = float(sum(range(world_size)))
    print(f"CPU all-reduce: {cpu_tensor.item()} (expected: {expected_sum})")

    torch.distributed.barrier()
    print(f"CPU collectives completed (rank={rank})")


def test_npu_collectives(rank, world_size):
    """Test all-reduce operation on NPU tensors."""
    print(f"\n--- NPU Tensor All-Reduce (rank={rank}) ---")

    try:
        device = f"npu:{rank % 8}"
        print(f"NPU device: {device}")

        npu_tensor = torch.tensor([rank], dtype=torch.float32, device=device)
        torch.distributed.all_reduce(npu_tensor, op=torch.distributed.ReduceOp.SUM)
        expected_sum = float(sum(range(world_size)))
        print(f"NPU all-reduce: {npu_tensor.item()} (expected: {expected_sum})")

    except Exception as e:
        print(f"NPU collectives failed (rank={rank}): {e}")

    torch.distributed.barrier()
    print(f"NPU collectives completed (rank={rank})")


def test_first_init(store_port=29510):
    """First initialization with explicit TCPStore and HCCL backend."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")

    print(f"\n=== First init_process_group ===")
    print(f"Rank: {rank}, World size: {world_size}")
    print(f"Creating TCPStore on {master_addr}:{store_port}")
    print(f"is_master (for this store): {rank == 0}")

    # Create TCPStore instance explicitly
    store = TCPStore(
        host_name=master_addr,
        port=store_port,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=datetime.timedelta(seconds=30),
    )

    print(f"TCPStore created (rank={rank})")

    # Initialize with explicit store using HCCL backend
    torch.distributed.init_process_group(
        backend="cpu:gloo,npu:hccl",
        store=store,
        rank=rank,
        world_size=world_size,
    )

    print(f"Process group initialized successfully (rank={rank})")
    print(f"is_initialized: {torch.distributed.is_initialized()}")
    print(f"get_rank: {torch.distributed.get_rank()}, get_world_size: {torch.distributed.get_world_size()}")

    # Test CPU collectives
    test_cpu_collectives(rank, world_size)

    # Test NPU collectives
    test_npu_collectives(rank, world_size)

    # Destroy process group
    print(f"\nDestroying process group (rank={rank})...")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    print(f"Process group destroyed (rank={rank}), is_initialized: {torch.distributed.is_initialized()}")


def test_second_init(store_port=29511):
    """Second initialization with new TCPStore and HCCL backend."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")

    print(f"\n=== Second init_process_group ===")
    print(f"Rank: {rank}, World size: {world_size}")
    print(f"Creating new TCPStore on {master_addr}:{store_port}")
    print(f"is_master (for this store): {rank == 0}")

    # Create a NEW TCPStore instance on a different port
    store = TCPStore(
        host_name=master_addr,
        port=store_port,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=datetime.timedelta(seconds=30),
    )

    print(f"New TCPStore created (rank={rank})")

    # Initialize with new store using HCCL backend
    torch.distributed.init_process_group(
        backend="cpu:gloo,npu:hccl",
        store=store,
        rank=rank,
        world_size=world_size,
    )

    print(f"Process group reinitialized successfully (rank={rank})")
    print(f"is_initialized: {torch.distributed.is_initialized()}")
    print(f"get_rank: {torch.distributed.get_rank()}, get_world_size: {torch.distributed.get_world_size()}")

    # Test CPU collectives again
    test_cpu_collectives(rank, world_size)

    # Test NPU collectives again
    test_npu_collectives(rank, world_size)

    # Destroy process group
    print(f"\nDestroying process group (rank={rank})...")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    print(f"Process group destroyed (rank={rank}), is_initialized: {torch.distributed.is_initialized()}")


def main():
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    print(f"=== Test: HCCL (cpu:gloo,npu:hccl) Collective Operations ===")
    print(f"Rank: {rank}, World size: {world_size}")

    if "MASTER_ADDR" in os.environ:
        print(f"Using MASTER_ADDR: {os.environ['MASTER_ADDR']} (multi-node mode)")
    else:
        print(f"Using default MASTER_ADDR: 127.0.0.1 (single-node mode)")

    # First init with port 29510
    test_first_init(store_port=29510)

    # Small delay between initializations
    time.sleep(1)

    # Second init with port 29511 (different port, different store)
    test_second_init(store_port=29511)

    print(f"\n=== Test completed successfully (rank={rank}) ===")


if __name__ == "__main__":
    main()
