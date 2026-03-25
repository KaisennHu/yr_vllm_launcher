#!/usr/bin/env python3
# coding=UTF-8
# Test: Reuse torch.distributed gloo communication group
# Usage: torchrun --nproc-per-node=2 test_gloo_reuse.py
#
# SOLUTION: Manually create TCPStore instances for each initialization.
# This avoids global state issues by explicitly managing store lifecycle.

import os
import time
import datetime
import torch
import torch.distributed
from torch.distributed import TCPStore


def test_first_init(store_port=29510):
    """First initialization with explicit TCPStore."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = "127.0.0.1"

    print(f"\n=== First init_process_group ===")
    print(f"Rank: {rank}, World size: {world_size}")
    print(f"Creating TCPStore on {master_addr}:{store_port}")
    print(f"is_master (for this store): {rank == 0}")

    # Create TCPStore instance explicitly
    # is_master=True only for rank 0 (server), others are clients
    store = TCPStore(
        host_name=master_addr,
        port=store_port,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=datetime.timedelta(seconds=30),
    )

    print(f"TCPStore created (rank={rank})")

    # Initialize with explicit store
    torch.distributed.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=world_size,
    )

    print(f"Process group initialized successfully (rank={rank})")
    print(f"is_initialized: {torch.distributed.is_initialized()}")
    print(f"get_rank: {torch.distributed.get_rank()}, get_world_size: {torch.distributed.get_world_size()}")

    # Simple all_reduce test
    tensor = torch.tensor([rank], dtype=torch.float32)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    print(f"All-reduce result (rank={rank}): {tensor.item()} (expected: {world_size * (world_size - 1) / 2})")

    # Barrier to synchronize
    torch.distributed.barrier()
    print(f"Barrier passed (rank={rank})")

    # Destroy the process group
    print(f"Destroying process group (rank={rank})...")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    print(f"Process group destroyed (rank={rank}), is_initialized: {torch.distributed.is_initialized()}")


def test_second_init(store_port=29511):
    """Second initialization with a new TCPStore on different port."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = "127.0.0.1"

    print(f"\n=== Second init_process_group ===")
    print(f"Rank: {rank}, World size: {world_size}")
    print(f"Creating new TCPStore on {master_addr}:{store_port}")
    print(f"is_master (for this store): {rank == 0}")

    # Create a NEW TCPStore instance on a different port
    # is_master=True only for rank 0 (server), others are clients
    store = TCPStore(
        host_name=master_addr,
        port=store_port,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=datetime.timedelta(seconds=30),
    )

    print(f"New TCPStore created (rank={rank})")

    # Initialize with new store
    torch.distributed.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=world_size,
    )

    print(f"Process group reinitialized successfully (rank={rank})")
    print(f"is_initialized: {torch.distributed.is_initialized()}")
    print(f"get_rank: {torch.distributed.get_rank()}, get_world_size: {torch.distributed.get_world_size()}")

    # Simple all_reduce test again
    tensor = torch.tensor([rank * 10], dtype=torch.float32)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    print(f"All-reduce result (rank={rank}): {tensor.item()} (expected: {10 * world_size * (world_size - 1) / 2})")

    # Barrier to synchronize
    torch.distributed.barrier()
    print(f"Barrier passed (rank={rank})")


def main():
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    print(f"=== Test: GLOO with Explicit TCPStore ===")
    print(f"Rank: {rank}, World size: {world_size}")

    # First init with port 29510
    test_first_init(store_port=29510)

    # Small delay between initializations
    time.sleep(1)

    # Second init with port 29511 (different port, different store)
    test_second_init(store_port=29511)

    print(f"\n=== Test completed successfully (rank={rank}) ===")


if __name__ == "__main__":
    main()
