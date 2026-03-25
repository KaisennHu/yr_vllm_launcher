#!/usr/bin/env python3
# coding=UTF-8
# Test: Cold Start - measures full cold startup including actor creation

import os
import sys
# Add parent directory to Python path for vllm_actor import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    model = "/workspace/models/qwen2.5_7B_Instruct"
    tp_size = 2

    yr.init(yr.Config(log_level="INFO"))
    _logger = yr.log.get_logger()

    print(f"=== Cold Start Test (TP={tp_size}) ===")
    print(f"Model: {model}")
    print()

    # Configure options for all instances
    opts = yr.InvokeOptions()
    opts.cpu = 2000
    opts.memory = 4096
    opts.custom_resources = {"NPU/.+/count": 1}

    # Measure full actor creation latency
    print("Creating Actors...")
    init_start = time.time()
    actors = []
    for rank in range(tp_size):
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(tp_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if "HCCL_IF_BASE_PORT" not in os.environ:
            os.environ["HCCL_IF_BASE_PORT"] = "50000"

        actor = vllm_actor.VLLMEngine.options(opts).invoke(model, tp_size, rank, tp_size)
        actors.append(actor)

    # Wait for all actors to be ready
    obj_refs = []
    for actor in actors:
        obj_ref = actor.ready.invoke()
        obj_refs.append(obj_ref)
    yr.wait(obj_refs)

    init_latency_ms = (time.time() - init_start) * 1000
    print(f"  Init latency: {init_latency_ms:.2f} ms")
    print()

    # Inference
    prompt = "Hello, my name is"
    print(f"Inference: '{prompt}'")
    inference_start = time.time()

    obj_refs = []
    for actor in actors:
        obj_ref = actor.generate.invoke(prompt, max_tokens=10)
        obj_refs.append(obj_ref)

    yr.wait(obj_refs)
    inference_latency_ms = (time.time() - inference_start) * 1000
    print(f"  Inference: {inference_latency_ms:.2f} ms")
    print()

    # Cleanup
    for actor in actors:
        actor.terminate()
    yr.finalize()

    # Output result
    output = {
        "test": "cold_start",
        "model": model,
        "tp_size": tp_size,
        "init_latency_ms": init_latency_ms,
        "inference_latency_ms": inference_latency_ms,
    }

    print("=== Results ===")
    print(json.dumps(output, indent=2))

    with open("../results_cold_start.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to results_cold_start.json")


if __name__ == "__main__":
    main()
