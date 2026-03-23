#!/usr/bin/env python3
# coding=UTF-8
# Yuanrong + vLLM Integration: Example Usage
#
# Simple example of running vLLM inference with 2 independent Yuanrong Actors.

import yr
import vllm_actor


def main():
    """Run vLLM inference with Yuanrong using 2 independent Actors."""
    # Initialize Yuanrong with DEBUG log level
    yr.init(yr.Config(log_level="DEBUG"))
    print("Yuanrong initialized with DEBUG logging")

    # Configure options for both instances
    opts = yr.InvokeOptions()
    opts.cpu = 2000
    opts.memory = 4096
    opts.custom_resources = {"NPU/.+/count": 1}

    model_path = "/workspace/models/qwen2.5_7B_Instruct"
    tp = 2
    print(f"Creating 2 vLLM Actors (TP={tp})...")

    # Create 2 independent Actors
    actor0 = vllm_actor.VLLMEngine.options(opts).invoke(model_path, tp, 0, 2)
    actor1 = vllm_actor.VLLMEngine.options(opts).invoke(model_path, tp, 1, 2)
    print("2 Actors created")

    # Run inference on both actors (SPMD: all ranks participate)
    prompt = "Hello, my name is"
    print(f"\nGenerating for: '{prompt}' (TP={tp}, both actors participating)")
    obj_ref0 = actor0.generate.invoke(prompt, max_tokens=10)
    obj_ref1 = actor1.generate.invoke(prompt, max_tokens=10)

    # Wait for both actors to complete
    yr.wait([obj_ref0, obj_ref1])

    # Only rank 0 returns the actual result
    result0 = yr.get(obj_ref0)
    print(f"Result from rank 0: {result0}")

    # Cleanup
    actor0.terminate()
    actor1.terminate()
    yr.finalize()
    print("Yuanrong finalized")


if __name__ == "__main__":
    main()
