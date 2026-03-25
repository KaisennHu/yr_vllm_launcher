#!/usr/bin/env python3
# coding=UTF-8
# Compare Results - aggregate and compare all test results

import json
import os
import sys
from typing import Dict, Optional


def load_result(filename: str) -> Optional[Dict]:
    """Load JSON result file from parent directory."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), filename)
    if not os.path.exists(path):
        print(f"Warning: {filename} not found", file=sys.stderr)
        return None
    with open(path, "r") as f:
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
    print("Model Switching - Performance Comparison")
    print("=" * 80)
    print()

    # Load all results
    cold_start = load_result("results_cold_start.json")
    model_switch = load_result("results_sleep_mode.json")

    if not cold_start and not model_switch:
        print("No test results found. Please run tests first.")
        print()
        print("Run:")
        print("  python3 tests/cold_start.py")
        print("  python3 tests/test_sleep_mode.py")
        return

    # Display configuration
    if model_switch:
        print(f"Model A: {model_switch['model_a']}")
        print(f"Model B: {model_switch['model_b']}")
        print(f"TP Size: {model_switch['tp_size']}")
        print()

    # Display init latency (actor creation) comparison
    print("=" * 80)
    print("Actor Creation (Init) Latency")
    print("=" * 80)
    print()

    if cold_start:
        init_latency = cold_start.get('init_latency_ms', 0)
        print(f"Cold Start (New Actor): {init_latency:.2f} ms")
        print()

    if model_switch:
        switch_init = model_switch.get('init_latency_ms', 0)
        print(f"Model Switch (New Actor): {switch_init:.2f} ms")
        print()

    # Display model switch latency
    print("=" * 80)
    print("Model Switch Latency (A -> B)")
    print("=" * 80)
    print()

    if model_switch:
        switch_latency = model_switch.get('switch_latency_ms', 0)
        print(f"Model Switch: {switch_latency:.2f} ms")
        print()

    # Display second switch (B -> A)
    if model_switch and 'second_switch_latency_ms' in model_switch:
        print("=" * 80)
        print("Second Model Switch Latency (B -> A)")
        print("=" * 80)
        print()

        second_switch = model_switch.get('second_switch_latency_ms', 0)
        print(f"Model Switch: {second_switch:.2f} ms")
        print()

        # Calculate variance between switches
        first_switch = model_switch.get('switch_latency_ms', 0)
        if first_switch > 0:
            diff_pct = abs(second_switch - first_switch) / first_switch * 100
            print(f"Variance between switches: {diff_pct:.1f}%")
            print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()

    if cold_start:
        init_time = cold_start.get('init_latency_ms', 0)
        print(f"Cold Start (new actor): {init_time:.2f} ms")
        print(f"  - Creates new actor process")
        print(f"  - Imports torch, torch_npu, torch.distributed (in actor process)")
        print(f"  - Initializes vLLM executor")
        print(f"  - Loads model weights (sharded across TP ranks)")
        print()

    if model_switch:
        switch_time = model_switch.get('switch_latency_ms', 0)
        print(f"Model switch: {switch_time:.2f} ms")
        print(f"  - Reuses actor process")
        print(f"  - Deletes old LLM instance")
        print(f"  - Cleans up vLLM internal state (destroy_model_parallel)")
        print(f"  - Creates new LLM instance with new model")
        print(f"  - Each TP rank loads its weight shard independently")
        print()

        if 'second_switch_latency_ms' in model_switch:
            second_switch = model_switch.get('second_switch_latency_ms', 0)
            print(f"Second switch time: {second_switch:.2f} ms")
            print()

            # Compare switch times
            first_switch = model_switch.get('switch_latency_ms', 0)
            if first_switch > 0:
                avg_switch = (first_switch + second_switch) / 2
                print(f"Average switch time: {avg_switch:.2f} ms")
                print()

    print("=" * 80)
    print("Model Switching Implementation Details")
    print("=" * 80)
    print()
    print("Model switching uses the following flow:")
    print("  1. Delete old LLM instance")
    print("  2. Call destroy_model_parallel() to clean up vLLM state")
    print("  3. Call gc.collect() and torch.npu.empty_cache()")
    print("  4. Create new LLM instance with new model path")
    print()
    print("This reuses the actor process, avoiding:")
    print("  - Actor process recreation overhead")
    print("  - torch/torch_npu import overhead")
    print()


if __name__ == "__main__":
    main()
