#!/usr/bin/env python3
# coding=UTF-8
# Yuanrong + vLLM Integration: Launcher Script
#
# Command-line tool to launch and interact with distributed vLLM clusters
# using Yuanrong's Function Group API.

import argparse
import sys

import yr
from yr.config import Config
from client import VLLMCluster, launch_vllm_cluster


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch distributed vLLM inference cluster using Yuanrong",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-node, TP=2 inference
  python launcher.py --model-path meta-llama/Llama-2-7b --tp 2 --pp 1

  # Multi-node, TP=4, PP=2 inference (4 actors per node)
  python launcher.py --model-path meta-llama/Llama-2-7b --tp 4 --pp 2 --gpus-per-actor 1 --nnodes 4

  # Interactive mode (launch and wait for prompts)
  python launcher.py --model-path meta-llama/Llama-2-7b --tp 2 --interactive

  # Run a single inference request
  python launcher.py --model-path meta-llama/Llama-2-7b --tp 2 --prompt "Hello, world"
        """
    )

    # Required arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to HuggingFace model or local model directory",
    )

    # Parallelism configuration
    parser.add_argument(
        "--tp",
        "--tensor-parallel-size",
        type=int,
        default=1,
        dest="tensor_parallel_size",
        help="Tensor parallelism degree (default: 1)",
    )
    parser.add_argument(
        "--pp",
        "--pipeline-parallel-size",
        type=int,
        default=1,
        dest="pipeline_parallel_size",
        help="Pipeline parallelism degree (default: 1)",
    )
    parser.add_argument(
        "--gpus-per-actor",
        type=int,
        default=1,
        help="Number of GPUs per actor (default: 1)",
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=None,
        help="Number of physical nodes (auto-calculated if not specified)",
    )

    # vLLM configuration
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model length / context window (default: 8192)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for deterministic inference (default: 1)",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_false",
        dest="trust_remote_code",
        help="Do not trust remote code when loading model",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Port for NCCL communication (default: 29500)",
    )

    # Inference configuration
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate from (non-interactive mode)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        nargs="*",
        help="Multiple prompts to generate from (non-interactive mode)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        dest="top_p",
        help="Top-p (nucleus) sampling parameter (default: 0.95)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        dest="top_k",
        help="Top-k sampling parameter (default: None)",
    )

    # Mode
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch cluster and enter interactive mode for multiple inferences",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Only print cluster configuration without launching",
    )

    # Yuanrong configuration
    parser.add_argument(
        "--yr-server-address",
        type=str,
        default="",
        help="Yuanrong server address (from YR_SERVER_ADDRESS env var if not set)",
    )
    parser.add_argument(
        "--yr-ds-address",
        type=str,
        default="",
        help="Yuanrong DataSystem address (from YR_DS_ADDRESS env var if not set)",
    )
    parser.add_argument(
        "--yr-in-cluster",
        action="store_true",
        default=True,
        dest="in_cluster",
        help="Run in cluster mode (default: True)",
    )

    return parser.parse_args()


def print_config(args, cluster: VLLMCluster):
    """Print cluster configuration."""
    config = cluster.get_config()
    print("=" * 60)
    print("vLLM Cluster Configuration")
    print("=" * 60)
    print(f"  Model Path:              {config['model_path']}")
    print(f"  Tensor Parallel Size:     {config['tensor_parallel_size']}")
    print(f"  Pipeline Parallel Size:   {config['pipeline_parallel_size']}")
    print(f"  World Size:              {config['world_size']}")
    print(f"  Nodes:                   {cluster._nnodes}")
    print(f"  Actors per Node:         {cluster._actors_per_node}")
    print(f"  GPUs per Actor:          {cluster._gpus_per_actor}")
    print(f"  Max Model Length:         {args.max_model_len}")
    print(f"  Seed:                    {args.seed}")
    print(f"  Master Port:              {args.master_port}")
    print("=" * 60)


def interactive_mode(cluster: VLLMCluster, args):
    """
    Run interactive inference mode.

    Repeatedly prompts user for input and performs inference.
    """
    print("\n" + "=" * 60)
    print("Interactive Mode")
    print("=" * 60)
    print("Enter prompts to generate (Ctrl+C or 'quit' to exit)\n")

    try:
        while True:
            # Prompt user for input
            user_input = input("Prompt: ").strip()

            # Check for exit conditions
            if user_input.lower() in ("q", "quit", "exit"):
                print("Exiting...")
                break

            if not user_input:
                continue

            # Perform inference
            print(f"\nGenerating with max_tokens={args.max_tokens}...")
            outputs = cluster.generate(
                prompts=[user_input],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )

            # Print results
            print("\nResults:")
            for i, output in enumerate(outputs):
                print(f"  [{i}] {output.prompt}")
                print(f"      Generated: {output.outputs[0].text}")
            print()

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting...")


def single_inference_mode(cluster: VLLMCluster, prompts: list[str], args):
    """
    Run single inference with provided prompts.

    Args:
        cluster: VLLMCluster instance
        prompts: List of prompts to generate from
        args: Parsed command-line arguments
    """
    print(f"\nGenerating for {len(prompts)} prompt(s)...")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Top-k: {args.top_k}")

    outputs = cluster.generate(
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    print("\nResults:")
    for i, output in enumerate(outputs):
        print(f"\n[{i}] {output.prompt}")
        print(f"    Generated: {output.outputs[0].text}")


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize Yuanrong
    yr_config = Config(
        server_address=args.yr_server_address,
        ds_address=args.yr_ds_address,
        in_cluster=args.yr_in_cluster,
    )
    print(f"Initializing Yuanrong...")
    yr.init(yr_config)
    print("Yuanrong initialized successfully")

    try:
        # Create cluster
        print(f"\nCreating vLLM cluster with model: {args.model_path}")
        cluster = VLLMCluster(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            gpus_per_actor=args.gpus_per_actor,
            nnodes=args.nnodes,
            trust_remote_code=args.trust_remote_code,
            max_model_len=args.max_model_len,
            seed=args.seed,
            master_port=args.master_port,
        )

        # Print configuration
        print_config(args, cluster)

        # Config-only mode
        if args.config_only:
            print("\nConfig-only mode. Exiting without launching cluster.")
            return 0

        # Launch cluster
        print("\nLaunching cluster...")
        cluster.launch()
        print("Cluster launched successfully")

        # Determine prompts to use
        if args.interactive:
            # Interactive mode
            interactive_mode(cluster, args)
        elif args.prompts:
            # Use provided prompts
            prompts = list(args.prompts)
        elif args.prompt:
            # Use single prompt
            prompts = [args.prompt]
        else:
            # No prompts provided, exit
            print("\nNo prompts provided. Use --interactive or --prompt/--prompts arguments.")
            return 0

        # Run inference
        single_inference_mode(cluster, prompts, args)

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Finalize Yuanrong
        print("\nFinalizing Yuanrong...")
        yr.finalize()
        print("Yuanrong finalized")


if __name__ == "__main__":
    sys.exit(main())
