#!/usr/bin/env python3
# coding=UTF-8
# Yuanrong + vLLM Integration: Client Code
#
# This module provides client-side functions to launch and interact with
# distributed vLLM clusters using Yuanrong's Function Group API.

from typing import Optional

# Yuanrong imports
import yr
from yr.fcc import create_function_group, FunctionGroupOptions
from yr.config import GroupOptions

import vllm_actor


class VLLMCluster:
    """
    Client-side wrapper for distributed vLLM cluster running on Yuanrong.

    Manages a Yuanrong Function Group of VLLMEngine actors and provides
    a convenient interface for distributed inference.

    Usage:
        cluster = VLLMCluster(
            model_path="meta-llama/Llama-2-7b",
            tensor_parallel_size=4,
            gpus_per_actor=1,
        )
        cluster.launch()

        # Inference
        prompts = ["Hello, my name is", "The capital of France is"]
        outputs = cluster.generate(prompts, max_tokens=50)
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        gpus_per_actor: int = 1,
        nnodes: Optional[int] = None,
        trust_remote_code: bool = True,
        max_model_len: int = 8192,
        seed: int = 1,
        master_port: int = 29500,
    ):
        """
        Initialize VLLMCluster client.

        Args:
            model_path: Path to HuggingFace model or local model directory
            tensor_parallel_size: Tensor parallelism degree
            pipeline_parallel_size: Pipeline parallelism degree (default: 1)
            gpus_per_actor: Number of GPUs per Actor (default: 1)
            nnodes: Number of physical nodes (auto-calculated if None)
            trust_remote_code: Whether to trust remote code when loading model
            max_model_len: Maximum model length (context window)
            seed: Random seed for deterministic inference
            master_port: Port for NCCL communication

        Raises:
            ValueError: If configuration is invalid
        """
        self._model_path = model_path
        self._tensor_parallel_size = tensor_parallel_size
        self._pipeline_parallel_size = pipeline_parallel_size
        self._gpus_per_actor = gpus_per_actor
        self._master_port = master_port

        # Calculate world size and number of nodes
        self._world_size = tensor_parallel_size * pipeline_parallel_size

        if nnodes is None:
            # Auto-calculate: nnodes = ceil(world_size / gpus_per_actor)
            import math
            nnodes = math.ceil(self._world_size / gpus_per_actor)
        self._nnodes = nnodes

        # Validate configuration
        if self._world_size <= 0:
            raise ValueError(f"world_size must be > 0, got {self._world_size}")
        if self._nnodes <= 0:
            raise ValueError(f"nnodes must be > 0, got {self._nnodes}")
        if gpus_per_actor <= 0:
            raise ValueError(f"gpus_per_actor must be > 0, got {gpus_per_actor}")

        # Calculate actors per node
        self._actors_per_node = self._world_size // self._nnodes
        if self._world_size % self._nnodes != 0:
            raise ValueError(
                f"world_size ({self._world_size}) must be evenly divisible by "
                f"nnodes ({self._nnodes})"
            )

        # Store other parameters
        self._trust_remote_code = trust_remote_code
        self._max_model_len = max_model_len
        self._seed = seed

        # Function Group reference (set after launch)
        self._engine_actors = None
        self._rank0_actor = None

    def launch(self):
        """
        Launch the distributed vLLM cluster.

        Creates a Yuanrong Function Group with the configured number of
        VLLMEngine actors.

        Raises:
            RuntimeError: If launch fails
        """
        yr.log.get_logger().info(
            f"Launching vLLM cluster: world_size={self._world_size}, "
            f"nnodes={self._nnodes}, actors_per_node={self._actors_per_node}"
        )

        # Configure Function Group options
        group_opts = FunctionGroupOptions(
            # CPU and memory resources per actor
            cpu=2000,  # 2 CPU cores per actor
            memory=4096,  # 4GB memory per actor
            # Custom resources (GPU/NPU)
            resources={
                f"NPU/Ascend910B4/count": self._gpus_per_actor,
            },
            # Scheduling affinity
            scheduling_affinity_each_bundle_size=self._actors_per_node,
        )

        # Configure Group options for lifecycle management
        group_opts = GroupOptions(
            timeout=-1,  # Wait indefinitely for resources
            same_lifecycle=True,  # Actors are created/destroyed together
            strategy="",  # Default placement strategy
        )

        # Create Function Group
        self._engine_actors = create_function_group(
            func=vllm_actor.VLLMEngine,
            args=(
                self._model_path,
                self._tensor_parallel_size,
                self._pipeline_parallel_size,
                self._trust_remote_code,
                self._max_model_len,
                self._seed,
                self._master_port,
            ),
            group_size=self._world_size,
            options=group_opts,
        )

        # Get rank 0 actor as the main interface
        self._rank0_actor = self._engine_actors[0]

        yr.log.get_logger().info("vLLM cluster launched successfully")

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
    ):
        """
        Execute distributed inference.

        Args:
            prompts: List of text prompts to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter (optional)

        Returns:
            vLLM outputs from rank 0. The exact type depends on
            vLLM's generate() return type.

        Raises:
            RuntimeError: If cluster is not launched or inference fails
        """
        if self._rank0_actor is None:
            raise RuntimeError(
                "Cluster not launched. Call launch() before generate()."
            )

        yr.log.get_logger().info(f"Generating for {len(prompts)} prompts")

        # Call generate on rank 0 actor
        outputs = self._rank0_actor.generate(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        yr.log.get_logger().info("Generation completed")
        return outputs

    def encode(self, prompts: list[str]):
        """
        Encode prompts to token IDs using rank 0 actor.

        Args:
            prompts: List of text prompts to encode

        Returns:
            Encoded token IDs from rank 0 actor.

        Raises:
            RuntimeError: If cluster is not launched or encoding fails
        """
        if self._rank0_actor is None:
            raise RuntimeError(
                "Cluster not launched. Call launch() before encode()."
            )

        return self._rank0_actor.encode(prompts=prompts)

    def get_config(self) -> dict:
        """
        Get cluster configuration.

        Returns:
            Dictionary containing cluster configuration.
        """
        return {
            "model_path": self._model_path,
            "tensor_parallel_size": self._tensor_parallel_size,
            "pipeline_parallel_size": self._pipeline_parallel_size,
            "world_size": self._world_size,
            "nnodes": self._nnodes,
            "actors_per_node": self._actors_per_node,
            "gpus_per_actor": self._gpus_per_actor,
            "max_model_len": self._max_model_len,
            "seed": self._seed,
        }


def launch_vllm_cluster(
    model_path: str,
    tensor_parallel_size: int = 4,
    pipeline_parallel_size: int = 2,
    gpus_per_actor: int = 1,
    nnodes: Optional[int] = None,
    trust_remote_code: bool = True,
    max_model_len: int = 8192,
    seed: int = 1,
    master_port: int = 29500,
) -> VLLMCluster:
    """
    Convenience function to launch a vLLM cluster.

    Args:
        model_path: Path to HuggingFace model or local model directory
        tensor_parallel_size: Tensor parallelism degree
        pipeline_parallel_size: Pipeline parallelism degree
        gpus_per_actor: Number of GPUs per actor
        nnodes: Number of physical nodes (auto-calculated if None)
        trust_remote_code: Whether to trust remote code
        max_model_len: Maximum model length
        seed: Random seed for deterministic inference
        master_port: Port for NCCL communication

    Returns:
        VLLMCluster instance with the cluster already launched.

    Example:
        >>> import yr
        >>> from client import launch_vllm_cluster
        >>>
        >>> yr.init(yr.Config())
        >>>
        >>> cluster = launch_vllm_cluster(
        ...     model_path="meta-llama/Llama-2-7b",
        ...     tensor_parallel_size=4,
        ... )
        >>>
        >>> outputs = cluster.generate(["Hello, world"], max_tokens=50)
        >>> print(outputs)
        >>>
        >>> yr.finalize()
    """
    cluster = VLLMCluster(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        gpus_per_actor=gpus_per_actor,
        nnodes=nnodes,
        trust_remote_code=trust_remote_code,
        max_model_len=max_model_len,
        seed=seed,
        master_port=master_port,
    )
    cluster.launch()
    return cluster
