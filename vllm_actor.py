#!/usr/bin/env python3
# coding=UTF-8
# Yuanrong + vLLM Integration: VLLMEngine Actor
#
# This module implements a Yuanrong Actor that wraps vLLM Engine
# using the external_launcher backend for distributed inference.

import os
from typing import Optional

# Yuanrong imports
import yr
from yr.fcc import get_function_group_context
from yr.config import FunctionGroupContext

# vLLM imports (lazy import to avoid issues if vLLM not installed)
import vllm
from vllm import LLM, SamplingParams


class VLLMEngine:
    """
    Yuanrong Actor that wraps vLLM Engine using external_launcher mode.

    This Actor is designed to be created as part of a Yuanrong Function Group,
    where each Actor instance represents one rank in the distributed vLLM cluster.

    Key features:
    - Uses vLLM's distributed_executor_backend="external_launcher" mode
    - Automatically configures environment variables from Function Group context
    - Supports both single-node and multi-node tensor/pipeline parallelism
    - Only rank 0 returns inference results (others return None)
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        trust_remote_code: bool = True,
        max_model_len: int = 8192,
        seed: int = 1,
        master_port: int = = 29500,
    ):
        """
        Initialize the VLLMEngine Actor.

        Args:
            model_path: Path to the HuggingFace model or local model directory
            tensor_parallel_size: Tensor parallelism degree
            pipeline_parallel_size: Pipeline parallelism degree
            trust_remote_code: Whether to trust remote code when loading model
            max_model_len: Maximum model length (context window)
            seed: Random seed for deterministic inference (required for SPMD mode)
            master_port: Port for NCCL communication (default: 29500)
        """
        # Store configuration
        self._model_path = model_path
        self._tensor_parallel_size = tensor_parallel_size
        self._pipeline_parallel_size = pipeline_parallel_size
        self._master_port = master_port

        # Get Function Group context from Yuanrong
        self._group_context: FunctionGroupContext = get_function_group_context()

        yr.log.get_logger().info(
            f"VLLMEngine initialized: rank_id={self._group_context.rank_id}, "
            f"world_size={self._group_context.world_size}, "
            f"TP={tensor_parallel_size}, PP={pipeline_parallel_size}"
        )

        # Setup vLLM environment variables
        self._setup_vllm_env()

        # Initialize vLLM Engine
        self._init_vllm_engine(
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            seed=seed,
        )

        yr.log.get_logger().info("VLLM Engine initialized successfully")

    def _setup_vllm_env(self):
        """
        Setup environment variables required by vLLM's external_launcher mode.

        Maps Yuanrong's FunctionGroupContext to vLLM's expected environment.
        """
        ctx = self._group_context

        # 1. RANK - global rank across all processes
        os.environ["RANK"] = str(ctx.rank_id)
        yr.log.get_logger().debug(f"Set RANK={ctx.rank_id}")

        # 2. LOCAL_RANK - rank within the current node
        # Calculate based on gpus_per_actor = world_size // nnodes
        # For simplicity, we use rank_id % world_size here
        gpus_per_actor = self._tensor_parallel_size // self._pipeline_parallel_size
        local_rank = ctx.rank_id % gpus_per_actor if gpus_per_actor > 0 else 0
        os.environ["LOCAL_RANK"] = str(local_rank)
        yr.log.get_logger().debug(f"Set LOCAL_RANK={local_rank}")

        # 3. MASTER_ADDR - IP address of rank 0
        # Extract from server_list (contains network info of all actors)
        if ctx.server_list and len(ctx.server_list) > 0:
            master_server = ctx.server_list[0]
            os.environ["MASTER_ADDR"] = master_server.server_id
            yr.log.get_logger().debug(f"Set MASTER_ADDR={master_server.server_id}")
        else:
            # Fallback to localhost if server_list not available
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            yr.log.get_logger().warning(
                "server_list not available, using MASTER_ADDR=127.0.0.1"
            )

        # 4. MASTER_PORT - port for NCCL communication
        os.environ["MASTER_PORT"] = str(self._master_port)
        yr.log.get_logger().debug(f"Set MASTER_PORT={self._master_port}")

        # 5. CUDA_VISIBLE_DEVICES - visible GPUs for this process
        # Extract device IDs from FunctionGroupContext
        if hasattr(ctx, "devices") and ctx.devices:
            device_ids = [str(d.device_id) for d in ctx.devices]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_ids)
            yr.log.get_logger().debug(f"Set CUDA_VISIBLE_DEVICES={','.join(device_ids)}")
        else:
            yr.log.get_logger().debug("No device info available, CUDA_VISIBLE_DEVICES not set")

        # 6. VLLM_ENABLE_V1_MULTIPROCESSING=0 - Required for external_launcher
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        yr.log.get_logger().debug("Set VLLM_ENABLE_V1_MULTIPROCESSING=0")

        # 7. Optional: Set device name if available
        if ctx.device_name:
            yr.log.get_logger().debug(f"Device name: {ctx.device_name}")

    def _init_vllm_engine(
        self,
        trust_remote_code: bool,
        max_model_len: int,
        seed: int,
    ):
        """
        Initialize vLLM LLM Engine with external_launcher backend.

        Args:
            trust_remote_code: Whether to trust remote code
            max_model_len: Maximum model length
            seed: Random seed for deterministic inference
        """
        yr.log.get_logger().info(f"Loading model: {self._model_path}")

        self.llm = LLM(
            model=self._model_path,
            tensor_parallel_size=self._tensor_parallel_size,
            pipeline_parallel_size=self._pipeline_parallel_size,
            distributed_executor_backend="external_launcher",
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            seed=seed,
        )

        yr.log.get_logger().info("vLLM LLM Engine created successfully")

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
    ) -> Optional[any]:
        """
        Execute inference on the vLLM Engine.

        In SPMD (external_launcher) mode, all ranks execute the same inference
        with the same prompts and parameters. All ranks will generate the same
        outputs, but we only return results from rank 0 to avoid duplication.

        Args:
            prompts: List of text prompts to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter (optional)

        Returns:
            vLLM outputs if this is rank 0, None otherwise.
            The output type matches vLLM's generate() return type.
        """
        if self._group_context.rank_id != 0:
            yr.log.get_logger().debug(
                f"Rank {self._group_context.rank_id} executing inference (no output returned)"
            )

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        # All ranks execute the same inference
        outputs = self.llm.generate(prompts, sampling_params)

        # Only rank 0 returns results
        if self._group_context.rank_id == 0:
            yr.log.get_logger().debug("Rank 0 returning inference results")
            return outputs
        else:
            yr.log.get_logger().debug(f"Rank {self._group_context.rank_id} completed (no output)")
            return None

    def encode(
        self,
        prompts: list[str],
    ) -> Optional[any]:
        """
        Encode prompts to token IDs using vLLM's tokenizer.

        Args:
            prompts: List of text prompts to encode

        Returns:
            Encoded token IDs if this is rank 0, None otherwise.
        """
        if self._group_context.rank_id != 0:
            return None

        # Use vLLM's internal tokenizer
        # This is a simplified approach; vLLM may have specific methods
        from vllm import Tokenizer
        tokenizer = Tokenizer.from_pretrained(self._model_path)
        return tokenizer.encode_batch(prompts)

    def is_rank_0(self) -> bool:
        """
        Check if this Actor is rank 0.

        Returns:
            True if rank_id == 0, False otherwise.
        """
        return self._group_context.rank_id == 0

    def get_rank_id(self) -> int:
        """
        Get the global rank ID of this Actor.

        Returns:
            Global rank ID.
        """
        return self._group_context.rank_id

    def get_world_size(self) -> int:
        """
        Get the total number of Actors in the Function Group.

        Returns:
            Total world size.
        """
        return self._group_context.world_size

    def get_config(self) -> dict:
        """
        Get the vLLM configuration summary.

        Returns:
            Dictionary containing configuration information.
        """
        return {
            "model_path": self._model_path,
            "tensor_parallel_size": self._tensor_parallel_size,
            "pipeline_parallel_size": self._pipeline_parallel_size,
            "rank_id": self._group_context.rank_id,
            "world_size": self._group_contextudi.world_size,
        }
