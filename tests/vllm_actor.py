#!/usr/bin/env python3
# coding=UTF-8
# Yuanrong + vLLM Integration: Simplified VLLMEngine Actor
#
# Minimal implementation: Yuanrong Actor Actor wrapping vLLM with external_launcher mode.

import contextlib
import gc
import os
import time

import torch
import torch.distributed

import yr
import yr.log
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (  # noqa E402
    destroy_distributed_environment,
    destroy_model_parallel,
)


@yr.instance
class VLLMEngine:
    """Yuanrong Actor Actor wrapping vLLM Engine (external_launcher mode)."""

    def __init__(self, model_path: str, tp: int = 1, rank_id: int = 0, world_size: int = 2):
        """
        Initialize VLLMEngine Actor.

        Args:
            model_path: Model name (e.g., "gpt2") or local path
            tp: Tensor parallel size (default: 1)
            rank_id: Rank of this instance (default: 0)
            world_size: Total number of instances (default: 2)
        """
        # Delay logger initialization to avoid module-level issues
        _logger = yr.log.get_logger()
        _logger.info(f"VLLMEngine __init__ called: rank={rank_id}, world_size={world_size}, tp={tp}")

        self._rank_id = rank_id
        self._world_size = world_size
        self._tp = tp

        # Setup vLLM environment variables for external_launcher
        os.environ["RANK"] = str(rank_id)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # HCCL environment variables
        if "HCCL_IF_BASE_PORT" not in os.environ:
            os.environ["HCCL_IF_BASE_PORT"] = "50000"

        _logger.info("Initializing torch:dist init")
        torch.distributed.init_process_group(
            init_method="env://",
            backend="cpu:gloo,npu:hccl",
            rank=rank_id,
            world_size=world_size,
        )
        _logger.info("torch.distributed process group initialized successfully")

        _logger.info(f"Initializing vLLM LLM: model={model_path}, tp={tp}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp,
            trust_remote_code=True,
            distributed_executor_backend="external_launcher",
        )
        _logger.info("vLLM LLM initialized successfully")

    def generate(self, prompt: str, max_tokens: int = 50) -> str | None:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text (rank 0 only), None otherwise
        """
        _logger = yr.log.get_logger()
        _logger.info(f"generate called on rank {self._rank_id}: prompt='{prompt}', max_tokens={max_tokens}")
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.8, top_p=0.95, logprobs=1)
        outputs = self.llm.generate([prompt], sampling_params)
        _logger.info(f"vLLM generate completed on rank {self._rank_id}")

        result = outputs[0].outputs[0].text
        _logger.info(f"Returning generated text from rank {self._rank_id}: {result[:100]}...")
        return result

    def switch_model(self, new_model_path: str):
        """Switch to a new model.

        This deletes the old LLM instance (which destroys its worker processes),
        properly cleans up the distributed environment, then creates a new LLM instance
        with the new model.

        Args:
            new_model_path: Path to new model
        """
        _logger = yr.log.get_logger()
        _logger.info(f"Switching to new model: {new_model_path}")

        # Delete old LLM instance (this should destroy its worker processes)
        _logger.info("Deleting old LLM instance")
        del self.llm

        # Clean up vLLM model parallel state
        _logger.info("Cleaning up model parallel state")
        destroy_model_parallel()

        # Destroy vLLM distributed environment (terminates worker processes)
        _logger.info("Destroying vLLM distributed environment")
        destroy_distributed_environment()

        # Destroy torch.distributed process group
        _logger.info("Destroying torch.distributed process group")
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()

        # Force garbage collection and NPU cache clear
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()

        # Small delay to ensure cleanup completes
        time.sleep(1)

        # Reinitialize torch.distributed process group for new LLM instance
        _logger.info("Reinitializing torch.distributed process group")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                init_method="env://",
                backend="cpu:gloo,npu:hccl",
                rank=self._rank_id,
                world_size=self._world_size,
            )

        # Create new LLM instance with new model
        _logger.info(f"Creating new LLM instance with model: {new_model_path}")
        self.llm = LLM(
            model=new_model_path,
            tensor_parallel_size=self._tp,
            trust_remote_code=True,
            distributed_executor_backend="external_launcher",
        )

        _logger.info("Model switched successfully")

    def ready(self) -> bool:
        """Return True if actor is ready (used for synchronization after init)."""
        return True
