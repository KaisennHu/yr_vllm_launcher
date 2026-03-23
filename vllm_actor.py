#!/usr/bin/env python3
# coding=UTF-8
# Yuanrong + vLLM Integration: Simplified VLLMEngine Actor
#
# Minimal implementation: Yuanrong Actor wrapping vLLM with external_launcher mode.

import os
os.environ["VLLM_PLATFORM"] = "ascend"
import torch.distributed

import yr
import yr.log
from vllm import LLM, SamplingParams


@yr.instance
class VLLMEngine:
    """Yuanrong Actor wrapping vLLM Engine (external_launcher mode)."""

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

        _logger.info("Initializing torch.distributed process group with init_method=env://")
        # Initialize torch.distributed with HCCL backend for Ascend NPU
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                init_method="env://",
                backend="cpu:gloo,npu:hccl",
                rank=rank_id,
                world_size=world_size,
            )
        _logger.info("torch.distributed process group initialized successfully")

        _logger.info(f"Initializing vLLM LLM: model={model_path}, tp={tp}")
        # Initialize vLLM Engine (using external_launcher mode)
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

        if self._rank_id == 0:
            result = outputs[0].outputs[0].text
            _logger.info(f"Returning generated text from rank 0: {result[:100]}...")
            return result
        _logger.info(f"Rank {self._rank_id} returning None (not rank 0)")
        return None
