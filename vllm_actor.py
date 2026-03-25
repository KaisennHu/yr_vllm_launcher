#!/usr/bin/env python3
# coding=UTF-8
# Yuanrong + vLLM Integration: Simplified VLLMEngine Actor
#
# Minimal implementation: Yuanrong Actor Actor wrapping vLLM with external_launcher mode.

import contextlib
import gc
import os
import time
import datetime

import torch
import torch.distributed
from torch.distributed import TCPStore

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

    def __init__(
        self,
        model_path: str,
        tp: int = 1,
        rank_id: int = 0,
        world_size: int = 2,
        gloo_port: int = 29500,
        hccl_port: int = 50000,
    ):
        """
        Initialize VLLMEngine Actor.

        Args:
            model_path: Model name (e.g., "gpt2") or local path
            tp: Tensor parallel size (default: 1)
            rank_id: Rank of this instance (default: 0)
            world_size: Total number of instances (default: 2)
            gloo_port: GLOO TCPStore port for distributed coordination
            hccl_port: HCCL base port for NPU communication
        """
        # Delay logger initialization to avoid module-level issues
        _logger = yr.log.get_logger()
        _logger.info(
            f"VLLMEngine __init__ called: rank={rank_id}, world_size={world_size}, "
            f"tp={tp}, gloo_port={gloo_port}, hccl_port={hccl_port}"
        )

        self._rank_id = rank_id
        self._world_size = world_size
        self._tp = tp
        self._model_path = model_path
        self._gloo_port = gloo_port
        self._hccl_port = hccl_port
        self._llm = None
        self._store = None

        # Setup vLLM environment variables for external_launcher
        os.environ["RANK"] = str(rank_id)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # HCCL environment variables
        os.environ["HCCL_IF_BASE_PORT"] = str(hccl_port)

        self._initialize_distributed()
        self._create_llm()

    def _initialize_distributed(self):
        """
        Initialize torch.distributed process group using TCPStore.

        Using TCPStore with explicit port avoids GLOO backend conflicts
        when switching between models.
        """
        _logger = yr.log.get_logger()

        # Create TCPStore for distributed coordination
        # Each model instance uses a different port to avoid conflicts
        _logger.info(f"Creating TCPStore on port {self._gloo_port}")
        self._store = TCPStore(
            host_name="127.0.0.1",
            port=self._gloo_port,
            world_size=self._world_size,
            is_master=(self._rank_id == 0),
            timeout=datetime.timedelta(seconds=30),
        )

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="cpu:gloo,npu:hccl",
                store=self._store,
                rank=self._rank_id,
                world_size=self._world_size,
            )
            _logger.info("torch.distributed initialized successfully")

    def _create_llm(self):
        """Create LLM instance."""
        _logger = yr.log.get_logger()
        _logger.info(f"Creating LLM: model={self._model_path}, tp={self._tp}")
        self._llm = LLM(
            model=self._model_path,
            tensor_parallel_size=self._tp,
            trust_remote_code=True,
            distributed_executor_backend="external_launcher",
        )
        _logger.info("LLM created successfully")

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
        outputs = self._llm.generate([prompt], sampling_params)
        _logger.info(f"vLLM generate completed on rank {self._rank_id}")

        result = outputs[0].outputs[0].text
        _logger.info(f"Returning generated text from rank {self._rank_id}: {result[:100]}...")
        return result

    def ready(self) -> bool:
        """Return True if actor is ready (used for synchronization after init)."""
        return True

    def cleanup(self):
        """
        Cleanup resources - complete version.

        References offline_external_launcher.py:251-258 cleanup_env_and_memory()

        Cleanup order (critical):
        0. Delete TCPStore (to free GLOO port)
        1. Delete LLM instance
        2. destroy_model_parallel() - vLLM model parallel groups (TP/PP/CP)
        3. destroy_distributed_environment() - vLLM world group
        4. torch.distributed.destroy_process_group() - HCCL/GLOO groups
        5. gc.collect() - garbage collection
        6. torch.npu.empty_cache() - clear NPU cache
        7. torch.npu.reset_peak_memory_stats() - reset stats
        """
        _logger = yr.log.get_logger()
        _logger.info(f"cleanup on rank {self._rank_id}")

        # 0. Delete TCPStore (to release GLOO port for next model)
        if self._store is not None:
            _logger.info("Deleting TCPStore")
            del self._store
            self._store = None

        # 1. Delete LLM instance
        if self._llm is not None:
            _logger.info("Deleting LLM instance")
            del self._llm
            self._llm = None

        # 2. Destroy vLLM model parallel state (TP/PP/CP groups)
        _logger.info("Destroying model parallel state")
        try:
            destroy_model_parallel()
        except Exception as e:
            _logger.warning(f"destroy_model_parallel failed: {e}")

        # 3. Destroy vLLM distributed environment (world group)
        _logger.info("Destroying distributed environment")
        try:
            destroy_distributed_environment()
        except Exception as e:
            _logger.warning(f"destroy_distributed_environment failed: {e}")

        # 4. Destroy torch.distributed process group (HCCL/GLOO)
        _logger.info("Destroying torch.distributed process group")
        try:
            with contextlib.suppress(AssertionError):
                torch.distributed.destroy_process_group()
        except Exception as e:
            _logger.warning(f"torch.distributed.destroy_process_group failed: {e}")

        # 5. Force garbage collection
        _logger.info("Running garbage collection")
        gc.collect()

        # 6. Clear NPU cache
        _logger.info("Clearing NPU cache")
        try:
            torch.npu.empty_cache()
            torch.npu.reset_peak_memory_stats()
        except Exception as e:
            _logger.warning(f"NPU cache cleanup failed: {e}")

        _logger.info("cleanup completed")
