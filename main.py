#!/usr/bin/env python3
# coding=UTF-8
# Yuanrong + vLLM Integration: Model Manager
#
# Model switching with actor handler for node consistency and resource management.

import dataclasses
import time
from typing import List, Optional
import yr
import vllm_actor


@dataclasses.dataclass
class ModelConfig:
    """
    Model configuration with flexible parallel strategy.

    This allows different models to have different TP/PP/CP/DP configurations.
    """
    model_path: str
    tp: int = 1
    pp: int = 1
    cp: int = 1
    dp: int = 1
    npu_per_rank: int = 1  # NPU cards per rank
    gloo_port: int = 29500  # GLOO TCPStore port for distributed coordination
    hccl_port: int = 50000  # HCCL base port for NPU communication

    @property
    def total_ranks(self) -> int:
        """Total number of ranks needed."""
        return self.tp * self.pp * self.cp * self.dp

    @property
    def total_npu(self) -> int:
        """Total NPU cards needed."""
        return self.total_ranks * self.npu_per_rank


class ModelManager:
    """
    Manager for model lifecycle using actor handlers.

    Features:
    - Uses InvokeOptions with custom_resources for NPU binding
    - Supports flexible resource scaling
    - Direct actor creation and cleanup
    - Port management for GLOO and HCCL to avoid conflicts on model switch
    """

    def __init__(self, base_gloo_port: int = 29500, base_hccl_port: int = 50000):
        """
        Initialize ModelManager.

        Args:
            base_gloo_port: Starting GLOO TCPStore port
            base_hccl_port: Starting HCCL base port
        """
        self.current_config: Optional[ModelConfig] = None
        self.actors: List = []
        self._base_gloo_port = base_gloo_port
        self._base_hccl_port = base_hccl_port
        self._switch_count = 0  # Track number of switches for port increment

    def _get_next_ports(self) -> tuple[int, int]:
        """
        Get next available GLOO and HCCL ports.

        Returns:
            (gloo_port, hccl_port)
        """
        gloo_port = self._base_gloo_port + (self._switch_count * 10)
        hccl_port = self._base_hccl_port + (self._switch_count * 10)
        return gloo_port, hccl_port

    def _create_invoke_options(self, config: ModelConfig) -> yr.InvokeOptions:
        """
        Create InvokeOptions with NPU binding.

        Uses custom_resources to ensure actors land on nodes with required NPU.
        """
        opts = yr.InvokeOptions()
        opts.cpu = 2000
        opts.memory = 4096

        # NPU binding: ensures actors land on nodes with required NPU count
        opts.custom_resources = {
            "NPU/.+/count": config.npu_per_rank
        }

        return opts

    def load_model(self, config: ModelConfig) -> bool:
        """
        Load a model with specified configuration.

        This will:
        1. Unload current model if any
        2. Create actors with NPU binding and fresh ports
        3. Return True on success

        Args:
            config: Model configuration (ports will be auto-generated if switching)

        Returns:
            True if loaded successfully
        """
        print(f"\n=== Loading model: {config.model_path} ===")
        print(f"Configuration: TP={config.tp}, PP={config.pp}, CP={config.cp}, DP={config.dp}")
        print(f"Total ranks: {config.total_ranks}, Total NPU: {config.total_npu}")
        start_time = time.time()

        # Step 1: Unload current model if exists
        if self.actors:
            self.unload_model()
            # Increment ports for next model to avoid GLOO/HCCL conflicts
            self._switch_count += 1

        # Step 2: Update config with fresh ports for switch
        if self._switch_count > 0:
            gloo_port, hccl_port = self._get_next_ports()
            config.gloo_port = gloo_port
            config.hccl_port = hccl_port
            print(f"Using fresh ports: GLOO={gloo_port}, HCCL={hccl_port}")

        # Step 3: Create actors (synchronous - invoke blocks until __init__ completes)
        actor_refs = self._create_actors(config)
        print(f"Created {len(actor_refs)} actors with GLOO port={config.gloo_port}, HCCL port={config.hccl_port}")

        # Step 4: Update current config
        self.current_config = config
        elapsed = time.time() - start_time
        print(f"Model loaded successfully: {config.model_path} (time: {elapsed:.2f}s)")
        return True

    def _create_actors(self, config: ModelConfig):
        """
        Create actors with specified configuration.

        Each actor gets:
        - InvokeOptions with NPU binding via custom_resources
        - Its own rank_id
        - Same world_size for SPMD
        - GLOO and HCCL ports (incremented on each switch)

        Returns:
            List of created actor ObjectRefs (async creation)
        """
        opts = self._create_invoke_options(config)

        actor_refs = []
        for rank_id in range(config.total_ranks):
            actor_ref = vllm_actor.VLLMEngine.options(opts).invoke(
                config.model_path,
                config.tp,  # Pass tp for model parallel
                rank_id,
                config.total_ranks,
                config.gloo_port,  # GLOO TCPStore port
                config.hccl_port,  # HCCL base port
            )
            actor_refs.append(actor_ref)

        self.actors = actor_refs
        return actor_refs

    def unload_model(self):
        """
        Unload current model and cleanup resources.

        Process:
        1. Call actor.cleanup.invoke() to clean up vLLM internal state
        2. Wait for cleanup to complete (yr.wait)
        3. Call actor.terminate() to kill yr actor instance
        4. New actors will be created in load_model() with same resource requirements

        The new actors will be scheduled to same nodes and NPU cards
        via custom_resources binding.
        """
        if not self.actors:
            return

        print(f"\n=== Unloading model: {self.current_config.model_path} ===")
        start_time = time.time()

        # Cleanup each actor: first cleanup vLLM, wait, then terminate yr actor
        cleanup_futures = []
        for actor in self.actors:
            try:
                print(f"Cleaning up vLLM state in actor...")
                cleanup_futures.append(actor.cleanup.invoke())
            except Exception as e:
                print(f"Warning: actor.cleanup failed: {e}")

        # Wait for all cleanup operations to complete
        if cleanup_futures:
            cleanup_elapsed = time.time() - start_time
            print(f"Waiting for cleanup operations to complete...")
            yr.wait(cleanup_futures)
            wait_elapsed = time.time() - start_time
            print(f"Cleanup wait completed in {wait_elapsed - cleanup_elapsed:.2f}s")

        # Terminate yr actors after cleanup completes
        print(f"Terminating {len(self.actors)} yr actor instances...")
        terminate_start = time.time()
        for actor in self.actors:
            try:
                actor.terminate(is_sync=True)  # Synchronous termination
            except Exception as e:
                print(f"Warning: actor.terminate failed: {e}")

        terminate_elapsed = time.time() - terminate_start
        print(f"Actor termination completed in {terminate_elapsed:.2f}s")

        total_elapsed = time.time() - start_time
        self.actors = []
        self.current_config = None
        print(f"Model unloaded (total: {total_elapsed:.2f}s)")

    def switch(self, new_config: ModelConfig) -> bool:
        """
        Switch to a new model.

        This is an alias for load_model with logging.

        Args:
            new_config: New model configuration

        Returns:
            True if switched successfully
        """
        old_path = self.current_config.model_path if self.current_config else "None"
        print(f"\n=== Switching model: {old_path} -> {new_config.model_path} ===")
        start_time = time.time()
        result = self.load_model(new_config)
        elapsed = time.time() - start_time
        print(f"Switch completed in {elapsed:.2f}s")
        return result

    def run_inference(self, prompt: str, max_tokens: int = 10) -> str:
        """
        Run inference on all actors (SPMD).

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text from rank 0
        """
        if not self.actors:
            raise RuntimeError("No model loaded")

        print(f"\nRunning inference: prompt='{prompt}'")
        start_time = time.time()

        # Run on all actors
        futures = []
        for actor in self.actors:
            futures.append(actor.generate.invoke(prompt, max_tokens))

        # Wait for all to complete
        yr.wait(futures)

        # Return result from rank 0
        result = yr.get(futures[0])
        elapsed = time.time() - start_time
        print(f"Result: {result} (time: {elapsed:.2f}s)")
        return result

    def finalize(self):
        """
        Finalize and cleanup all resources.
        """
        self.unload_model()


def main():
    """Example usage with model switching."""
    yr.init(yr.Config(log_level="DEBUG"))
    print("Yuanrong initialized")

    # Create model manager
    manager = ModelManager()

    # Define model configurations with different parallel strategies
    model_a = ModelConfig(
        model_path="/workspace/models/qwen2.5_7B_Instruct",
        tp=2,           # TP=2
        npu_per_rank=1   # 1 NPU per rank -> 2 NPUs total
    )

    model_b = ModelConfig(
        model_path="/workspace/models/Qwen2.5-VL-7B-Instruct",
        tp=2,           # TP=2
        npu_per_rank=1   # Same NPU usage -> node consistency
    )

    # Example: Model with different parallel strategy (extensibility demo)
    # model_c = ModelConfig(
    #     model_path="/workspace/models/large_model",
    #     tp=4,           # Different TP
    #     dp=2,           # Add DP
    #     npu_per_rank=1   # 8 NPUs total
    # )

    prompt = "Hello, my name is"

    # Load and run Model A
    manager.load_model(model_a)
    manager.run_inference(prompt, max_tokens=10)

    # Switch to Model B
    manager.switch(model_b)
    manager.run_inference(prompt, max_tokens=10)

    print("\n=== Test completed ===")

    # Cleanup
    manager.finalize()
    yr.finalize()
    print("Yuanrong finalized")


if __name__ == "__main__":
    main()
