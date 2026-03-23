# Yuanrong + vLLM Integration

This project provides integration between **Yuanrong** (distributed computing framework, similar to Ray) and **vLLM** (high-throughput LLM inference engine) using vLLM's `external_launcher` backend.

## Features

- **Distributed Inference**: Leverage Yuanrong's Function Group to create distributed vLLM clusters
- **Tensor & Pipeline Parallelism**: Support for both tensor parallel (TP) and pipeline parallel (PP)
- **SPMD Mode**: Use vLLM's deterministic Single Program Multiple Data (SPMD) mode
- **Multi-node Support**: Seamlessly scale inference across multiple physical nodes
- **Simple API**: Easy-to-use Python API and command-line launcher

## Architecture

```
Driver Process
    │
    ├─> Yuanrong Function Group (world_size = TP * PP)
    │     ├─> Actor 0 (rank=0, local_rank=0)  ──> vLLM Engine (external_launcher)
    │     ├─> Actor 1 (rank=1, local_rank=1)  ──> vLLM Engine (external_launcher)
    │     ├─> Actor 2 (rank=2, local_rank=0)  ──> vLLM Engine (external_launcher)
    │     └─> Actor 3 (rank=3, local_rank=1)  ──> vLLM Engine (external_launcher)
    │
    └─> Client calls inference on Actor 0
```

## Installation

### Prerequisites

1. **Yuanrong**: Install and configure Yuanrong runtime
2. **vLLM**: Install vLLM with GPU support
   ```bash
   pip install vllm
   ```
3. **Python**: Python 3.8+

### Setup

Clone or copy the `yr_vllm_launcher` directory to your project:

```bash
cp -r /path/to/yr_vllm_launcher /your/project/path/
```

## Usage

### Python API

#### Basic Example

```python
import yr
from yr.config import Config
from client import launch_vllm_cluster

# Initialize Yuanrong
yr.init(yr.Config())

# Launch vLLM cluster
cluster = launch_vllm_cluster(
    model_path="meta-llama/Llama-2-7b",
    tensor_parallel_size=4,  # TP=4
    gpus_per_actor=1,         # 1 GPU per actor
)

# Perform inference
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is",
]
outputs = cluster.generate(prompts, max_tokens=50)

# Print results
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")

# Cleanup
yr.finalize()
```

#### Advanced Configuration

```python
from client import VLLMCluster

# Create cluster with custom configuration
cluster = VLLMCluster(
    model_path="meta-llama/Llama-3.1-70b",
    tensor_parallel_size=8,
    pipeline_parallel_size=2,  # Enable pipeline parallelism
    gpus_per_actor=2,          # 2 GPUs per actor
    nnodes=4,                  # 4 physical nodes
    trust_remote_code=True,
    max_model_len=32768,
    seed=42,                    # Reproducible inference
    master_port=29500,           # NCCL port
)

# Launch
cluster.launch()

# Inference
outputs = cluster.generate(
    prompts=["Write a short poem about AI"],
    max_tokens=100,
    temperature=0.9,
    top_p=0.95,
)

# Finalize
yr.finalize()
```

### Command Line Launcher

#### Single Inference

```bash
python launcher.py \
    --model-path meta-llama/Llama-2-7b \
    --tp 4 \
    --pp 1 \
    --prompt "Hello, world" \
    --max-tokens 50 \
    --temperature 0.8
```

#### Multiple Prompts

```bash
python launcher.py \
    --model-path meta-llama/Llama-2-7b \
    --tp 4 \
    --prompts "Hello" "How are you?" "Write a poem" \
    --max-tokens 100
```

#### Interactive Mode

```bash
python launcher.py \
    --model-path meta-llama/Llama-2-7b \
    --tp 4 \
    --interactive
```

Then enter prompts interactively:

```
===========================================================
Interactive Mode
===========================================================
Enter prompts to generate (Ctrl+C or 'quit' to exit)

Prompt: What is the meaning of life?

Generating with max_tokens=128...

Results:
  [0] What is the meaning of life?
      Generated: The meaning of life is a profound philosophical question that has been debated by thinkers, scientists, and ordinary people throughout human history. At its core, life's meaning is deeply personal and can vary greatly from person to person...

Prompt: quit
Exiting...
```

#### Multi-node Configuration

```bash
python launcher.py \
    --model-path meta-llama/Llama-3.1-70b \
    --tp 8 \
    --pp 2 \
    --gpus-per-actor 2 \
    --nnodes 4 \
    --interactive
```

This creates:
- World size: 16 (8 TP × 2 PP)
- Nodes: 4
- Actors per node: 4
- GPUs per actor: 2
- Total GPUs: 32 (4 nodes × 8 GPUs/node)

#### View Configuration Only

```bash
python launcher.py \
    --model-path meta-llama/Llama-2-7b \
    --tp 4 \
    --pp 1 \
    --config-only
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model-path` | Path to HuggingFace model or local directory (required) | - |
| `--tp`, `--tensor-parallel-size` | Tensor parallelism degree | 1 |
| `--pp`, `--pipeline-parallel-size` | Pipeline parallelism degree | 1 |
| `--gpus-per-actor` | Number of GPUs per actor | 1 |
| `--nnodes` | Number of physical nodes (auto-calculated if not set) | None |
| `--max-model-len` | Maximum model length / context window | 8192 |
| `--seed` | Random seed for deterministic inference | 1 |
| `--no-trust-remote-code` | Disable trusting remote code | False |
| `--master-port` | Port for NCCL communication | 29500 |
| `--prompt` | Single prompt to generate | None |
| `--prompts` | Multiple prompts to generate | None |
| `--max-tokens` | Maximum tokens to generate | 128 |
| `--temperature` | Sampling temperature | 0.8 |
| `--top-p` | Top-p (nucleus) sampling | 0.95 |
| `--top-k` | Top-k sampling | None |
| `--interactive` | Enter interactive mode | False |
| `--config-only` | Only print configuration without launching | False |
| `--yr-server-address` | Yuanrong server address | From env |
| `--yr-ds-address` | Yuanrong DataSystem address | From env |
| `--yr-in-cluster` | Run in cluster mode | True |

## API Reference

### VLLMCluster

```python
class VLLMCluster:
    """
    Client-side wrapper for distributed vLLM cluster.

    Manages a Yuanrong Function Group of VLLMEngine actors.
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
    )

    def launch(self):
        """Launch the distributed vLLM cluster."""

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
    ):
        """Execute distributed inference."""

    def encode(self, prompts: list[str]):
        """Encode prompts to token IDs."""

    def get_config(self) -> dict:
        """Get cluster configuration."""
```

### VLLMEngine (Actor)

```python
class VLLMEngine:
    """
    Yuanrong Actor that wraps vLLM Engine using external_launcher mode.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        trust_remote_code: bool = True,
        max_model_len: int = 8192,
        seed: int = 1,
        master_port: int = 29500,
    )

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
    ) -> Optional[any]:
        """
        Execute inference on vLLM Engine.
        Only rank 0 returns results to avoid duplication.
        """

    def is_rank_0(self) -> bool:
        """Check if this Actor is rank 0."""

    def get_rank_id(self) -> int:
        """Get global rank ID of this Actor."""

    def get_world_size(self) -> int:
        """Get total number of Actors."""
```

## How It Works

### Environment Variable Mapping

The integration maps Yuanrong's `FunctionGroupContext` to vLLM's expected environment variables:

| vLLM Variable | Yuanrong Source | Description |
|----------------|-----------------|-------------|
| `RANK` | `FunctionGroupContext.rank_id` | Global rank |
| `LOCAL_RANK` | `rank_id % gpus_per_actor` | Local rank within node |
| `MASTER_ADDR` | `server_list[0].server_id` | Master node IP |
| `MASTER_PORT` | User config (default: 29500) | NCCL port |
| `CUDA_VISIBLE_DEVICES` | Device info | Visible GPU list |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | Fixed to "0" | Required by external_launcher |

### Distributed Communication

1. **NCCL Initialization**: vLLM uses PyTorch NCCL for cross-node communication
2. **Network Configuration**: Yuanrong's `server_list` provides IP addresses and device info
3. **Port Configuration**: NCCL uses the configured `master_port` (default: 29500)

### Key Constraints

1. **`distributed_executor_backend="external_launcher"` must be set** (automatically done)
2. **`VLLM_ENABLE_V1_MULTIPROCESSING="0"` must be set** (automatically done)
3. **All Actors must have identical vLLM Engine configuration** for deterministic scheduling
4. **`seed` must be the same across all ranks** for reproducible results

## File Structure

```
yr_vllm_launcher/
├── vllm_actor.py      # VLLMEngine Actor class
├── client.py          # VLLMCluster client code
├── launcher.py         # Command-line launcher script
└── README.md          # This file
```

## Troubleshooting

### Common Issues

**Issue**: "world_size must be evenly divisible by nnodes"
- **Cause**: The total world size (TP × PP) is not evenly divisible by the number of nodes
- **Solution**: Adjust `nnodes`, `gpus_per_actor`, or parallelism degrees

**Issue**: "server_list not available"
- **Cause**: Yuanrong's FunctionGroupContext doesn't provide server information
- **Solution**: Check Yuanrong configuration and cluster connectivity

**Issue**: "CUDA out of memory"
- **Cause**: Model is too large for available GPU memory
- **Solution**: Reduce tensor parallelism, use a smaller model, or increase GPU memory per actor

**Issue**: "NCCL timeout"
- **Cause**: Network connectivity or firewall issues between nodes
- **Solution**: Check network configuration, open NCCL port (default: 29500)

### Debugging

Enable Yuanrong logging:

```python
import yr
from yr.config import Config

config = Config(log_level="DEBUG")
yr.init(config)
```

Set vLLM logging level:

```bash
export VLLM_LOGGING_LEVEL=DEBUG
```

## Performance Tips

1. **Use appropriate parallelism**:
   - For large models (70B+), use high TP (4, 8)
   - For long sequences, consider PP
2. **Batch prompts**: Process multiple prompts together for better throughput
3. **Use correct `max_tokens`**: Don't over-allocate for short generations
4. **Adjust `gpus_per_actor`**: Balance between number of actors and GPUs per actor

## Comparison with Ray

| Feature | Yuanrong | Ray |
|----------|----------|-----|
| Native SPMD support | ✅ Yes | ⚠️  Requires external_launcher |
| Function Group API | ✅ Built-in | ⚠️  Requires custom implementation |
| Scheduler integration | ✅ Yuanrong-native | ✅ Ray-native |
| Cross-node support | ✅ Yes | ✅ Yes |

## License

This integration code follows the Apache 2.0 license, compatible with both Yuanrong and vLLM.

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM External Launcher Example](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/torchrun_example.py)
- [Yuanrong Documentation](https://yuanrong.example.com/docs)

## Support

For issues specific to this integration, please check the Yuanrong and vLLM documentation first.
