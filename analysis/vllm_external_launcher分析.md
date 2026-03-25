# vLLM基于ExternalAscent拉起LLM实例流程及NPU显存/GLOO HCCL通信组管理分析

## 一、External Launcher拉起LLM实例流程

### 1.1 入口文件与流程概览

**主入口文件**: `examples/offline_external_launcher.py`

**核心流程**:

```
用户脚本
  └─> offline_external_launcher.py (主进程)
        ├─> 创建多个子进程 (multiprocessing.Process)
        │    每个子进程设置:
        │    - MASTER_ADDR, MASTER_PORT, RANK, LOCAL_RANK, WORLD_SIZE
        │    - 调用 torch.distributed.init_process_group()
        │    - 创建 LLM 实例 (distributed_executor_backend="external_launcher")
        └─> 等待所有子进程完成 (join)
```

### 1.2 详细流程分析

#### 阶段1: 进程启动与参数设置 ([offline_external_launcher.py:262-312](examples/offline_external_launcher.py))

```python
# 1. 解析命令行参数
args = parse_args()

# 2. 计算world_size和进程配置
world_size = node_size * proc_per_node

# 3. 为每个本地rank创建一个子进程
for local_rank, rank in enumerate(range(proc_per_node * node_rank, ...)):
    proc = Process(
        target=main,  # 每个子进程运行main函数
        args=(
            local_rank,
            rank,              # 全局rank
            master_addr,        # 多节点通信地址
            master_port,
            args.model_weight_gib,
            args.model,
            world_size,        # 总进程数
            tp_size,
            args.enable_expert_parallel,
            ...
        ),
    )
    proc.start()
    procs.append(proc)

# 4. 等待所有子进程完成
for proc in procs:
    proc.join(timeout=600)
```

#### 阶段2: 子进程初始化分布式环境 ([offline_external_launcher.py:173-184](examples/offline_external_launcher.py))

```python
def main(local_rank, rank, master_addr, master_port, ...):
    # 设置环境变量供torch.distributed使用
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # 关键：初始化torch分布式进程组
    # backend="cpu:gloo,npu:hccl" 表示:
    # - CPU通信使用gloo后端
    # - NPU通信使用hccl后端
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="cpu:gloo,npu:hccl",
            world_size=world_size,
            rank=rank,
        )
```

#### 阶段3: 创建LLM实例 ([offline_external_launcher.py:195-204](examples/offline_external_launcher.py))

```python
# 创建LLM实例，使用external_launcher后端
llm = LLM(
    model = model,
    tensor_parallel_size = tensor_parallel_size,
    enable_expert_parallel = enable_expert_parallel,
    ...
    distributed_executor_backend = "external_launcher",  # 关键配置
    ...
)

# 执行推理
outputs = llm.generate(prompts, sampling_params)
```

#### 阶段4: LLM内部执行器选择

当 `distributed_executor_backend="external_launcher"` 时：
- `LLMEngine.from_engine_args()` 会选择 `ExecutorWithExternalLauncher`
- 由于进程已由外部启动，不会重复创建worker进程
- 通过 `ExecutorWithExternalLauncher` 的 `collective_rpc()` 直接调用worker方法

### 1.3 External Launcher关键特点

1. **进程管理**: 用户脚本负责进程生命周期管理（启动、监控、销毁）
2. **分布式通信**: 每个进程独立初始化 `torch.distributed.init_process_group`
3. **多节点支持**: 通过MASTER_ADDR/MASTER_PORT实现跨节点通信
4. **无重复进程**: vLLM内部的executor不会重复创建worker，而是通过RPC与外部进程通信

---

## 二、NPU显存释放逻辑

### 2.1 Sleep Mode实现架构

**核心组件**: `vllm_ascend/device_allocator/camem.py` - CaMemAllocator

```
CaMemAllocator (单例模式)
  └─> 管理内存池 (Memory Pool)
       ├─> use_memory_pool(tag): 上下文管理器，标记分配的内存
       ├─> sleep(offload_tags): 休眠模式，释放/卸载内存
       └─> wake_up(tags): 唤醒模式，重新加载内存
```

### 2.2 内存分配流程 ([camem.py:225-259](vllm_ascend/device_allocator/camem.py))

```python
@contextmanager
def use_memory_pool(self, tag: str):
    """
    上下文管理器：所有在此上下文中创建的tensor都会
    分配在内存池中，并标记为指定tag
    """
    old_tag = self.current_tag
    self.current_tag = tag

    # 使用torch的pluggable allocator
    with use_memory_pool_with_allocator(
        self.python_malloc_callback,  # 自定义malloc回调
        self.python_free_callback      # 自定义free回调
    ) as data:
        self.allocator_and_pools[tag] = data
        yield  # 在此上下文中分配的内存

        self.current_tag = old_tag
```

**内存分配回调** ([camem.py:160-166](vllm_ascend/device_allocator/camem.py)):

```python
def python_malloc_callback(self, allocation_handle: HandleType):
    """
    当内存池分配内存时调用
    allocation_handle = (py_device, py_alignedSize, py_d_mem, py_p_memHandle)
    """
    py_d_mem = allocation_handle[2]  # 设备内存指针
    # 保存分配元数据，用于后续sleep/wake_up
    self.pointer_to_data[py_d_mem] = AllocationData(
        handle = allocation_handle,
        tag = self.current_tag,  # 使用当前tag标记
        cpu_backup_tensor = None
    )
```

### 2.3 Sleep Mode - 释放NPU显存 ([camem.py:177-205](vllm_ascend/device_allocator/camem.py))

```python
def sleep(self, offload_tags: tuple[str, ...] | None = None):
    """
    进入休眠模式：
    - 指定tag的内存会被卸载到CPU内存
    - 其他内存会被直接丢弃（释放）
    """
    if offload_tags is None:
        offload_tags = (CaMemAllocator.default_tag,)

    for ptr, data in self.pointer_to_data.items():
        handle = data.handle
        size_in_bytes = handle[1]  # 分配大小

        if data.tag in offload_tags:
            # 策略1：：卸载到CPU
            # 1. 创建CPU pinned内存用于备份
            cpu_backup_tensor = torch.empty(
                size_in_bytes,
                dtype = torch.uint8,
                device = "cpu",
                pin_memory = True
            )
            cpu_ptr = cpu_backup_tensor.data_ptr()

            # 2. 使用ACL API从NPU拷贝到CPU
            memcpy(
                cpu_ptr,           # 目标（CPU）
                dest_max,
                ptr,                # 源（NPU）
                size_in_bytes,
                ACL_MEMCPY_DEVICE_TO_HOST  # 方向：Device -> Host
            )

            # 3. 保存备份tensor
            data.cpu_backup_tensor = cpu_backup_tensor

        # 3. 释放NPU内存映射
        unmap_and_release(handle)  # 调用python_unmap_and_release
```

**ACL C++层面释放** ([camem.py:88-89](vllm_ascend/device_allocator/camem.py)):

```python
def unmap_and_release(allocation_handle):
    """
    调用C扩展的python_unmap_and_release
    allocation_handle = (py_device, py_alignedSize, py_d_mem, py_p_memHandle)
    """
    python_unmap_and_release(*allocation_handle)
    # 这会：
    # 1. aclrtUnmapDevice (取消内存映射)
    # 2. aclrtFree (释放内存块)
```

### 2.4 Wake Up - 重新加载内存 ([camem.py:206-224](vllm_ascend/device_allocator/camem.py))

```python
def wake_up(self, tags: list[str] | None = None):
    """
    唤醒模式：
    - 重新映射NPU内存
    - 将CPU备份的数据拷回NPU
    - 清理CPU备份
    """
    for ptr, data in self.pointer_to_data.items():
        if tags is None or data.tag in tags:
            handle = data.handle

            # 1. 重新创建NPU内存映射
            create_and_map(handle)  # 调用python_create_and_map

            # 2. 如果有CPU备份，拷回NPU
            if data.cpu_backup_tensor is not None:
                cpu_backup_tensor = data.cpu_backup_tensor
                size_in_bytes = cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
                cpu_ptr = cpu_backup_tensor.data_ptr()

                # 使用ACL API从CPU拷贝到NPU
                memcpy(
                    ptr,                # 目标（NPU）
                    dest_max,
                    cpu_ptr,           # 源（CPU）
                    size_in_bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE  # 方向：Host -> Device
                )

                # 3. 清理CPU备份
                data.cpu_backup_tensor = None
```

### 2.5 Worker层面的Sleep/Wake Up ([worker.py:190-244](vllm_ascend/worker/worker.py))

```python
class NPUWorker:
    def sleep(self, level: int = 1) -> None:
        """
        Worker层级的sleep接口
        level = 1: 释放weights
        level = 2: 完全释放，包括buffers
        """
        free_bytes_before_sleep = torch.npu.mem_get_info()[0]

        # Level 2: 保存model的buffers到CPU
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone()
                for name, buffer in model.named_buffers()
            }

        # 调用allocator的sleep
        allocator = CaMemAllocator.get_instance()
        allocator.sleep(offload_tags = ("weights",) if level == 1 else tuple())

        free_bytes_after_sleep, total = torch.npu.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        logger.info(
            "Sleep mode freed %.2f GiB memory, %.2f GiB memory is still in use.",
            freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes,
        )

    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        Worker层级的wake_up接口
        """
        allocator = CaMemAllocator.get_instance()
        allocator.wake_up(tags = tags)

        # 处理MOE模型的特殊转置需求
        if tags is None or "weights" in tags:
            for name, param in model.named_parameters():
                if "w2_weight" in name or "w13_weight" in name:
                    # 转置参数维度以适配wake_up后的布局
                    param.transpose(1, 2)

        # Level 2: 恢复buffers
        if len(self._sleep_saved_buffers):
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}
```

### 2.6 模型加载时的内存池使用 ([worker.py:429-440](vllm_ascend/worker/worker.py))

```python
def load_model(self) -> None:
    if self.vllm_config.model_config.enable_sleep_mode:
        allocator = CaMemAllocator.get_instance()
        assert allocator.get_current_usage() == 0, \
            "Sleep mode can only be used for one instance per process."

        # 在内存池中加载模型权重，标记为"weights"
        context = allocator.use_memory_pool(tag = "weights")
    else:
        from contextlib import nullcontext
        context = nullcontext()

    with context:
        self.model_runner.load_model()  # 所有tensor分配都在内存池中
```

### 2.7 KV Cache的内存池使用 ([worker.py:514-525](vllm_ascend/worker/worker.py))

```python
def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
    if self.vllm_config.model_config.enable_sleep_mode:
        allocator = CaMemAllocator.get_instance()
        # KV cache标记为"kv_cache"
        context = allocator.use_memory_pool(tag = "kv_cache")
    else:
        context = nullcontext()

()

    with context:
        self.model_runner.initialize_kv_cache(kv_cache_config)
```

---

## 三、GLOO HCCL通信组生命周期管理

### 3.1 通信后端架构

vLLM-Ascend使用**双后端**架构：
- **HCCL (NPU设备)**: 用于设备间tensor通信（all-reduce, all-gather等）
- **GLOO (CPU)**: 用于进程间协调、消息传递

```
torch.distributed.init_process_group(
    backend = "cpu:gloo,npu:hccl"  # 双backend
)
```

### 3.2 GroupCoordinatorPatch实现 ([patch_distributed.py:28-88](vllm_ascend/patch/worker/patch_distributed.py))

```python
class GroupCoordinatorPatch(GroupCoordinator):
    def __init__(self, group_ranks, local_rank, torch_distributed_backend,
                 use_device_communicator, group_name, ...):
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank

        self_device_group = None
        self_cpu_group = None
        hccl_pg_options = create_hccl_pg_options(group_name)

        for ranks in group_ranks:
            # 1. 创建HCCL设备组（用于tensor通信）
            device_group = torch.distributed.new_group(
                ranks,
                backend = torch_distributed_backend,  # npu:hccl
                pg_options = hccl_pg_options
            )

            # 2. 创建GLOO CPU组（用于进程协调）
            cpu_group = torch.distributed.new_group(
                ranks,
                backend = "gloo"  # CPU通信
            )

            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self_device_group = device_group
                self_cpu_group = cpu_group

        assert self_cpu_group is not None
        assert self_device_group is not None

        self.cpu_group = self_cpu_group
        self.device_group = self_device_group
        self.device = torch.npu.current_device()

        # 可选：使用NPUCommunicator进行自定义设备通信
        if use_device_communicator and self.world_size > 1:
            self.device_communicator = NPUCommunicator(
                cpu_group = self.cpu_group,
                device = self.device,
                device_group = self.device_group,
                unique_name = self.unique_name,
            )
```

### 3.3 Worker初始化分布式环境 ([worker.py:575-588](vllm_ascend/worker/worker.py))

```python
def _init_worker_distributed_environment(self) -> None:
    """初始化Worker的分布式环境"""
    init_batch_invariance()

    # 初始化全局分布式环境
    init_distributed_environment(
        self.parallel_config.world_size,
        self.rank,
        self.distributed_init_method,
        self.local_rank,
        "hccl"  # 后端
    )

    # 初始化模型并行组（TP, PP, CP等）
    ensure_model_parallel_initialized(
        self.parallel_config.tensor_parallel_size,
        self.parallel_config.pipeline_parallel_size,
        self.parallel_config.prefill_context_parallel_size,
        self.parallel_config.decode_context_parallel_size,
    )

    # 初始化Ascend特有的并行组（MC2, OTP, MLP_TP等）
    init_ascend_model_parallel(self.parallel_config)

    ensure_ec_transfer_initialized(self.vllm_config)
```

### 3.4 Ascend并行组初始化 ([parallel_state.py:30-227](vllm_ascend/distributed/parallel_state.py))

```python
def init_ascend_model_parallel(parallel_config):
    """
    初始化Ascend特有的并行组：
    - _MC2: MC2操作组
    - _MLP_TP, _OTP, _LMTP, _EMBED_TP: 细粒度TP组
    - _FLASHCOMM2_OTP, _FLASHCOMM2_ODP: FlashComm2操作组
    - _SHARD_WEIGHT: 权重分片组
    - _P_TP: Prefill Tensor Parallel组组
    """
    world_size = torch.distributed.get_world_size()
    backend = torch.distributed.get_backend(get_world_group().device_group)

    # 为每个组创建GroupCoordinatorPatch
    # 每个GroupCoordinatorPatch内部会创建：
    # 1. HCCL device_group（用于tensor通信）
    # 2. GLOO cpu_group（用于协调）

    # 例如：创建MC2组
    _MC2 = init_model_parallel_group(
        group_ranks,
        local_rank,
        backend,
        group_name = "mc2"
    )
    # 这会创建一个GroupCoordinatorPatch实例
```

### 3.5 GroupCoordinator销毁流程

#### GroupCoordinator.destroy() ([parallel_state.py:1055-1065](vllm/distributed/parallel_state.py))

```python
def destroy(self):
    """销毁通信组"""
    # 1. 销毁设备组（HCCL）
    if hasattr(self, "device_group"):
        torch.distributed.destroy_process_group(self.device_group)
        del self.device_group

    # 2. 销毁CPU组（GLOO）
    if hasattr(self, "cpu_group"):
        torch.distributed.destroy_process_group(self.cpu_group)
        del self.cpu_group

    # 3. 销毁设备通信器
    if self.device_communicator is not None:
        self.device_communicator.destroy()

    # 4. 清理消息广播器
    if self.mq_broadcaster is not None:
        self.mq_broadcaster = None
```

#### destroy_model_parallel() ([parallel_state.py:1856-1887](vllm/distributed/parallel_state.py))

```python
def destroy()_ascend_model_parallel():
    """销毁所有Ascend并行组"""
    # 销毁MC2组
    global _MC2
    if _MC2:
        _MC2.destroy()
        _MC2 = None

    # 销毁其他组（MLP_TP, OTP, LMTP等）
    ...
```

#### destroy_distributed_environment() ([parallel_state.py:1895-1903](vllm/distributed/parallel_state.py))

```python
def destroy_distributed_environment():
    """销毁分布式环境"""
    global _WORLD, _NODE_COUNT

    # 1. 销毁world组
    if _WORLD:
        _WORLD.destroy()
        _WORLD = None
        _NODE_COUNT = None

    # 2. 销毁torch分布式进程组
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        # 这会销毁所有HCCL和GLOO组及其资源
```

### 3.6 Worker进程退出时的清理 ([multiproc_executor.py:743-752](vllm/v1/executor/multiproc_executor.py))

```python
def shutdown(self):
    """Worker进程退出时的清理"""
    # 1. 关闭消息队列
    if self.rpc_broadcast_mq is not None:
        self.rpc_broadcast_mq.shutdown()
    if self.worker_response_mq is not None:
        self.worker_response_mq.shutdown()

    # 2. 关闭worker
    self.worker.shutdown()

    # 3. 销毁模型并行组（包括所有HCCL/GLOO组）
    destroy_model_parallel()

    # 4. 销毁分布式环境
    destroy_distributed_environment()
```

### 3.7 主进程退出时的清理 ([offline_external_launcher.py:251-258](examples/offline_external_launcher.py))

```python
def cleanup_env_and_memory():
    """主进程退出时的清理"""
    # 1. 销毁模型并行组
    destroy_model_parallel()

    # 2. 销毁分布式环境
    destroy_distributed_environment()

    # 3. 销毁torch进程组（如果尚未销毁）
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()

    # 4. 显存清理
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
```

---

## 四、LLM使用External Launcher后的内部调用流程

### 4.1 Executor选择 ([abstract.py:47-86](vllm/v1/executor/abstract.py))

```python
@staticmethod
def get_class(vllm_config: VllmConfig) -> type["Executor"]:
    executor_class: type[Executor]
    parallel_config = vllm_config.parallel_config
    distributed_executor_backend = parallel_config.distributed_executor_backend

    # 根据distributed_executor_backend选择executor
    if distributed_executor_backend == "ray":
        from vllm.v1.executor.ray_executor import RayDistributedExecutor
        executor_class = RayDistributedExecutor
    elif distributed_executor_backend == "mp":
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor
        executor_class = MultiprocExecutor
    elif distributed_executor_backend == "uni":
        from vllm.v1.executor.uniproc_executor import UniProcExecutor
        executor_class = UniProcExecutor
    elif distributed_executor_backend == "external_launcher":
        # 关键：使用ExecutorWithExternalLauncher
        executor_class = ExecutorWithExternalLauncher

    return executor_class
```

### 4.2 ExecutorWithExternalLauncher实现 ([uniproc_executor.py:142-188](vllm/v1/executor/uniproc_executor.py))

```python
class ExecutorWithExternalLauncher(UniProcExecutor):
    """
    使用外部启动器的执行器
    - 专为torchrun兼容的启动器设计
    - 适用于离线推理的tensor并行场景
    - 用户使用torchrun启动多个进程，vLLM不重复创建worker
    """

    def _init_executor(self) -> None:
        """
        初始化executor（但与UniProcExecutor的关键区别）
        """
        # 不调用super()._init_executor()，避免重复创建worker
        # 只初始化driver_worker，用于主进程的本地通信
        self.driver_worker = WorkerWrapperBase(rpc_rank = 0)

    def _distributed_args(self) -> tuple[str, int, int]:
        """
        从环境变量读取分布式参数
        - 外部启动器已通过torchrun设置这些环境变量
        """
        # required env vars:
        # - RANK
        # - LOCAL_RANK
        # - MASTER_ADDR
        # - MASTER_PORT
        distributed_init_method = "env://"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        return distributed_init_method, rank, local_rank

    def determine_available_memory(self) -> list[int]:
        """
        获取可用内存（跨所有rank取最小值）
        """
        # 使用GLOO CPU组进行all_reduce
        memory = super().determine_available_memory()
        from vllm.distributed.parallel_state import get_world_group
        cpu_group = get_world_group().cpu_group
        memory_tensor = torch.tensor([memory], device = "cpu", dtype = torch.int64)
        dist.all_reduce(memory_tensor, group = cpu_group, op = dist.ReduceOp.MIN)
        return [memory_tensor.item()]

    def collective_rpc(self, method, args, kwargs, ...) -> Any:
        """
        通过RPC调用worker的方法
        - 与UniProcExecutor不同，这里不需要进程间队列通信
        - 直接在本进程中调用run_method
        """
        result = run_method(self.driver_worker, method, args, kwargs)
        # 处理异步输出
        if isinstance(result, AsyncModelRunnerOutput):
            result = result.get_output()
        return result if single_value else [result]
```

### 4.3 LLMEngine初始化流程 ([llm_engine.py:48-156](vllm/v1/engine/llm_engine.py))

```python
class LLMEngine:
    def __init__(self, vllm_config, executor_class, ...):
        parallel_config = vllm_config.parallel_config
        executor_backend = parallel_config.distributed_executor_backend

        # 检查是否使用external_launcher + DP
        self.external_launcher_dp = (
            parallel_config.data_parallel_size > 1
            and executor_backend == "external_launcher"
        )

        # 如果是外部launcher且非多进程模式，初始化DP组
        if (
            not multiprocess_mode
            and parallel_config.data_parallel_size > 1
            and not self.external_launcher_dp
        ):
            self.dp_group = parallel_config.stateless_init_dp_group()
        else:
            self.dp_group = None

        # 创建EngineCoreClient
        # 对于external_launcher，使用InprocClient（内部单进程）
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode = multiprocess_mode,  # = False
            asyncio_mode = False,
            vllm_config = vllm_config,
            executor_class = executor_class,  # ExecutorWithExternalLauncher
            ...
        )

        # 对于external_launcher DP，复用外部启动的DP组
        if self.external_launcher_dp:
            self.dp_group = get_dp_group().cpu_group
```

### 4.4 完整调用链路

```
用户脚本启动多个进程（torch.distributed已初始化）
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ LLmEngine(model = ..., distributed_executor_backend = "external_launcher") │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ LLmEngine.__init__()                                        │
│  └─> Executor.get_class()                          │
│       └─> ExecutorWithExternalLauncher           │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ ExecutorWithExternalLauncher._init_executor()             │
│  - 创建driver_worker（WorkerWrapperBase）          │
│  - 不创建子进程（由外部管理）          │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ ExecutorWithExternalLauncher.collective_rpc()              │
│  - run_method(driver_worker, "execute_model", ...)     │
│  - 直接在本进程中调用worker方法                     │
└───────────────────────────────────────────────────────────────────┘
```

### 4.5 External Launcher模式下的关键点

1. **进程管理**
   - 用户脚本负责启动和管理多个torchrun进程
   - vLLM不创建子进程，只在本进程内运行

2. **分布式通信**
   - 外部已初始化torch.distributed（backend = "cpu:gloo,npu:hccl"）
   - vLLM从环境变量读取RANK, LOCAL_RANK, MASTER_ADDR等

3. **Executor选择**
   - Executor.get_class()返回ExecutorWithExternalLauncher
   - 继承UniProcExecutor，但修改了_init_executor和_distributed_args

4. **Worker通信**
   - ExecutorWithExternalLauncher使用run_method直接调用
   - 没有进程间消息队列（如MultiprocExecutor使用的）

5. **模型并行组**
   - Worker初始化时会调用init_distributed_environment()
   - 创建TP、PP等组（通过GroupCoordinatorPatch）
   - 使用已存在的torch.distributed进程组

---

## 五、关键文件索引

| 功能模块 | 文件路径 | 关键类/函数 |
|---------|---------|------------|
| External Launcher入口 | examples/offline_external_launcher.py | main(), cleanup_env_and_memory() |
| Worker实现 | vllm_ascend/worker/worker.py | NPUWorker, sleep(), wake_up() |
| 内存池管理 | vllm_ascend/device_allocator/camem.py | CaMemAllocator, use_memory_pool() |
| 分布式补丁 | vllm_ascend/patch/worker/patch_distributed.py | GroupCoordinatorPatch |
| Ascend并行组 | vllm_ascend/distributed/parallel_state.py | init_ascend_model_parallel(), destroy_ascend_model_parallel() |
| 上游并行管理 | vllm/distributed/parallel_state.py | destroy_model_parallel(), destroy_distributed_environment() |
| ExecutorWithExternalLauncher | vllm/v1/executor/uniproc_executor.py | ExecutorWithExternalLauncher |
| UniProcExecutor | vllm/v1/executor/uniproc_executor.py | UniProcExecutor |
| Multiproc执行器 | vllm/v1/executor/multiproc_executor.py | MultiprocExecutor |

---

## 六、重要注意事项

### 6.1 GLOO/HCCL双后端注意事项

1. **GLOO组用于CPU协调**
   - 进程间同步、barrier、简单数据交换
   - 基于TCP/Socket通信
   - 不传输大tensor数据

2. **HCCL组用于NPU通信**
   - all-reduce, all-gather, all-to-all等集合通信
   - 环网通信，高带宽
   - 传输大tensor数据

3. **GroupCoordinatorPatch同时维护两个组**
   - self.cpu_group: GLOO组
   - self.device_group: HCCL组
   - 销毁时需要分别销毁

### 6.2 Sleep Mode注意事项

1. **单进程限制**
   `assert allocator.get_current_usage() == 0`
   Sleep Mode只能用于每个进程一个实例的场景

2. **内存池使用范围**
   - model weights: tag = "weights"
   - KV cache: tag = "kv_cache"
   - 未标记的内存默认为"default" tag

3. **Level 2模式**
   - 需要手动重新加载模型权重
   - buffers会被保存和恢复
   - 适用于需要完全释放显存的场景

### 6.3 External Launcher注意事项

1. **必须设置环境变量**
   MASTER_ADDR, MASTER_PORT, RANK, LOCAL_RANK, WORLD_SIZE

2. **backend配置**
   `torch.distributed.init_process_group(backend = "cpu:gloo,npu:hccl")`

3. **cleanup顺序**
   - 先调用destroy_model_parallel()
   - 再调用destroy_distributed_environment()
   - 最后调用torch.distributed.destroy_process_group()
