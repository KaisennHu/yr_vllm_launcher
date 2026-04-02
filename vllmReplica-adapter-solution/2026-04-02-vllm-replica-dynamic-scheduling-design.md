# vLLMReplica 实例动态调度增减设计

## 1. 设计目标

实现 vLLMReplica 实例的动态调度能力，解决多任务 RL 场景下的推理长尾问题：

1. **核心目标**：根据任务负载动态增减 vLLM 实例，最大化 GPU 利用率
2. **调度接口**：对接 `multi-rl-task-scheduler` 的 GroupScheduler 接口
3. **实例隔离**：每个任务的实例之间是同任务的不同 DP 切片
4. **TP 动态支持**：支持不同任务使用不同 TP 配置，共享 GPU 动态切换

---

## 2. 当前架构分析

### 2.1 现有 vLLMReplica 结构

```
vLLMReplica (RolloutReplica)
  ├── config: RolloutConfig
  ├── model_config: HFModelConfig
  ├── world_size = TP × PP
  ├── nnodes = world_size / gpus_per_node
  ├── workers: List[ActorHandle]        # Ray workers
  ├── servers: List[ActorHandle]        # vLLMHttpServer actors
  ├── resource_pool: RayResourcePool  # 固定的资源池
  └── launch_servers(): 创建固定数量的 server
```

**当前问题**：
- `resource_pool` 在 `init_standalone()` 时固定创建
- `servers` 一旦创建后无法动态增减
- GPU 资源在推理服务启动时就分配完毕

### 2.2 vLLMHttpServer 结构

```
vLLMHttpServer (Ray Actor)
  ├── engine: AsyncLLM
  ├── rollout_mode: RolloutMode
  ├── _server_address, _server_port
  └── launch_server(): 初始化 vLLM 引擎
```

**关键限制**：
- vLLM 引擎启动后需要显存初始化
- HCCL/GLOO 通信组在进程启动时创建
- 模型权重和 KV Cache 需要加载

---

## 3. 设计方案：Dynamic vLLMReplica

### 3.1 核心架构

```python
class DynamicvLLMReplica(RolloutReplica):
    """
    支持动态实例调度的 vLLMReplica 实现
    
    核心特性：
    1. 与 GroupScheduler 对接，上报任务状态
    2. 接收 reclaim/assign 指令，动态管理实例
    3. 支持相同 GPU 上切换不同 TP 配置
    4. 智能判断是否需要重建通信组
    """
    
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        task_id: str,  # 新增：任务标识
        scheduler_handle: ray.ActorHandle = None,  # 新增：调度器句柄
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        
        # ========== 调度器相关 ==========
        self.task_id = task_id
        self.scheduler_handle = scheduler_handle
        self.task_state_tracker = TaskStateTracker()
        
        # ========== 动态实例管理 ==========
        self.min_instances = config.get("min_instances", 1)  # 最小保持实例数
        self.max_instances = config.get("max_instances", None)  # 最大实例数限制
        self.current_instances = {}  # {instance_id: InstanceInfo}
        self.next_instance_id = 0
        
        # ========== GPU 资源跟踪 ==========
        self.allocated_gpus = set()  # {(machine_id, gpu_id), ...}
        self.last_allocation_config = None  # 上次分配的 TP/PP 配置
        
        # ========== 通信组复用缓存 ==========
        self.comm_group_cache = CommGroupCache()  # 缓存可复用的通信组
        
        # ========== 异步锁 ==========
        self.management_lock = asyncio.Lock()
```

### 3.2 实例信息结构

```python
@dataclass
class InstanceInfo:
    """单个 vLLM 实例的信息"""
    instance_id: int
    server_handle: ray.ActorHandle  # vLLMHttpServer actor
    placement: List[GPUPlacement]  # [(machine_id, gpu_id), ...]
    tp: int
    pp: int
    state: str  # "idle", "busy", "starting", "stopping"
    created_at: float
    
@dataclass
class GPUPlacement:
    """GPU 位置信息"""
    machine_id: str
    gpu_id: int
    rank_in_tp: int  # TP 组内的 rank
```

### 3.3 通信组缓存

```python
class CommGroupCache:
    """
    通信组缓存，支持相同 TP 配置复用
    
    核心思想：
    - 对于相同 TP 配置，复用 GLOO/HCCL 通信组
    - 仅重建模型并行相关的组
    - 避免频繁的 torch.distributed.destroy/init
    """
    
    def __init__(self):
        self.groups: Dict[Tuple[int, int], CommGroupInfo] = {}
        self.lock = asyncio.Lock()
    
    async def acquire(
        self, 
        tp: int, 
        world_size: int,
        gloo_port: int,
        hccl_port_range: Tuple[int, int]
    ) -> CommGroupInfo:
        """获取或创建通信组"""
        key = (tp, world_size)
        
        async with self.lock:
            if key in self.groups:
                # 复用现有通信组
                info = self.groups[key]
                info.ref_count += 1
                return info
            
            # 创建新通信组
            info = CommGroupInfo(
                tp=tp,
                world_size=world_size,
                gloo_port=gloo_port,
                hccl_port_range=hccl_port_range,
                ref_count=1
            )
            await info.initialize()
            self.groups[key] = info
            return info
    
    async def release(self, tp: int, world_size: int) -> None:
        """释放通信组引用"""
        key = (tp, world_size)
        
        async with self.lock:
            if key in self.groups:
                info = self.groups[key]
                info.ref_count -= 1
                
                if info.ref_count == 0:
                    # 无引用，销毁通信组
                    await info.destroy()
                    del self.groups[key]
    
    async def destroy_all(self) -> None:
        """销毁所有通信组"""
        async with self.lock:
            for info in self.groups.values():
                await info.destroy()
            self.groups.clear()

@dataclass
class CommGroupInfo:
    """通信组信息"""
    tp: int
    world_size: int
    gloo_port: int
    hccl_port_range: Tuple[int, int]
    ref_count: int
    store: Optional[Any] = None  # TCPStore
    hccl_pg: Optional[Any] = None  # HCCL process group
    gloo_pg: Optional[Any] = None  # GLOO process group
    
    async def initialize(self):
        """初始化分布式环境"""
        import torch.distributed
        from torch.distributed import TCPStore
        
        # 创建 TCPStore
        self.store = TCPStore(
            host_name="127.0.0.1",
            port=self.gloo_port,
            world_size=self.world_size,
            is_master=(rank == 0),
        )
        
        # 初始化进程组
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="cpu:gloo,npu:hccl",
                store=self.store,
                rank=rank,
                world_size=self.world_size,
            )
    
    async def destroy(self):
        """销毁分布式环境"""
        import torch.distributed
        
        # 销毁 HCCL/GLOO 组
        # ... 销毁逻辑
```

---

## 4. 调度器接口对接

### 4.1 任务状态上报

```python
async def _report_state_to_scheduler(self):
    """向 GroupScheduler 上报任务状态"""
    if self.scheduler_handle is None:
        return
    
    report = TaskStateReport(
        task_id=self.task_id,
        done_samples=self.task_state_tracker.done_samples,
        done_rounds=self.task_state_tracker.done_rounds,
        elapsed_time_sec=self.task_state_tracker.elapsed_time,
        remaining_samples=self.task_state_tracker.remaining_samples,
        current_instances=len(self.current_instances),
        idle_instances=sum(1 for i in self.current_instances.values() 
                       if i.state == "idle"),
        busy_instances=sum(1 for i in self.current_instances.values() 
                       if i.state == "busy"),
        in_rollout_phase=self.task_state_tracker.in_rollout_phase,
    )
    
    # 异步上报，不等待响应
    await self.scheduler_handle.report_state.remote(report)

@dataclass
class TaskStateReport:
    """任务状态报告（对应调度器文档中的接口）"""
    task_id: str
    done_samples: int
    done_rounds: int
    elapsed_time_sec: float
    remaining_samples: int
    current_instances: int
    idle_instances: int
    busy_instances: int
    in_rollout_phase: bool
    voluntary_reclaim: Optional[ReclaimConfirm] = None
```

### 4.2 回收指令处理

```python
async def reclaim_instances(self, num_instances: int) -> ReclaimConfirm:
    """
    回收指定数量的实例
    
    由 GroupScheduler 调用
    """
    async with self.management_lock:
        # 选择要回收的实例
        # 优先级：idle > busy
        idle_instances = [i for i in self.current_instances.values() 
                        if i.state == "idle"]
        busy_instances = [i for i in self.current_instances.values() 
                       if i.state == "busy"]
        
        to_reclaim = idle_instances[:num_instances]
        remaining = num_instances - len(to_reclaim)
        if remaining > 0:
            to_reclaim.extend(busy_instances[:remaining])
        
        # 回收实例
        reclaimed_gpus = []
        for instance_info in to_reclaim:
            gpus = await self._destroy_instance(instance_info)
            reclaimed_gpus.extend(gpus)
            del self.current_instances[instance_info.instance_id]
        
        return ReclaimConfirm(
            task_id=self.task_id,
            reclaimed_instances=len(to_reclaim),
            reclaimed_gpus=reclaimed_gpus
        )

async def _destroy_instance(self, instance_info: InstanceInfo) -> List[GPUPlacement]:
    """销毁单个实例"""
    logger.info(f"Destroying instance {instance_info.instance_id}")
    
    # 1. 停止 server
    try:
        await instance_info.server_handle.wait_for_requests_to_drain.remote()
    except Exception as e:
        logger.warning(f"Wait for requests drain failed: {e}")
    
    # 2. 释放通信组引用
    await self.comm_group_cache.release(
        tp=instance_info.tp, 
        world_size=instance_info.tp * instance_info.pp
    )
    
    # 3. 销毁 server actor
    ray.kill(instance_info.server_handle)
    
    return instance_info.placement
```

### 4.3 分配指令处理

```python
async def assign_instances(self, placements: List[List[GPUPlacement]]) -> None:
    """
    分配新实例
    
    placements: 每个实例的 GPU 列表
    """
    async with self.management_lock:
        for placement in placements:
            instance_id = self.next_instance_id
            self.next_instance_id += 1
            
            # 获取通信组
            comm_info = await self.comm_group_cache.acquire(
                tp=self.config.tensor_model_parallel_size,
                world_size=len(placement),
                gloo_port=self._get_free_gloo_port(),
                hccl_port_range=self._get_free_hccl_port_range()
            )
            
            # 创建实例
            instance_info = await self._create_instance(
                instance_id=instance_id,
                placement=placement,
                comm_info=comm_info
            )
            self.current_instances[instance_id] = instance_info
```

---

## 5. 实例创建与销毁

### 5.1 创建实例流程

```python
async def _create_instance(
    self,
    instance_id: int,
    placement: List[GPUPlacement],
    comm_info: CommGroupInfo
) -> InstanceInfo:
    """创建新的 vLLM 实例"""
    logger.info(f"Creating instance {instance_id} with TP={comm_info.tp}")
    
    # 1. 创建 Ray workers
    cuda_visible_devices = ",".join(str(g.gpu_id) for g in placement)
    
    workers = self._create_workers_for_instance(
        placement=placement,
        cuda_visible_devices=cuda_visible_devices
    )
    
    # 2. 创建 vLLMHttpServer actor
    server = self.server_class.options(
        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=placement[0].machine_id,
            soft=False,
        ),
        runtime_env={
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                "NCCL_CUMEM_ENABLE": "0",
                # 传递通信组配置
                "HCCL_IF_BASE_PORT": str(comm_info.hccl_port_range[0]),
            }
        },
        name=f"vllm_server_{self.task_id}_{instance_id}",
        max_concurrency=self.max_concurrency,
    ).remote(
        config=self.config,
        model_config=self.model_config,
        rollout_mode=self.rollout_mode,
        workers=workers,
        replica_rank=instance_id,
        node_rank=0,  # 暂时假设单节点
        gpus_per_node=len(placement),
        nnodes=1,
        cuda_visible_devices=cuda_visible_devices,
        # 传递通信组信息
        comm_group_info=comm_info,
    )
    
    # 3. 启动 server
    master_address, master_port, dp_rpc_port = await server.get_master_address.remote()
    await server.launch_server.remote(
        master_address=master_address, 
        master_port=master_port, 
        dp_rpc_port=dp_rpc_port
    )
    
    # 4. 更新跟踪
    self.allocated_gpus.update(
        (p.machine_id, p.gpu_id) for p in placement
    )
    
    return InstanceInfo(
        instance_id=instance_id,
        server_handle=server,
        placement=placement,
        tp=comm_info.tp,
        pp=1,  # 暂时假设 PP=1
        state="idle",
        created_at=time.time()
    )
```

### 5.2 智能通信组判断

```python
async def _should_rebuild_comm_group(self, new_tp: int) -> bool:
    """
    判断是否需要重建通信组
    
    策略：
    1. 如果 TP 相同，复用现有通信组
    2. 如果 TP 不同，必须重建（因为 HCCL 通信拓扑改变）
    """
    if self.last_allocation_config is None:
        return True  # 首次创建
    
    old_tp = self.last_allocation_config["tp"]
    
    if old_tp == new_tp:
        logger.info(f"TP unchanged ({new_tp}), reusing comm groups")
        return False
    else:
        logger.info(f"TP changed from {old_tp} to {new_tp}, rebuilding comm groups")
        return True
```

---

## 6. 初始化流程适配

### 6.1 修改 init_standalone

```python
async def init_standalone(self, scheduler_handle: ray.ActorHandle = None):
    """
    初始化动态资源池
    
    与原版区别：
    1. 不预先固定分配 GPU
    2. 注册到 GroupScheduler
    3. 等待首次分配指令
    """
    self.rollout_mode = RolloutMode.STANDALONE
    self.scheduler_handle = scheduler_handle
    
    # 注册任务到调度器
    if scheduler_handle is not None:
        raise ValueError("scheduler_handle is required for dynamic scheduling")
    
    # 创建最小资源池（仅用于注册，实际 GPU 由调度器分配）
    resource_pool_name = f"rollout_pool_{self.task_id}"
    resource_pool_spec = {
        resource_pool_name: [0] * self.nnodes,  # 初始为空
    }
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, 
        mapping=None
    )
    resource_pool_manager.create_resource_pool()
    self.resource_pool = resource_pool_manager.resource_pool_dict[resource_pool_name]
    
    # 注册任务
    task_config = TaskConfig(
        task_id=self.task_id,
        base_instances=self.config.get("base_instances", 1),
        tp=self.config.tensor_model_parallel_size,
        pp=self.config.pipeline_model_parallel_size,
        samples_per_round=self.config.get("samples_per_round", 1000),
        total_samples=self.config.get("total_samples", 10000),
    )
    
    registered = await scheduler_handle.register_task.remote(task_config)
    if not registered:
        raise ValueError(f"Task {self.task_id} registration failed")
    
    logger.info(f"Task {self.task_id} registered with scheduler")
```

---

## 7. 集成到现有系统

### 7.1 修改 FullyAsyncRollouter

```python
class FullyAsyncRollouter(SeparateRayPPOTrainer):
    def __init__(...):
        # ...
        
        # 创建 GroupScheduler 引用
        self.scheduler_handle = ray.get_actor("GroupScheduler")
        if self.scheduler_handle is None:
            raise ValueError("GroupScheduler not found")
    
    async def create_rollout_worker_group(self):
        """创建动态 rollout worker group"""
        rollout_config, model_config = _get_rollout_and_model_config(self.config)
        
        # 使用 DynamicvLLMReplica
        self.rollout_replica = DynamicvLLMReplica(
            replica_rank=0,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=self.config.rollout.n_gpus_per_node,
            is_reward_model=False,
            task_id=f"task_{self.config.task_id}",
            scheduler_handle=self.scheduler_handle,
        )
        
        # 初始化但不预分配资源
        await self.rollout_replica.init_standalone(
            scheduler_handle=self.scheduler_handle
        )
```

### 7.2 修改 vLLMHttpServer 支持通信组注入

```python
class vLLMHttpServer:
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
        comm_group_info: Optional[CommGroupInfo] = None,  # 新增
    ):
        # ...
        self.comm_group_info = comm_group_info
        
        if comm_group_info is not None:
            # 原有逻辑：从环境变量读取
            self._master_address = ...
        else:
            # 新逻辑：使用注入的通信组信息
            self._master_address = comm_group_info.master_address
            self._master_port = comm_group_info.master_port
            self._dp_rpc_port = comm_group_info.dp_rpc_port
    
    async def launch_server(self, ...):
        if self.comm_group_info is not None:
            # 原有逻辑：启动分布式
            await self._launch_server_distributed(...)
        else:
            # 新逻辑：使用已初始化的通信组
            await self._launch_server_with_existing_comm(...)
```

---

## 8. 关键实现细节

### 8.1 请求路由

```python
async def generate_with_routing(self, request_id: str, prompt_ids: List[int], ...):
    """
    支持动态实例的生成请求路由
    
    策略：
    1. 检查可用的 idle 实例
    2. 如果没有，等待某个实例变为 idle
    3. 轮询调度确保资源充足
    """
    while True:
        # 查找 idle 实例
        idle_instances = [
            i for i in self.current_instances.values() 
            if i.state == "idle"
        ]
        
        if idle_instances:
            # 选择负载最低的实例
            instance = min(idle_instances, key=lambda i: i.active_requests)
            instance.state = "busy"
            instance.active_requests += 1
            
            try:
                result = await instance.server_handle.generate.remote(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    ...
                )
                return result
            finally:
                instance.state = "idle"
                instance.active_requests -= 1
        
        else:
            # 无可用实例，触发状态更新和等待
            await self._report_state_to_scheduler()
            await asyncio.sleep(0.1)  # 短暂等待
```

### 8.2 实例健康检查

```python
async def _health_check_loop(self):
    """
    周期性健康检查和状态上报
    
    频率：每 5 秒
    """
    while self.running:
        try:
            # 检查实例状态
            for instance_id, instance_info in list(self.current_instances.items()):
                # 检查 server 是否存活
                is_alive = await _check_actor_alive(instance_info.server_handle)
                
                if not is_alive:
                    logger.error(f"Instance {instance_id} dead, recreating...")
                    gpus = await self._destroy_instance(instance_info)
                    del self.current_instances[instance_id]
                    
                    # 通知调度器释放了 GPU
                    if self.scheduler_handle:
                        await self.scheduler_handle.report_state.remote(
                            TaskStateReport(
                                task_id=self.task_id,
                                # ... 其他字段
                                voluntary_reclaim=ReclaimConfirm(
                                    task_id=self.task_id,
                                    reclaimed_instances=1,
                                    reclaimed_gpus=gpus
                                )
                            )
                        )
            
            # 上报状态
            await self._report_state_to_scheduler()
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
        
        await asyncio.sleep(5)
```

### 8.3 优雅关闭

```python
async def shutdown(self):
    """优雅关闭所有实例"""
    logger.info("Shutting down DynamicvLLMReplica...")
    
    # 1. 注销任务
    if self.scheduler_handle:
        await self.scheduler_handle.unregister_task.remote(self.task_id)
    
    # 2. 销毁所有实例
    for instance_info in list(self.current_instances.values()):
        await self._destroy_instance(instance_info)
    
    # 3. 销毁通信组缓存
    await self.comm_group_cache.destroy_all()
    
    logger.info("DynamicvLLMReplica shutdown complete")
```

---

## 9. 配置扩展

### 9.1 RolloutConfig 新增字段

```yaml
rollout:
  # 原有字段
  tensor_model_parallel_size: 8
  pipeline_model_parallel_size: 1
  
  # 新增字段
  min_instances: 1           # 最小保持实例数
  max_instances: 16          # 最大实例数限制
  base_instances: 4           # 调度器基准实例数
  samples_per_round: 1000     # 每轮样本数
  total_samples: 10000        # 总样本数
  
  # 调度器配置
  scheduler:
    scheduler_actor_name: "GroupScheduler"  # 调度器 actor 名
    enable_dynamic_scheduling: true            # 启用动态调度
```

---

## 10. 测试策略

### 10.1 单元测试

1. **测试实例创建/销毁**
   - 创建单个实例
   - 验证 GPU 分配
   - 销毁实例
   - 验证 GPU 释放

2. **测试通信组复用**
   - 创建 TP=2 的实例
   - 创建另一个 TP=2 的实例
   - 验证通信组复用
   - 销毁后验证引用计数

3. **测试 TP 切换**
   - 创建 TP=2 的实例
   - 创建 TP=4 的实例
   - 验证通信组重建

### 10.2 集成测试

1. **模拟调度器**
   - 创建 mock GroupScheduler
   - 测试 reclaim/assign 流程
   - 验证状态上报

2. **压力测试**
   - 频繁增减实例
   - 验证稳定性
   - 监控 GPU 利用率

---

## 11. 性能优化

### 11.1 冷启动优化

```python
class WarmupCache:
    """模型权重预热缓存"""
    
    def __init__(self):
        self.cached_weights = {}  # {model_path: weights}
    
    async def get_weights(self, model_path: str) -> bytes:
        """获取预加载的权重"""
        if model_path not in self.cached_weights:
            # 从共享存储加载
            self.cached_weights[model_path] = await self._load_from_shared(model_path)
        return self.cached_weights[model_path]
```

### 11.2 KV Cache 预分配

```python
async def _create_instance_with_preallocation(self, ...):
    """
    创建实例并预分配 KV Cache
    
    减少首次请求的延迟
    """
    instance_info = await self._create_instance(...)
    
    # 预分配 KV Cache
    await instance_info.server_handle.prefill_kv_cache.remote(
        max_prompts=8,
        max_tokens=2048
    )
    
    return instance_info
```

---

## 12. 监控指标

### 12.1 指标定义

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `vllm_replica.instances.total` | Gauge | 总实例数 |
| `vllm_replica.instances.idle` | Gauge | 空闲实例数 |
| `vllm_replica.instances.busy` | Gauge | 忙实例数 |
| `vllm_replica.gpus.allocated` | Gauge | 已分配 GPU 数 |
| `vllm_replica.gpus利用率` | Gauge | GPU 利用率 |
| `vllm_replica.comm_group.cache_hit_rate` | Gauge | 通信组缓存命中率 |
| `vllm_replica.scheduling.latency_ms` | Histogram | 调度延迟 |
| `vllm_replica.instance.creation_time_ms` | Histogram | 实例创建耗时 |
| `vllm_replica.instance.destruction_time_ms` | Histogram | 实例销毁耗时 |

---

## 13. 错误处理

### 13.1 常见错误

1. **GPU 分配失败**
   - 检测到 GPU 不可用
   - 通知调度器主动释放
   - 重新注册任务

2. **实例创建超时**
   - 设置超时机制（默认 60 秒）
   - 超时后销毁并重试

3. **通信组初始化失败**
   - 记录详细错误信息
   - 尝试清理部分资源
   - 触发重建流程

### 13.2 重试策略

```python
@retry(max_attempts=3, backoff_factor=2)
async def _create_instance_with_retry(self, ...):
    """带重试的实例创建"""
    pass
```

---

## 14. 向后兼容

### 14.1 降级策略

当调度器不可用时：

```python
async def init_standalone(self, scheduler_handle: ray.ActorHandle = None):
    if scheduler_handle is None:
        logger.warning("Scheduler not available, falling back to static mode")
        # 降级到原有的固定分配模式  
        await super().init_standalone()
        return
    
    # 启用动态调度
    await self._init_dynamic_mode(scheduler_handle)
```

---

## 15. 实施步骤

### Phase 1: 基础设施 (1-2 周)
1. 实现通信组缓存
2. 实现任务状态跟踪器
3. 添加监控指标

### Phase 2: 核心功能 (2-3 周)
1. 实现 DynamicvLLMReplica 类
2. 实现 reclaim/assign 接口
3. 实现实例创建/销毁

### Phase 3: 集成 (1-2 周)
1. 修改 vLLMHttpServer
2. 修改 FullyAsyncRollouter
3. 集成到现有系统

### Phase 4: 测试优化 (1-2 周)
1. 单元测试
2. 集成测试
3. 性能优化

### Phase 5: 部署验证 (1 周)
1. 灰度环境测试
2. 生产环境小规模验证
3. 监控和告警配置
