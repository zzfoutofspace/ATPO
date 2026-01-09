# 离线树搜索重构计划

## 概述
将 `vllm_rollout_with_tools_tree_offline.py` 从**在线树搜索**（边rollout边分支）改造为**离线树搜索**（先构建树结构，再进行搜索）。

## 核心区别

### 当前实现（在线树搜索）
```
初始化分支 → 并行rollout → 实时决策分支 → 继续rollout → ...
```
- 分支决策基于当前熵值
- 新分支立即加入下一轮生成
- 状态用列表管理（curr_inputs, result_masks等）

### 目标实现（离线树搜索）
```
1. 构建初始树 → 2. 生成初始链 → 3. 迭代扩展树 → 4. 采样最终叶子
```
- 先完整构建树结构
- 使用TreeNode类管理节点
- 分阶段执行：初始化 → 扩展 → 采样

---

## 步骤1：添加TreeNode类

在文件开头添加（参考generation_ts.py的TreeNode）：

```python
import uuid
from typing import Optional

class ToolTreeNode:
    """Tree node for managing tool-based generation with tree search."""
    
    def __init__(
        self,
        tree_uid: str,
        node_uid: str,
        prompt_token_ids: List[int],
        curr_token_ids: List[int],
        result_mask: List[int],
        call_counter: int = 0,
        parent_node: Optional['ToolTreeNode'] = None,
        is_root: bool = False,
        is_active: bool = True,
        is_leaf: bool = False,
        depth: int = 0,
        entropy: float = 0.0,
    ):
        self.tree_uid = tree_uid
        self.node_uid = node_uid
        
        # Token sequences
        self.prompt_token_ids = prompt_token_ids
        self.curr_token_ids = curr_token_ids.copy()
        self.result_mask = result_mask.copy()
        
        # Tool call tracking
        self.call_counter = call_counter
        
        # Tree structure
        self.parent_node = parent_node
        self._child_nodes = []
        
        # Node status
        self.is_root = is_root
        self.is_active = is_active
        self.is_leaf = is_leaf
        self.depth = depth
        
        # Entropy tracking
        self.entropy = entropy
        self.initial_entropy = entropy if is_root else 0.0
        
    def add_child(self, child_node: 'ToolTreeNode'):
        """Add a child node."""
        if child_node is self or child_node.node_uid == self.node_uid:
            raise ValueError("A node cannot be its own child")
        self._child_nodes.append(child_node)
    
    @property
    def child_nodes(self):
        return self._child_nodes
    
    def get_subtree_nodes(self):
        """Get all descendant nodes."""
        nodes = []
        nodes_to_visit = list(self._child_nodes)
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            nodes.append(current_node)
            nodes_to_visit.extend(current_node._child_nodes)
        return nodes
    
    def get_all_leaves(self):
        """Get all leaf nodes."""
        leaves = []
        if self.is_leaf:
            leaves.append(self)
        for node in self.get_subtree_nodes():
            if node.is_leaf:
                leaves.append(node)
        return leaves
    
    def get_non_leaf_nodes(self):
        """Get all non-leaf nodes that can be expanded."""
        non_leaves = []
        if not self.is_leaf and self.is_active:
            non_leaves.append(self)
        for node in self.get_subtree_nodes():
            if not node.is_leaf and node.is_active:
                non_leaves.append(node)
        return non_leaves
    
    def sample_expansion_nodes(self, n: int, mode: str = 'random'):
        """Sample n nodes for expansion."""
        candidate_nodes = self.get_non_leaf_nodes()
        if not candidate_nodes:
            return []
        
        if len(candidate_nodes) >= n:
            return random.sample(candidate_nodes, n)
        else:
            return random.choices(candidate_nodes, k=n)
    
    def sample_leaves(self, n: int):
        """Sample n leaf nodes."""
        all_leaves = self.get_all_leaves()
        if len(all_leaves) < n:
            # Duplicate if needed
            result = all_leaves.copy()
            while len(result) < n:
                result.extend(all_leaves[:min(n - len(result), len(all_leaves))])
            return result[:n]
        return random.sample(all_leaves, n)
    
    def create_branch(self, branch_id: int = 0):
        """Create a new branch from this node."""
        child_uid = f"{self.node_uid}_b{branch_id}"
        child_node = ToolTreeNode(
            tree_uid=self.tree_uid,
            node_uid=child_uid,
            prompt_token_ids=self.prompt_token_ids,
            curr_token_ids=self.curr_token_ids.copy(),
            result_mask=self.result_mask.copy(),
            call_counter=self.call_counter,
            parent_node=self,
            is_root=False,
            is_active=True,
            is_leaf=False,
            depth=self.depth,
            entropy=self.entropy,
        )
        self.add_child(child_node)
        return child_node
```

---

## 步骤2：添加辅助方法

在`vLLMRolloutWithTools`类中添加以下方法：

### 2.1 构建初始树
```python
def _build_initial_trees(self, prompt_token_ids_list: List[List[int]], 
                        initial_rollouts_list: List[int]) -> List[ToolTreeNode]:
    """Build initial tree structures with root nodes and initial branches."""
    root_nodes = []
    
    for i, prompt_ids in enumerate(prompt_token_ids_list):
        tree_uid = str(uuid.uuid4())
        node_uid = f"tree_{i}_root"
        
        # Create root node
        root_node = ToolTreeNode(
            tree_uid=tree_uid,
            node_uid=node_uid,
            prompt_token_ids=prompt_ids,
            curr_token_ids=prompt_ids.copy(),
            result_mask=[],
            call_counter=0,
            parent_node=None,
            is_root=True,
            is_active=True,
            is_leaf=False,
            depth=0,
            entropy=0.0,
        )
        
        # Create initial branches from root
        initial_rollouts = initial_rollouts_list[i]
        for branch_idx in range(initial_rollouts):
            child_node = root_node.create_branch(branch_idx)
        
        root_nodes.append(root_node)
    
    return root_nodes
```

### 2.2 树扩展迭代
```python
def _expand_tree_iteration(self, root_nodes: List[ToolTreeNode], 
                          num_expansions: int, beam_size: int) -> List[ToolTreeNode]:
    """Perform one iteration of tree expansion."""
    expansion_nodes = []
    
    for root in root_nodes:
        # Sample nodes to expand from this tree
        nodes_to_expand = root.sample_expansion_nodes(num_expansions, mode='random')
        expansion_nodes.extend(nodes_to_expand)
    
    # Create branches for each expansion node
    new_nodes = []
    for node in expansion_nodes:
        # Create beam_size - 1 new branches
        for branch_idx in range(beam_size - 1):
            new_node = node.create_branch(branch_idx)
            new_nodes.append(new_node)
    
    return expansion_nodes + new_nodes  # Return all nodes to process
```

### 2.3 处理生成输出
```python
def _process_generation_outputs(self, outputs, active_nodes: List[ToolTreeNode], 
                               eos_token_id: int, max_len: int) -> Dict:
    """Process generation outputs and update node states."""
    tool_requests = {tag: [] for tag in self.tools}
    vocab_size = len(self.tokenizer.get_vocab())
    entropy_norm_factor = math.log(vocab_size)
    
    for i, node in enumerate(active_nodes):
        output = outputs[i]
        generated_tokens = output.outputs[0].token_ids
        
        # Update node with generated tokens
        node.curr_token_ids.extend(generated_tokens)
        node.result_mask.extend([1] * len(generated_tokens))
        
        # Calculate entropy
        logprobs = []
        tokens = output.outputs[0].token_ids
        for j in range(min(20, len(tokens))):
            try:
                logprob_info = output.outputs[0].logprobs[j]
            except Exception:
                logprob_info = output.outputs[0].logprobs[-1]
            token_list = list(logprob_info.values())
            token_logprobs = [token.logprob for token in token_list]
            logprobs.extend(token_logprobs)
        
        if logprobs:
            node.entropy = self._calc_entropy(logprobs) / entropy_norm_factor
        
        # Store initial entropy
        if node.is_root or (node.parent_node and node.parent_node.is_root):
            if node.initial_entropy == 0.0:
                node.initial_entropy = node.entropy
        
        # Check finish reason
        finish_reason = output.outputs[0].finish_reason
        stop_reason = output.outputs[0].stop_reason
        is_tool_call = finish_reason == "stop" and stop_reason in self.stop_sequences
        
        if is_tool_call:
            tag = stop_reason.strip("</>")
            if node.call_counter < self.tool_call_limit:
                node.call_counter += 1
                full_text = self.tokenizer.decode(node.curr_token_ids)
                content = self._extract_content(full_text, tag)
                if content:
                    tool_requests[tag].append({"node": node, "content": content})
            else:
                node.curr_token_ids.append(eos_token_id)
                node.result_mask.append(1)
                node.is_active = False
                node.is_leaf = True
        
        elif finish_reason == "length":
            response_len = len(node.curr_token_ids) - len(node.prompt_token_ids)
            if response_len >= max_len:
                node.is_active = False
                node.is_leaf = True
        
        elif finish_reason == "stop":
            node.is_active = False
            node.is_leaf = True
    
    return tool_requests
```

### 2.4 执行工具请求
```python
def _execute_tool_requests(self, tool_requests: Dict, tool_metrics: Dict, 
                          calls_per_tool: Counter, success_per_tool: Counter,
                          total_time_per_tool: Counter):
    """Execute tool requests in parallel and update nodes."""
    if not any(tool_requests.values()):
        return
    
    total_requests = sum(len(reqs) for reqs in tool_requests.values())
    logger.info(f"Processing {total_requests} tool requests...")
    
    futures = {}
    for tag, requests in tool_requests.items():
        if not requests:
            continue
        tool = self.tools[tag]
        for req in requests:
            future = self.executor.submit(
                self._execute_tool_with_retry, tool, req["content"]
            )
            futures[future] = {"node": req["node"], "tag": tag}
    
    # Wait for completion
    for future in concurrent.futures.as_completed(futures):
        fut_info = futures[future]
        node = fut_info["node"]
        tag = fut_info["tag"]
        
        try:
            result = future.result(timeout=self.tool_timeout)
            success = result["success"]
            retry_count = result["retry_count"]
            execution_time = result["execution_time"]
            result_text = result["result"]
            
            # Update metrics
            tool_metrics["tools/total_calls"] += 1
            calls_per_tool[tag] += 1
            
            if success:
                tool_metrics["tools/successful_calls"] += 1
                success_per_tool[tag] += 1
            else:
                tool_metrics["tools/failed_calls"] += 1
                result_text = f"Tool({tag}) returned empty output."
            
            tool_metrics["tools/total_execution_time"] += execution_time
            tool_metrics["tools/max_execution_time"] = max(
                tool_metrics["tools/max_execution_time"], execution_time
            )
            tool_metrics["tools/total_retries"] += retry_count
            tool_metrics["tools/max_retries"] = max(
                tool_metrics["tools/max_retries"], retry_count
            )
            total_time_per_tool[tag] += execution_time
            
            # Add tool result to node
            formatted_result = f" <result>\n{result_text}\n</result>"
            result_tokens = self.tokenizer.encode(formatted_result)
            node.curr_token_ids.extend(result_tokens)
            node.result_mask.extend([0] * len(result_tokens))
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            tool_metrics["tools/failed_calls"] += 1
            result_text = f"Error: Tool({tag}) execution failed"
            formatted_result = f" <result>\n{result_text}\n</result>"
            result_tokens = self.tokenizer.encode(formatted_result)
            node.curr_token_ids.extend(result_tokens)
            node.result_mask.extend([0] * len(result_tokens))
```

---

## 步骤3：重构generate_sequences方法

将原来的主循环替换为以下结构：

```python
@GPUMemoryLogger(role="vllm rollout spmd with tools", logger=logger)
@torch.no_grad()
def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    # ... 初始化代码保持不变 ...
    
    with self.update_sampling_params(**kwargs):
        num_samples = self.sampling_params.n
        prompt_token_ids_list = [
            _pre_process_inputs(self.pad_token_id, prompt) for prompt in input_ids
        ]
        
        # Calculate initial rollouts
        if self.enable_dynamic_rollouts:
            initial_rollouts_list = self._calculate_initial_rollouts_dynamical(prompts, **kwargs)
        else:
            initial_rollouts_list = [max(1, min(self.initial_rollouts, num_samples))] * len(prompt_token_ids_list)
        
        max_len = self.config.response_length
        
        # ===== PHASE 1: BUILD INITIAL TREES =====
        logger.info("=" * 60)
        logger.info("PHASE 1: BUILD INITIAL TREE STRUCTURES")
        logger.info("=" * 60)
        
        root_nodes = self._build_initial_trees(prompt_token_ids_list, initial_rollouts_list)
        logger.info(f"Created {len(root_nodes)} root nodes with initial branches")
        
        # ===== PHASE 2: GENERATE INITIAL ACTION CHAINS =====
        logger.info("=" * 60)
        logger.info("PHASE 2: GENERATE INITIAL ACTION CHAINS")
        logger.info("=" * 60)
        
        iteration = 0
        active_nodes = []
        for root in root_nodes:
            active_nodes.extend(root.child_nodes)
        
        while active_nodes:
            iteration += 1
            logger.info(f"--- Iteration {iteration}: {len(active_nodes)} active nodes ---")
            
            # Prepare prompts
            active_prompts = [node.curr_token_ids for node in active_nodes]
            
            # Calculate max_tokens
            max_tokens_list = [
                max(1, max_len - (len(node.curr_token_ids) - len(node.prompt_token_ids)))
                for node in active_nodes
            ]
            max_tokens = max(max_tokens_list)
            
            # Generate
            with self.update_sampling_params(
                n=1,
                stop=self.stop_sequences,
                max_tokens=max_tokens,
                detokenize=True,
                logprobs=self.logprobs,
            ):
                outputs = self.inference_engine.generate(
                    prompt_token_ids=active_prompts,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
            
            # Process outputs
            tool_requests = self._process_generation_outputs(
                outputs, active_nodes, eos_token_id, max_len
            )
            
            # Execute tools
            self._execute_tool_requests(
                tool_requests, tool_metrics, calls_per_tool,
                success_per_tool, total_time_per_tool
            )
            
            # Update active nodes
            active_nodes = [node for node in active_nodes if node.is_active and not node.is_leaf]
        
        # ===== PHASE 3: TREE EXPANSION =====
        logger.info("=" * 60)
        logger.info("PHASE 3: TREE EXPANSION")
        logger.info("=" * 60)
        
        num_expansion_iterations = getattr(self.config, 'expansion_iterations', 2)
        for exp_iter in range(num_expansion_iterations):
            logger.info(f"=== Expansion Iteration {exp_iter + 1}/{num_expansion_iterations} ===")
            
            # Select nodes and create branches
            nodes_to_process = self._expand_tree_iteration(
                root_nodes, num_expansions=1, beam_size=beam_size
            )
            
            if not nodes_to_process:
                logger.info("No nodes available for expansion")
                break
            
            # Generate for expanded nodes
            iteration = 0
            active_nodes = [node for node in nodes_to_process if node.is_active and not node.is_leaf]
            
            while active_nodes:
                iteration += 1
                logger.info(f"  Expansion {exp_iter+1}, Iteration {iteration}: {len(active_nodes)} nodes")
                
                active_prompts = [node.curr_token_ids for node in active_nodes]
                max_tokens_list = [
                    max(1, max_len - (len(node.curr_token_ids) - len(node.prompt_token_ids)))
                    for node in active_nodes
                ]
                max_tokens = max(max_tokens_list)
                
                with self.update_sampling_params(
                    n=1,
                    stop=self.stop_sequences,
                    max_tokens=max_tokens,
                    detokenize=True,
                    logprobs=self.logprobs,
                ):
                    outputs = self.inference_engine.generate(
                        prompt_token_ids=active_prompts,
                        sampling_params=self.sampling_params,
                        use_tqdm=False,
                    )
                
                tool_requests = self._process_generation_outputs(
                    outputs, active_nodes, eos_token_id, max_len
                )
                
                self._execute_tool_requests(
                    tool_requests, tool_metrics, calls_per_tool,
                    success_per_tool, total_time_per_tool
                )
                
                active_nodes = [node for node in active_nodes if node.is_active and not node.is_leaf]
        
        # ===== PHASE 4: COLLECT FINAL OUTPUTS =====
        logger.info("=" * 60)
        logger.info("PHASE 4: COLLECT FINAL OUTPUTS")
        logger.info("=" * 60)
        
        output_sequences = []
        output_result_masks = []
        
        for root in root_nodes:
            sampled_leaves = root.sample_leaves(num_samples)
            
            for leaf in sampled_leaves:
                response_tokens = leaf.curr_token_ids[len(leaf.prompt_token_ids):]
                output_sequences.append(response_tokens)
                output_result_masks.append(leaf.result_mask)
        
        # ... 后续处理代码保持不变 ...
```

---

## 步骤4：配置参数

在配置文件中添加：

```yaml
rollout:
  expansion_iterations: 2  # 树扩展迭代次数
  initial_rollouts: 3      # 初始分支数（如果不使用动态计算）
  beam_size: 2             # 每次扩展的分支数
```

---

## 关键改进

### 1. **清晰的阶段划分**
- Phase 1: 构建树结构
- Phase 2: 生成初始链
- Phase 3: 迭代扩展
- Phase 4: 采样输出

### 2. **节点管理**
- 使用TreeNode类管理状态
- 清晰的父子关系
- 便于追踪和调试

### 3. **可扩展性**
- 易于添加新的扩展策略
- 可以实现更复杂的采样方法
- 支持树剪枝和优化

### 4. **代码可维护性**
- 模块化的辅助方法
- 清晰的职责分离
- 易于测试和调试

---

## 测试建议

1. **单元测试**
   - TreeNode类的基本功能
   - 树构建和扩展逻辑
   - 节点采样方法

2. **集成测试**
   - 完整的生成流程
   - 工具调用集成
   - 多样本批处理

3. **性能测试**
   - 与原实现对比
   - 内存使用情况
   - 生成速度

---

## 迁移检查清单

- [ ] 添加ToolTreeNode类
- [ ] 实现_build_initial_trees方法
- [ ] 实现_expand_tree_iteration方法
- [ ] 实现_process_generation_outputs方法
- [ ] 实现_execute_tool_requests方法
- [ ] 重构generate_sequences主循环
- [ ] 更新配置文件
- [ ] 添加单元测试
- [ ] 进行集成测试
- [ ] 性能对比测试
- [ ] 更新文档

---

## 注意事项

1. **向后兼容性**：保留原文件作为备份
2. **渐进式迁移**：可以先实现基本功能，再优化
3. **日志记录**：保持详细的日志以便调试
4. **错误处理**：确保异常情况下的正确行为
