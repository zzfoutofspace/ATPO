import concurrent.futures
import logging
import os
from collections import Counter
from copy import deepcopy
from typing import List, Dict, Any, Tuple
import random

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps

from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import pad_2d_list_to_length
from verl.workers.agent.base import BaseAgent
from verl.workers.agent.tools.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class AgentMetrics:
    """A helper class to manage and update metrics during the agent rollout."""

    def __init__(self, tool_tags: List[str]):
        self.metrics = {
            "tools/total_calls": 0,
            "tools/successful_calls": 0,
            "tools/failed_calls": 0,
            "tools/total_execution_time": 0.0,
            "tools/avg_execution_time": 0.0,
            "tools/max_execution_time": 0.0,
            "tools/max_retries": 0,
            "tools/total_retries": 0,
            "tools/call_limit_reached_count": 0,
        }
        self.calls_per_tool = Counter()
        self.success_per_tool = Counter()
        self.total_time_per_tool = Counter()
        self.tool_tags = tool_tags

    def update_on_tool_request(self, tag: str):
        self.metrics["tools/total_calls"] += 1
        self.calls_per_tool[tag] += 1

    def update_on_tool_result(self, tag: str, result: Dict[str, Any]):
        success = result["success"]
        execution_time = result["execution_time"]
        retry_count = result["retry_count"]

        if success:
            self.metrics["tools/successful_calls"] += 1
            self.success_per_tool[tag] += 1
        else:
            self.metrics["tools/failed_calls"] += 1

        self.metrics["tools/total_execution_time"] += execution_time
        self.metrics["tools/max_execution_time"] = max(self.metrics["tools/max_execution_time"], execution_time)
        self.metrics["tools/total_retries"] += retry_count
        self.metrics["tools/max_retries"] = max(self.metrics["tools/max_retries"], retry_count)
        self.total_time_per_tool[tag] += execution_time

    def update_on_limit_reached(self):
        self.metrics["tools/call_limit_reached_count"] += 1

    def finalize(self) -> Dict[str, Any]:
        if self.metrics["tools/total_calls"] > 0:
            self.metrics["tools/avg_execution_time"] = \
                self.metrics["tools/total_execution_time"] / self.metrics["tools/total_calls"]

        for tag in self.tool_tags:
            calls = self.calls_per_tool[tag]
            self.metrics[f"tools/{tag}/calls"] = calls
            if calls > 0:
                self.metrics[f"tools/{tag}/avg_time"] = self.total_time_per_tool[tag] / calls
                self.metrics[f"tools/{tag}/success_rate"] = self.success_per_tool[tag] / calls
            else:
                self.metrics[f"tools/{tag}/avg_time"] = 0
                self.metrics[f"tools/{tag}/success_rate"] = 0
        return self.metrics


class ToolAgent(BaseAgent):
    """
    A stateful agent that interacts with an environment over discrete steps,
    using tools to accomplish tasks. It aligns with the OpenAI Gym interface.
    """

    def __init__(self, config: DictConfig, vllm_engine: LLM, tokenizer: Any, **kwargs):
        super().__init__(config, vllm_engine, tokenizer, **kwargs)
        tools_config = self.agent_config.get("tools", {})
        self.tool_executor = ToolExecutor(tools_config)
        if not self.tool_executor.has_tools():
            raise ValueError("ToolAgent requires tools to be configured.")

        # Distributed settings
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()
        self.tp_group = vllm_ps.get_tensor_model_parallel_group()

        # Static parameters
        self.tool_call_limit = tools_config.get("call_limit", 5)
        self.max_tool_workers = tools_config.get("max_workers", 64)
        self.tool_timeout = tools_config.get("timeout", 120)
        self.max_len = self.config.response_length
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        # Beam search parameters
        self.initial_rollouts = self.config.get("initial_rollouts", 1)
        self.beam_size = self.config.get("beam_size", 1)
        self.branch_probability = self.config.get("branch_probability", 0.5)

        # State variables, initialized in `reset`
        self.sampling_params = None
        self.batch_size = 0
        self.device = None
        self.prompts_len = None
        self.curr_inputs = None
        self.init_inputs = None
        self.result_masks = None
        self.call_counters = None
        self.dones = None
        self.metrics_tracker = None
        self.active_indices = None
        self.original_batch_size = 0
        self.rollouts_per_sample = None
        self.sample_to_indices = None

    def reset(self, prompts: DataProto, sampling_params: SamplingParams) -> DataProto:
        self.sampling_params = sampling_params
        num_samples = self.sampling_params.n

        # Get beam search params from agent_config
        self.initial_rollouts = self.config.get("initial_rollouts", num_samples)
        self.beam_size = self.config.get("beam_size", 1)
        self.branch_probability = self.config.get("branch_probability", 0.5)
        self.initial_rollouts = min(self.initial_rollouts, num_samples)

        input_ids = prompts.batch['input_ids']
        self.original_batch_size = input_ids.size(0)
        self.device = input_ids.device

        if 'raw_prompt_ids' not in prompts.non_tensor_batch:
            raise ValueError(
                "`raw_prompt_ids` not found in `prompts.non_tensor_batch`. "
                "It should be pre-processed and added by the calling rollout worker."
            )
        raw_prompt_ids = prompts.non_tensor_batch.pop('raw_prompt_ids')

        self.curr_inputs = []
        self.init_inputs = []
        self.result_masks = []
        self.prompts_len = []

        for ids in raw_prompt_ids:
            for _ in range(self.initial_rollouts):
                self.curr_inputs.append(ids.copy())
                self.init_inputs.append(ids.copy())
                self.prompts_len.append(len(ids))
                self.result_masks.append([])

        self.batch_size = len(self.curr_inputs)
        self.call_counters = torch.zeros(self.batch_size, dtype=torch.int)
        self.dones = torch.zeros(self.batch_size, dtype=torch.bool)
        self.metrics_tracker = AgentMetrics(list(self.tool_executor.tools.keys()))
        self.active_indices = list(range(self.batch_size))

        # Tracking for beam search
        self.rollouts_per_sample = [self.initial_rollouts] * self.original_batch_size
        self.sample_to_indices = {
            i: list(range(i * self.initial_rollouts, (i + 1) * self.initial_rollouts))
            for i in range(self.original_batch_size)
        }


    @torch.no_grad()
    def step(self) -> torch.Tensor:
        if not self.active_indices:
            # If all episodes are done, return the final dones state.
            return self.dones

        # --- 1. Generate with vLLM ---
        max_tokens = max(1, max(self.max_len - (len(self.curr_inputs[i]) - self.prompts_len[i]) for i in self.active_indices))

        current_sampling_params = deepcopy(self.sampling_params)
        current_sampling_params.n = 1
        current_sampling_params.detokenize = True
        
        # Combine original and tool stop sequences
        original_stop = current_sampling_params.stop
        tool_stop = self.tool_executor.stop_sequences
        combined_stop = set(tool_stop)
        if original_stop:
            if isinstance(original_stop, str):
                combined_stop.add(original_stop)
            else:
                combined_stop.update(original_stop)
        current_sampling_params.stop = list(combined_stop)

        current_sampling_params.max_tokens = max_tokens

        active_prompts = [self.curr_inputs[i] for i in self.active_indices]
        outputs = self.vllm_engine.generate(
            prompt_token_ids=active_prompts,
            sampling_params=current_sampling_params,
            use_tqdm=False
        )

        # --- 2. Process outputs and prepare tool calls ---
        tool_requests = {tag: [] for tag in self.tool_executor.tools}
        next_active_indices = []

        for i, out_idx in enumerate(self.active_indices):
            output = outputs[i].outputs[0]
            generated_tokens = output.token_ids
            self.curr_inputs[out_idx].extend(generated_tokens)
            self.result_masks[out_idx].extend([1] * len(generated_tokens))

            is_tool_call = output.finish_reason == 'stop' and output.stop_reason in self.tool_executor.stop_sequences

            if is_tool_call:
                if self.call_counters[out_idx] < self.tool_call_limit:
                    tag = output.stop_reason.strip("</>")
                    full_text = self.tokenizer.decode(self.curr_inputs[out_idx])
                    content = self.tool_executor.extract_content(full_text, tag)
                    if content:
                        tool_requests[tag].append({"index": out_idx, "content": content})
                        next_active_indices.append(out_idx)
                    # If content is not extracted, it's considered finished and will not be in next_active_indices.
                    self.call_counters[out_idx] += 1
                else:
                    self.metrics_tracker.update_on_limit_reached()
            elif output.finish_reason == 'length':
                # If stopped due to length, but still within max_len, continue.
                response_len = len(self.curr_inputs[out_idx]) - self.prompts_len[out_idx]
                if response_len < self.max_len:
                    next_active_indices.append(out_idx)
            # If 'stop' due to EOS, or any other case, it's finished and not added to next_active_indices.

        # --- 3. Execute tools and broadcast results ---
        broadcast_data = {}
        if self.tp_rank == 0:
            tool_results_for_broadcast = []
            if any(tool_requests.values()):
                # Update metrics for requests on rank 0
                for tag, requests in tool_requests.items():
                    for _ in requests:
                        self.metrics_tracker.update_on_tool_request(tag)

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_tool_workers) as executor:
                    futures = {
                        executor.submit(self.tool_executor.execute_with_retry, self.tool_executor.get_tool(tag), req["content"]):
                            {"index": req["index"], "tag": tag}
                        for tag, requests in tool_requests.items() for req in requests
                    }
                    for future in concurrent.futures.as_completed(futures):
                        info = futures[future]
                        idx, tag = info["index"], info["tag"]
                        try:
                            result = future.result(timeout=self.tool_timeout)
                            self.metrics_tracker.update_on_tool_result(tag, result)
                            result_text = result["result"]
                        except Exception as e:
                            logger.error(f"Tool execution future failed for sample {idx}: {e}")
                            result_text = f"Error: Tool execution failed."
                            self.metrics_tracker.update_on_tool_result(tag, {"success": False, "execution_time": 0, "retry_count": 0})

                        formatted_result = f" <result>\n{result_text}\n</result>"
                        result_tokens = self.tokenizer.encode(formatted_result, add_special_tokens=False)
                        tool_results_for_broadcast.append({"idx": idx, "tokens": result_tokens})
            broadcast_data['tool_results'] = tool_results_for_broadcast

        # Broadcast results from rank 0 to all other ranks
        broadcast_list = [broadcast_data]
        dist.broadcast_object_list(broadcast_list, src=0, group=self.tp_group)
        broadcast_data = broadcast_list[0]

        # All ranks apply the results to maintain consistent state
        tool_execution_results = broadcast_data.get('tool_results', [])
        for res in tool_execution_results:
            idx = res['idx']
            result_tokens = res['tokens']
            self.curr_inputs[idx].extend(result_tokens)
            self.result_masks[idx].extend([0] * len(result_tokens))

        # --- 4. Update state and prepare outputs ---
        # Final check on active indices based on length after tool results are added
        final_active_indices = []
        for idx in next_active_indices:
            response_len = len(self.curr_inputs[idx]) - self.prompts_len[idx]
            if response_len < self.max_len:
                final_active_indices.append(idx)

        # Update dones for newly finished trajectories before branching
        newly_finished_indices = set(self.active_indices) - set(final_active_indices)
        for idx in newly_finished_indices:
            self.dones[idx] = True

        # --- 5. Beam Branching ---
        num_samples = self.sampling_params.n
        new_inputs, new_init_inputs, new_result_masks, new_call_counters_list, new_sample_origins = [], [], [], [], []

        print(f"--- Beam Branching [Step Start] ---")
        print(f"Config: num_samples(n)={num_samples}, beam_size={self.beam_size}, branch_prob={self.branch_probability}")
        print(f"State before branching: final_active_indices={final_active_indices}, rollouts_per_sample={self.rollouts_per_sample}")

        active_by_sample = {}
        for idx in final_active_indices:
            orig_sample = -1
            for sample_idx, indices in self.sample_to_indices.items():
                if idx in indices:
                    orig_sample = sample_idx
                    break
            if orig_sample != -1:
                if orig_sample not in active_by_sample:
                    active_by_sample[orig_sample] = []
                active_by_sample[orig_sample].append(idx)

        # For each original sample, create additional branches up to beam_size
        for orig_sample, active_idxs in active_by_sample.items():
            remaining_slots = num_samples - self.rollouts_per_sample[orig_sample]
            if remaining_slots <= 0:
                continue
            
            print(f"  Branching for active sample {orig_sample}: active_idxs={active_idxs}, remaining_slots={remaining_slots}")
            branches_created = 0
            for source_idx in active_idxs:
                branches_per_idx = min(self.beam_size - 1, remaining_slots - branches_created)
                if branches_per_idx <= 0:
                    break
                for _ in range(branches_per_idx):
                    if random.random() > self.branch_probability:
                        continue
                    print(f"    Creating new branch from source_idx {source_idx} for orig_sample {orig_sample}")
                    new_inputs.append(self.curr_inputs[source_idx].copy())
                    new_init_inputs.append(self.init_inputs[source_idx].copy())
                    new_result_masks.append(self.result_masks[source_idx].copy())
                    new_call_counters_list.append(self.call_counters[source_idx].item())
                    new_sample_origins.append(orig_sample)
                    self.rollouts_per_sample[orig_sample] += 1
                    branches_created += 1

        # Add non-active samples that still need more rollouts 
        for orig_sample in range(self.original_batch_size):
            if orig_sample not in active_by_sample and self.rollouts_per_sample[orig_sample] < num_samples:
                branches_to_add = min(1, num_samples - self.rollouts_per_sample[orig_sample])
                if branches_to_add <= 0: continue
                
                print(f"  Restarting for finished sample {orig_sample}: need to add {branches_to_add} new rollouts.")
                source_idx = self.sample_to_indices[orig_sample][0]
                for _ in range(branches_to_add):
                    print(f"    Restarting rollout from init_state for orig_sample {orig_sample}")
                    new_inputs.append(self.init_inputs[source_idx].copy())
                    new_init_inputs.append(self.init_inputs[source_idx].copy())
                    new_result_masks.append([])
                    new_call_counters_list.append(0)
                    new_sample_origins.append(orig_sample)
                    self.rollouts_per_sample[orig_sample] += 1

        if new_inputs:
            print(f"  Total new branches/rollouts created: {len(new_inputs)}")
            start_idx = self.batch_size
            self.curr_inputs.extend(new_inputs)
            self.init_inputs.extend(new_init_inputs)
            self.result_masks.extend(new_result_masks)
            self.prompts_len.extend([len(p) for p in new_init_inputs])

            new_indices = list(range(start_idx, start_idx + len(new_inputs)))
            final_active_indices.extend(new_indices)

            for i, new_idx in enumerate(new_indices):
                orig_sample = new_sample_origins[i]
                self.sample_to_indices.setdefault(orig_sample, []).append(new_idx)

            new_call_counters_tensor = torch.tensor(new_call_counters_list, dtype=torch.int)
            self.call_counters = torch.cat([self.call_counters, new_call_counters_tensor])
            self.dones = torch.cat([self.dones, torch.zeros(len(new_inputs), dtype=torch.bool)])
            self.batch_size = len(self.curr_inputs)

        self.active_indices = final_active_indices
        print(f"State after branching: active_indices={self.active_indices}, rollouts_per_sample={self.rollouts_per_sample}")
        print(f"--- Beam Branching [Step End] ---")

        return self.dones

    def get_final_responses(self) -> Dict[str, Any]:
        output_sequences, output_result_masks = [], []
        num_samples = self.sampling_params.n

        # rearrange the responses to the original order
        for i in range(self.original_batch_size):
            sample_indices = self.sample_to_indices.get(i, [])
            selected_indices = sample_indices[:num_samples]

            assert len(selected_indices) == num_samples, f"len(selected_indices): {len(selected_indices)} != num_samples: {num_samples}"

            for idx in selected_indices:
                response_tokens = self.curr_inputs[idx][self.prompts_len[idx]:]
                response_mask = self.result_masks[idx]

                # Truncate if over max length and replace the last token with EOS
                if len(response_tokens) > self.max_len:
                    response_tokens = response_tokens[:self.max_len]
                    response_mask = response_mask[:self.max_len]
                    if len(response_tokens) > 0:
                        response_tokens[-1] = self.eos_token_id
                        response_mask[-1] = 1  # Ensure mask is 1 for the EOS token

                output_sequences.append(response_tokens)
                output_result_masks.append(response_mask)

        padded_responses = pad_2d_list_to_length(output_sequences, self.pad_token_id, self.max_len).to(self.device)
        padded_masks = pad_2d_list_to_length(output_result_masks, 0, self.max_len).to(self.device)

        print(f"padded_responses(shape: {padded_responses.shape}): {padded_responses}")
        print(f"padded_masks(shape: {padded_masks.shape}): {padded_masks}")

        return {
            "responses": padded_responses,
            "loss_mask": padded_masks,
            "meta_info": {"metrics": self.metrics_tracker.finalize()}
        }
