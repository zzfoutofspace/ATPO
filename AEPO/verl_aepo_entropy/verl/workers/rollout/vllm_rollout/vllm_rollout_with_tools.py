# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import importlib
import logging
import os
import time
import random
from copy import deepcopy
from typing import Dict, List, Counter

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.tools.base_tool import BaseTool
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
    vLLMRollout,
    _pre_process_inputs,
    _repeat_interleave,
)
import math

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _load_tool_from_config(tool_config: DictConfig) -> BaseTool:
    """Dynamically loads a tool from its configuration."""
    module_path, class_name = tool_config.class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)

        tool_class = getattr(module, class_name)

        tool_params = OmegaConf.to_container(
            tool_config.get("params", {}), resolve=True
        )

        tool_instance = tool_class(**tool_params)

        return tool_instance
    except ImportError as e:
        logger.error(f"Failed to import module {module_path}: {e}")
        raise
    except AttributeError as e:
        logger.error(f"Failed to find class {class_name} in module {module_path}: {e}")
        raise
    except TypeError as e:
        logger.error(
            f"Failed to instantiate {class_name} with provided parameters: {e}"
        )
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error loading tool from {tool_config.class_path}: {e}"
        )
        raise


class vLLMRolloutWithTools(vLLMRollout):
    """
    An advanced vLLM rollout engine capable of handling multiple tools like
    code interpreters and search engines during generation.

    This class extends vLLMRollout by orchestrating a multi-step generation
    process where the language model can emit special tokens to trigger external
    tools. The tool outputs are then fed back into the model to continue
    generation.
    """

    def __init__(
        self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs
    ):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer

        # initial rollouts, used when enable_dynamic_rollouts is **False**
        self.initial_rollouts = self.config.initial_rollouts
        # Get beam search related parameters from config
        self.beam_size = self.config.beam_size
        self.branch_probability = self.config.branch_probability
        self.entropy_weight = self.config.entropy_weight

        self.enable_dynamic_rollouts = self.config.enable_dynamic_rollouts
        logger.info(f"enable_dynamic_rollouts: {self.enable_dynamic_rollouts}")

        # Get tool settings from config
        tools_config = self.config.get("tools", OmegaConf.create({}))

        # Get general tool configuration
        self.tool_call_limit = tools_config.get("call_limit", 6)
        self.max_tool_workers = tools_config.get("max_workers", 64)
        self.tool_timeout = tools_config.get("timeout", 120)

        # Other possible general tool configurations
        self.tool_retry_count = tools_config.get("retry_count", 4)
        self.tool_verbose_logging = tools_config.get("verbose_logging", False)

        self.tools: Dict[str, BaseTool] = {}
        if "tool_instances" in tools_config:
            for tool_name, tool_config in tools_config.tool_instances.items():
                logger.info(f"Loading tool '{tool_name}' from {tool_config.class_path}")
                try:
                    tool_instance = _load_tool_from_config(tool_config)
                    self.tools[tool_instance.trigger_tag] = tool_instance
                except Exception as e:
                    logger.error(
                        f"Could not initialize tool '{tool_name}'. Please check your configuration. Error: {e}"
                    )
                    if tools_config.get("fail_on_error", False):
                        raise

        self.stop_sequences = [f"</{tag}>" for tag in self.tools.keys()]
        self.logprobs = 10  # entropy
        self.initial_entropy_dict = {}  # Record initial entropy for each active index in first round
        self.consecutive_branches = {}  # Record consecutive branch count for each sample

        if not self.tools:
            logger.warning(
                "vLLMRolloutWithTools initialized, but no tools were configured."
            )

        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_tool_workers
        )

    def __del__(self):
        self.executor.shutdown(wait=False)

    def _extract_content(self, text: str, tag: str) -> str:
        """Extracts content from within the last <tag>...</tag> block."""
        try:
            start_tag = f"<{tag}>"
            end_tag = f"</{tag}>"
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag) : end_pos].strip()
        except ValueError:
            logger.warning(
                f"Could not extract content for tag '{tag}' from text: {text}"
            )
            return ""

    def _execute_tool_with_retry(self, tool, content):
        retry_count = 0
        start_time = time.time()

        while retry_count < self.tool_retry_count:
            try:
                result_text = tool.execute(content)
                if result_text:
                    execution_time = time.time() - start_time
                    return {
                        "success": True,
                        "retry_count": retry_count,
                        "execution_time": execution_time,
                        "result": result_text,
                    }
                else:
                    logger.warning(
                        f"Tool({tool.trigger_tag}) returned empty output. Retrying {retry_count + 1}/{self.tool_retry_count}"
                    )
                    retry_count += 1
            except Exception as e:
                logger.error(
                    f"Tool({tool.trigger_tag}) execution failed. Retrying {retry_count + 1}/{self.tool_retry_count}: {e}"
                )
                retry_count += 1

        execution_time = time.time() - start_time
        logger.warning(
            f"Tool({tool.trigger_tag}) execution failed after {self.tool_retry_count} retries. Appending EOS."
        )
        return {
            "success": False,
            "retry_count": retry_count,
            "execution_time": execution_time,
            "result": "",
        }

    def _calc_entropy(self, logprobs):
        if not logprobs:
            return 0.0
        p_list = [math.exp(l) for l in logprobs]
        entropy = -sum(p * l for p, l in zip(p_list, logprobs))
        return entropy

    def _calculate_initial_rollouts_dynamical(self, prompts: DataProto, **kwargs) -> List[int]:
        """
        Calculate initial_rollouts value for each sample in parallel
        Based on initial entropy and average entropy during tool invocation process
        """
        input_ids = prompts.batch["input_ids"]
        batch_size = input_ids.size(0)

        # Get sampling parameters
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        # Set inference parameters
        if not do_sample:
            kwargs.update(
                {
                    "best_of": 1,
                    "top_p": 1.0,
                    "top_k": -1,
                    "min_p": 0.0,
                    "temperature": 0,
                    "n": 1,
                }
            )
        elif is_validate:
            kwargs.update(
                {
                    "top_k": self.config.val_kwargs.top_k,
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "n": 1,
                }
            )

        kwargs["allowed_token_ids"] = list(self.tokenizer.get_vocab().values())

        with self.update_sampling_params(**kwargs):
            prompt_token_ids_list = [
                _pre_process_inputs(self.pad_token_id, prompt) for prompt in input_ids
            ]

            # Print all sample information
            logger.info("=" * 60)
            logger.info("PHASE 1: CALCULATING DYNAMIC INITIAL_ROLLOUTS")
            logger.info("=" * 60)
            for i, prompt_ids in enumerate(prompt_token_ids_list):
                prompt_text = self.tokenizer.decode(
                    prompt_ids, skip_special_tokens=True
                )
                logger.info(
                    f"[{i+1}/{batch_size}] Sample {i}: {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}"
                )

            # Step 1: Calculate initial entropy H_root in parallel
            logger.info(
                "Step 1: Calculating initial entropy (H_root) for all samples in parallel..."
            )
            initial_entropy_tokens = self.config.get("initial_entropy_tokens", 50)
            with self.update_sampling_params(
                n=1,
                stop=self.stop_sequences,
                max_tokens=initial_entropy_tokens,
                detokenize=True,
                logprobs=self.logprobs,
            ):
                initial_outputs = self.inference_engine.generate(
                    prompt_token_ids=prompt_token_ids_list,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )

            # Calculate initial entropy
            vocab_size = len(self.tokenizer.get_vocab())
            entropy_norm_factor = math.log(vocab_size)
            initial_entropies = []

            for i, initial_output in enumerate(initial_outputs):
                initial_entropy = 0.0
                if initial_output.outputs[0].logprobs:
                    logprobs = []
                    for j in range(min(20, len(initial_output.outputs[0].token_ids))):
                        try:
                            logprob_info = initial_output.outputs[0].logprobs[j]
                        except Exception:
                            logprob_info = initial_output.outputs[0].logprobs[-1]
                        token_list = list(logprob_info.values())
                        token_logprobs = [token.logprob for token in token_list]
                        logprobs.extend(token_logprobs)

                    if logprobs:
                        initial_entropy = (
                            self._calc_entropy(logprobs) / entropy_norm_factor
                        )
                initial_entropies.append(initial_entropy)

            # Step 2: Execute tool invocation inference flow in parallel
            logger.info(
                "Step 2: Calculating step entropy (H_high) for all samples in parallel..."
            )

            # Initialize state
            curr_inputs = [prompt_ids.copy() for prompt_ids in prompt_token_ids_list]
            step_entropies_list = [[] for _ in range(batch_size)]
            call_counters = [0] * batch_size
            active_indices = list(range(batch_size))

            iteration = 0
            while active_indices:
                iteration += 1
                logger.info(
                    f"Entropy calculation iteration {iteration}: {len(active_indices)} active samples"
                )

                # Generate next batch of tokens in parallel
                active_prompts = [curr_inputs[i] for i in active_indices]

                # Check sequence length, filter out sequences that are too long
                max_model_len = getattr(
                    self.inference_engine.llm_engine.model_config, "max_model_len", 4096
                )
                valid_indices = []
                valid_prompts = []
                for i, prompt in enumerate(active_prompts):
                    if len(prompt) < max_model_len:
                        valid_indices.append(active_indices[i])
                        valid_prompts.append(prompt)
                    else:
                        logger.warning(
                            f"Sample {active_indices[i]} too long ({len(prompt)} tokens), skipping"
                        )

                if not valid_prompts:
                    logger.warning("All samples too long, ending entropy calculation")
                    break

                active_indices = valid_indices
                active_prompts = valid_prompts

                # Calculate max_tokens, avoid empty list case
                if active_indices:
                    max_remaining_tokens = max(
                        1,
                        self.config.response_length
                        - max(
                            (len(curr_inputs[i]) - len(prompt_token_ids_list[i]))
                            for i in active_indices
                        ),
                    )
                    # Ensure not exceeding model maximum length
                    max_model_len = getattr(
                        self.inference_engine.llm_engine.model_config,
                        "max_model_len",
                        4096,
                    )
                    max_remaining_tokens = min(
                        max_remaining_tokens,
                        max_model_len
                        - max(len(curr_inputs[i]) for i in active_indices),
                    )
                    max_remaining_tokens = max(1, max_remaining_tokens)
                else:
                    max_remaining_tokens = 1

                with self.update_sampling_params(
                    n=1,
                    stop=self.stop_sequences,
                    max_tokens=max_remaining_tokens,
                    detokenize=True,
                    logprobs=self.logprobs,
                ):
                    outputs = self.inference_engine.generate(
                        prompt_token_ids=active_prompts,
                        sampling_params=self.sampling_params,
                        use_tqdm=False,
                    )

                # Calculate entropy for current step
                current_entropy_dict = {}
                for i, out_idx in enumerate(active_indices):
                    output = outputs[i]
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
                        entropy = self._calc_entropy(logprobs) / entropy_norm_factor
                    else:
                        entropy = 0.0
                    current_entropy_dict[out_idx] = entropy
                    step_entropies_list[out_idx].append(entropy)

                # Process outputs and tool calls
                tool_requests = {tag: [] for tag in self.tools}
                next_active_indices = []

                for i, out_idx in enumerate(active_indices):
                    output = outputs[i]
                    generated_tokens = output.outputs[0].token_ids
                    curr_inputs[out_idx].extend(generated_tokens)

                    finish_reason = output.outputs[0].finish_reason
                    stop_reason = output.outputs[0].stop_reason
                    is_tool_call = (
                        finish_reason == "stop" and stop_reason in self.stop_sequences
                    )

                    if is_tool_call and call_counters[out_idx] < self.tool_call_limit:
                        tag = stop_reason.strip("</>")
                        full_text = self.tokenizer.decode(curr_inputs[out_idx])
                        content = self._extract_content(full_text, tag)
                        if content:
                            tool_requests[tag].append(
                                {"index": out_idx, "content": content}
                            )
                            next_active_indices.append(out_idx)
                            call_counters[out_idx] += 1
                    elif finish_reason == "length":
                        if (
                            len(curr_inputs[out_idx])
                            - len(prompt_token_ids_list[out_idx])
                            < self.config.response_length
                        ):
                            next_active_indices.append(out_idx)
                    # End when finish_reason == 'stop', but entropy value has been recorded

                # Execute tool calls in parallel
                if any(tool_requests.values()):
                    total_requests = sum(len(reqs) for reqs in tool_requests.values())
                    logger.info(
                        f"Processing {total_requests} tool requests in parallel..."
                    )

                    futures = {}
                    for tag, requests in tool_requests.items():
                        if not requests:
                            continue
                        tool = self.tools[tag]
                        for req in requests:
                            future = self.executor.submit(
                                self._execute_tool_with_retry, tool, req["content"]
                            )
                            futures[future] = {"index": req["index"], "tag": tag}

                    # Wait for all tool calls to complete
                    for future in concurrent.futures.as_completed(futures):
                        fut_info = futures[future]
                        idx = fut_info["index"]
                        tag = fut_info["tag"]
                        try:
                            result = future.result(timeout=self.tool_timeout)
                            result_text = (
                                result["result"]
                                if result["success"]
                                else                                 f"Tool({tag}) returned empty output."
                            )

                            # Add tool result to input
                            formatted_result = f" <result>\n{result_text}\n</result>"
                            result_tokens = self.tokenizer.encode(formatted_result)

                            # Check if exceeding maximum length after adding tool result
                            max_model_len = getattr(
                                self.inference_engine.llm_engine.model_config,
                                "max_model_len",
                                4096,
                            )
                            if (
                                len(curr_inputs[idx]) + len(result_tokens)
                                > max_model_len
                            ):
                                logger.warning(
                                    f"Sample {idx} would exceed max length after tool result, truncating"
                                )
                                # Truncate to safe length
                                max_safe_length = (
                                    max_model_len - len(result_tokens) - 100
                                )  # Leave some margin
                                curr_inputs[idx] = curr_inputs[idx][:max_safe_length]

                            curr_inputs[idx].extend(result_tokens)

                        except Exception as e:
                            logger.error(f"Tool execution failed for sample {idx}: {e}")
                            result_text = (
                                f"Error: Tool({tag}) execution failed with message: {e}"
                            )
                            formatted_result = f" <result>\n{result_text}\n</result>"
                            result_tokens = self.tokenizer.encode(formatted_result)

                            # Check if exceeding maximum length after adding error result
                            max_model_len = getattr(
                                self.inference_engine.llm_engine.model_config,
                                "max_model_len",
                                4096,
                            )
                            if (
                                len(curr_inputs[idx]) + len(result_tokens)
                                > max_model_len
                            ):
                                logger.warning(
                                    f"Sample {idx} would exceed max length after error result, truncating"
                                )
                                # Truncate to safe length
                                max_safe_length = (
                                    max_model_len - len(result_tokens) - 100
                                )  # Leave some margin
                                curr_inputs[idx] = curr_inputs[idx][:max_safe_length]

                            curr_inputs[idx].extend(result_tokens)

                active_indices = next_active_indices

            # Step 3: Calculate average step entropy and initial_rollouts for each sample
            logger.info("Step 3: Calculating final initial_rollouts for all samples...")
            initial_rollouts_list = []
            for i in range(batch_size):
                # Calculate average step entropy
                if step_entropies_list[i]:
                    avg_step_entropy = sum(step_entropies_list[i]) / len(
                        step_entropies_list[i]
                    )
                else:
                    # If no step entropy, use initial entropy as fallback
                    avg_step_entropy = initial_entropies[i]
                    logger.warning(
                        f"Sample {i} has no step entropies, using initial entropy as fallback"
                    )

                # Calculate initial_rollouts
                entropy_diff = initial_entropies[i] - avg_step_entropy
                sigmoid_input = 0.5 * entropy_diff
                sigmoid_value = 1 / (1 + math.exp(-sigmoid_input))

                num_samples = self.sampling_params.n
                # Round up
                initial_rollouts = int(num_samples * sigmoid_value) + 1

                initial_rollouts = max(1, min(initial_rollouts, num_samples))

                initial_rollouts_list.append(initial_rollouts)

                # Print results
                logger.info(
                    f"Sample {i} -> H_root: {initial_entropies[i]:.3f}, H_high: {avg_step_entropy:.3f}, Initial_rollouts: {initial_rollouts}"
                )

            logger.info("=" * 60)
            logger.info(
                "PHASE 1 COMPLETED: Dynamic initial_rollouts calculation finished"
            )
            logger.info("=" * 60)

        return initial_rollouts_list

    @GPUMemoryLogger(role="vllm rollout spmd with tools", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if vllm_version in ("0.5.4", "0.6.3") and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = self.tokenizer.eos_token_id
        batch_size = input_ids.size(0)

        # Initialize tool call statistics
        tool_metrics = {
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

        # Statistics for each tool
        calls_per_tool = Counter()
        success_per_tool = Counter()
        total_time_per_tool = Counter()

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        # Update sampling parameter settings
        beam_size = self.beam_size
        if not do_sample:
            kwargs.update(
                {
                    "best_of": 1,
                    "top_p": 1.0,
                    "top_k": -1,
                    "min_p": 0.0,
                    "temperature": 0,
                    "n": 1,
                }
            )
            beam_size = 1
        elif is_validate:
            kwargs.update(
                {
                    "top_k": self.config.val_kwargs.top_k,
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "n": 1,  # Use single sample in validation mode
                }
            )
            beam_size = 1

        # fix oov error
        kwargs["allowed_token_ids"] = list(self.tokenizer.get_vocab().values())

        with self.update_sampling_params(**kwargs):
            num_samples = self.sampling_params.n

            prompt_token_ids_list = [
                _pre_process_inputs(self.pad_token_id, prompt) for prompt in input_ids
            ]

            if self.enable_dynamic_rollouts:
                # Dynamically calculate initial_rollouts for each sample
                initial_rollouts_list = self._calculate_initial_rollouts_dynamical(prompts, **kwargs)
            else:
                initial_rollouts_list = [max(1, min(self.initial_rollouts, num_samples))] * len(prompt_token_ids_list)
            logger.info(f"Final calculated initial_rollouts: {initial_rollouts_list}")

            # State for each sample in the batch
            # Create initial rollout for each sample, count controlled by dynamically calculated initial_rollouts

            curr_inputs = []
            init_inputs = []
            result_masks = []
            call_counters = []
            active_indices = []

            # Create initial samples, using dynamically calculated initial_rollouts
            for i, ids in enumerate(prompt_token_ids_list):
                initial_rollouts = initial_rollouts_list[i]
                initial_rollouts = min(
                    initial_rollouts, num_samples
                )  # But not exceeding num_samples

                for _ in range(initial_rollouts):
                    curr_inputs.append(ids.copy())
                    init_inputs.append(ids.copy())
                    result_masks.append([])
                    call_counters.append(0)
                    active_indices.append(len(curr_inputs) - 1)

            # Track rollouts per original sample
            rollouts_per_sample = (
                initial_rollouts_list.copy()
            )  # Use dynamically calculated rollouts count
            # Initially each sample has multiple indices
            sample_to_indices = {}
            current_idx = 0
            for i, initial_rollouts in enumerate(initial_rollouts_list):
                sample_to_indices[i] = list(
                    range(current_idx, current_idx + initial_rollouts)
                )
                current_idx += initial_rollouts

            # Initialize consecutive branch counter
            for i in range(batch_size):
                self.consecutive_branches[i] = 0

            max_len = self.config.response_length

            logger.info("=" * 60)
            logger.info("PHASE 2: MAIN INFERENCE WITH TOOLS")
            logger.info("=" * 60)

            iteration = 0
            while active_indices:
                iteration += 1
                active_prompts = [curr_inputs[i] for i in active_indices]
                logger.info(
                    f"--- Iteration {iteration}: {len(active_indices)} active samples ---"
                )
                logger.debug(f"rollouts_per_sample: {rollouts_per_sample}")
                logger.debug(f"active_indices: {active_indices}")
                logger.debug(f"active_prompts: {active_prompts}")

                # Update max_tokens for each active sample
                with self.update_sampling_params(
                    n=1,
                    stop=self.stop_sequences,
                    max_tokens=max(
                        1,
                        max(
                            (
                                max_len - (len(curr_inputs[i]) - len(init_inputs[i]))
                                for i in active_indices
                            )
                        ),
                    ),
                    detokenize=True,
                    logprobs=self.logprobs,
                ):
                    outputs = self.inference_engine.generate(
                        prompt_token_ids=active_prompts,
                        sampling_params=self.sampling_params,
                        use_tqdm=False,
                    )
                # ========== Entropy recording and normalization ==========
                vocab_size = len(self.tokenizer.get_vocab())
                entropy_norm_factor = math.log(vocab_size)
                current_entropy_dict = {}
                for i, out_idx in enumerate(active_indices):
                    output = outputs[i]
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
                        entropy = self._calc_entropy(logprobs) / entropy_norm_factor
                    else:
                        entropy = 0.0
                    current_entropy_dict[out_idx] = entropy
                    if out_idx not in self.initial_entropy_dict:
                        self.initial_entropy_dict[out_idx] = entropy
                # ========== Entropy recording and normalization END ==========

                tool_requests: Dict[str, List[Dict]] = {tag: [] for tag in self.tools}
                next_active_indices = []

                for i, out_idx in enumerate(active_indices):
                    output = outputs[i]
                    generated_tokens = output.outputs[0].token_ids

                    curr_inputs[out_idx].extend(generated_tokens)
                    result_masks[out_idx].extend([1] * len(generated_tokens))

                    finish_reason = output.outputs[0].finish_reason
                    stop_reason = output.outputs[0].stop_reason

                    is_tool_call = (
                        finish_reason == "stop" and stop_reason in self.stop_sequences
                    )

                    # Debug information
                    logger.debug(f"  Sample {out_idx} output:")
                    logger.debug(f"  Finish reason: {finish_reason}")
                    logger.debug(f"  Stop reason: {stop_reason}")
                    logger.debug(f"  Is tool call: {is_tool_call}")
                    logger.debug(
                        f"  Tool: {stop_reason.strip('</>') if is_tool_call else 'No tool call'}"
                    )

                    if is_tool_call:
                        tag = stop_reason.strip("</>")
                        if call_counters[out_idx] < self.tool_call_limit:
                            call_counters[out_idx] += 1
                            full_text = self.tokenizer.decode(curr_inputs[out_idx])
                            content = self._extract_content(full_text, tag)
                            if content:
                                tool_requests[tag].append(
                                    {"index": out_idx, "content": content}
                                )
                                next_active_indices.append(out_idx)
                                # Update tool call count statistics
                                tool_metrics["tools/total_calls"] += 1
                                calls_per_tool[tag] += 1
                        else:
                            logger.warning(
                                f"Tool call limit reached for sample {out_idx}. Appending EOS."
                            )
                            curr_inputs[out_idx].append(eos_token_id)
                            result_masks[out_idx].append(1)
                            tool_metrics["tools/call_limit_reached_count"] += 1

                    elif finish_reason == "length":
                        if (
                            len(curr_inputs[out_idx]) - len(init_inputs[out_idx])
                            < max_len
                        ):
                            next_active_indices.append(out_idx)

                    elif finish_reason == "stop":  # EOS
                        pass

                if any(tool_requests.values()):
                    total_requests = sum(len(reqs) for reqs in tool_requests.values())
                    logger.info(f"Processing {total_requests} tool requests...")
                    futures = {}
                    for tag, requests in tool_requests.items():
                        if not requests:
                            continue
                        logger.debug(
                            f"Processing {len(requests)} requests for tool '{tag}'"
                        )
                        tool = self.tools[tag]
                        for req in requests:
                            logger.debug(
                                f"Submitting tool request: tool={tag}, idx={req['index']}, content={req['content']}"
                            )
                            future = self.executor.submit(
                                self._execute_tool_with_retry, tool, req["content"]
                            )
                            futures[future] = {"index": req["index"], "tag": tag}

                    total_futures = len(futures)
                    completed_futures = 0
                    logger.debug(
                        f"Submitted {total_futures} tool requests for execution"
                    )
                    for future in concurrent.futures.as_completed(futures):
                        completed_futures += 1
                        fut_info = futures[future]
                        idx = fut_info["index"]
                        tag = fut_info["tag"]
                        try:
                            result = future.result(timeout=self.tool_timeout)
                            # Parse tool execution result
                            success = result["success"]
                            retry_count = result["retry_count"]
                            execution_time = result["execution_time"]
                            result_text = result["result"]

                            # Update statistics
                            if success:
                                tool_metrics["tools/successful_calls"] += 1
                                success_per_tool[tag] += 1
                                logger.info(
                                    f"Tool({tag}) for sample {idx} completed successfully in {execution_time:.2f}s, result length: {len(result_text)}"
                                )
                            else:
                                tool_metrics["tools/failed_calls"] += 1
                                result_text = f"Tool({tag}) returned empty output."
                                logger.warning(
                                    f"Tool({tag}) for sample {idx} failed after {retry_count} retries, execution time: {execution_time:.2f}s"
                                )

                            tool_metrics["tools/total_execution_time"] += execution_time
                            tool_metrics["tools/max_execution_time"] = max(
                                tool_metrics["tools/max_execution_time"], execution_time
                            )
                            tool_metrics["tools/total_retries"] += retry_count
                            tool_metrics["tools/max_retries"] = max(
                                tool_metrics["tools/max_retries"], retry_count
                            )

                            # Update time statistics for each tool
                            total_time_per_tool[tag] += execution_time

                            if not result_text:
                                result_text = f"Tool({tag}) returned empty output."
                                logger.warning(
                                    f"Tool({tag}) for sample {idx} returned empty output, execution time: {execution_time:.2f}s"
                                )
                            else:
                                logger.debug(f"Tool({tag}) result: {result_text}")

                        except Exception as e:
                            logger.error(
                                f"Tool({tag}) execution failed for sample {idx}: {e}"
                            )
                            result_text = (
                                f"Error: Tool({tag}) execution failed with message: {e}"
                            )
                            tool_metrics["tools/failed_calls"] += 1

                        if (
                            completed_futures % max(1, total_futures // 4) == 0
                            or completed_futures == total_futures
                        ):
                            logger.info(
                                f"Tool execution progress: {completed_futures}/{total_futures} ({completed_futures/total_futures*100:.1f}%)"
                            )
                        formatted_result = f" <result>\n{result_text}\n</result>"
                        result_tokens = self.tokenizer.encode(formatted_result)
                        logger.debug(
                            f"Result for tool({tag}), sample {idx} tokenized to {len(result_tokens)} tokens"
                        )
                        curr_inputs[idx].extend(result_tokens)
                        result_masks[idx].extend([0] * len(result_tokens))

                final_active_indices = []
                for idx in next_active_indices:
                    response_len = len(curr_inputs[idx]) - len(init_inputs[idx])
                    if response_len < max_len:
                        final_active_indices.append(idx)

                # Apply beam search: split active samples into multiple branches
                new_inputs = []
                new_init_inputs = []
                new_result_masks = []
                new_call_counters = []
                new_sample_origins = []  # Record original sample for each new branch

                # Map from original sample index to its active rollouts
                active_by_sample = {}
                for idx in final_active_indices:
                    # Find which original sample this index belongs to
                    orig_sample = None
                    for sample_idx, indices in sample_to_indices.items():
                        if idx in indices:
                            orig_sample = sample_idx
                            break

                    if orig_sample is not None:
                        if orig_sample not in active_by_sample:
                            active_by_sample[orig_sample] = []
                        active_by_sample[orig_sample].append(idx)

                # Consider entropy and random prob
                for orig_sample, active_idxs in active_by_sample.items():
                    remaining_slots = num_samples - rollouts_per_sample[orig_sample]
                    if remaining_slots <= 0:
                        continue
                    branches_created = 0
                    for source_idx in active_idxs:
                        branches_per_idx = min(
                            beam_size - 1, remaining_slots - branches_created
                        )
                        if branches_per_idx <= 0:
                            break
                        for _ in range(branches_per_idx):
                            # ==== Branch probability and entropy difference ====

                            entropy_now = current_entropy_dict.get(source_idx, 0.0)
                            entropy_init = self.initial_entropy_dict.get(
                                source_idx, 0.0
                            )
                            entropy_delta = entropy_now - entropy_init
                            prob = random.random() + self.entropy_weight * entropy_delta

                            # Apply consecutive branch penalty factor
                            consecutive_branches = self.consecutive_branches.get(
                                orig_sample, 0
                            )
                            penalty_factor = 1.0 - 0.05 * consecutive_branches
                            penalty_factor = max(0.0, penalty_factor)  # Ensure not negative
                            original_prob = prob
                            prob = prob * penalty_factor

                            # Debug log
                            if consecutive_branches > 0:
                                logger.debug(
                                    f"Sample {orig_sample}: consecutive_branches={consecutive_branches}, penalty_factor={penalty_factor:.3f}, prob: {original_prob:.3f} -> {prob:.3f}"
                                )

                            prob = max(0.0, min(1.0, prob))
                            if (
                                prob < self.branch_probability
                            ):  # If prob > 0.5, copy a tree
                                continue
                            # ==== END ====
                            new_inputs.append(curr_inputs[source_idx].copy())
                            new_init_inputs.append(init_inputs[source_idx].copy())
                            new_result_masks.append(result_masks[source_idx].copy())
                            new_call_counters.append(call_counters[source_idx])
                            new_sample_origins.append(orig_sample)
                            rollouts_per_sample[orig_sample] += 1
                            branches_created += 1
                        if branches_created >= remaining_slots:
                            break

                # Add non-active samples that still need more rollouts
                for orig_sample in range(batch_size):
                    if (
                        orig_sample not in active_by_sample
                        and rollouts_per_sample[orig_sample] < num_samples
                    ):
                        # For inactive samples, only add one branch at a time
                        branches_to_add = min(
                            1, num_samples - rollouts_per_sample[orig_sample]
                        )
                        if branches_to_add <= 0:
                            continue

                        # Use first index of this sample as template
                        source_idx = sample_to_indices[orig_sample][0]

                        # Create new branches
                        for _ in range(branches_to_add):
                            new_inputs.append(init_inputs[source_idx].copy())
                            new_init_inputs.append(init_inputs[source_idx].copy())
                            new_result_masks.append([])
                            new_call_counters.append(0)
                            new_sample_origins.append(orig_sample)  # Record original sample
                            rollouts_per_sample[orig_sample] += 1

                # Add new branches to existing lists
                if new_inputs:
                    start_idx = len(curr_inputs)
                    curr_inputs.extend(new_inputs)
                    init_inputs.extend(new_init_inputs)
                    result_masks.extend(new_result_masks)
                    call_counters.extend(new_call_counters)

                    # Add new indices to active list
                    final_active_indices.extend(
                        range(start_idx, start_idx + len(new_inputs))
                    )

                    # Update mapping using correct original sample information
                    for i, new_idx in enumerate(
                        range(start_idx, start_idx + len(new_inputs))
                    ):
                        orig_sample = new_sample_origins[i]
                        sample_to_indices.setdefault(orig_sample, []).append(new_idx)

                    # Update consecutive branch count
                    # Count which samples created new branches
                    samples_with_new_branches = set(new_sample_origins)

                    # Update consecutive branch count for all samples
                    for orig_sample in range(batch_size):
                        if orig_sample in samples_with_new_branches:
                            # If this sample created new branches, increment consecutive branch count
                            self.consecutive_branches[orig_sample] += 1
                        else:
                            # If this sample didn't create new branches, reset consecutive branch count
                            self.consecutive_branches[orig_sample] = 0

                active_indices = final_active_indices
                logger.info(
                    f"--- End iteration {iteration}: {len(active_indices)} samples remain active ---"
                )

            logger.info("=" * 60)
            logger.info("PHASE 2 COMPLETED: Main inference with tools finished")
            logger.info("=" * 60)

            # Ensure all sequences don't exceed max_len
            for idx in range(len(curr_inputs)):
                response_len = len(curr_inputs[idx]) - len(init_inputs[idx])
                if response_len > max_len:
                    offset = len(init_inputs[idx])
                    curr_inputs[idx] = curr_inputs[idx][: offset + max_len]
                    result_masks[idx] = result_masks[idx][:max_len]

            # Reorganize outputs to match original batch structure and select up to num_samples per sample
            output_sequences = []
            output_result_masks = []
            for i in range(batch_size):
                # Get all indices for this sample
                sample_indices = sample_to_indices.get(i, [])
                # Ensure we have exactly num_samples outputs per sample
                selected_indices = sample_indices[:num_samples]

                # If we have fewer rollouts than requested, duplicate the last one
                while len(selected_indices) < num_samples:
                    if selected_indices:
                        selected_indices.append(selected_indices[-1])
                    else:
                        break  # Should not happen but just in case

                # Extract outputs for selected indices
                for idx in selected_indices:
                    output_sequences.append(
                        curr_inputs[idx][len(prompt_token_ids_list[i]) :]
                    )
                    output_result_masks.append(result_masks[idx])

            padded_response_list = []
            padded_result_mask_list = []
            for output_ids, result_mask in zip(output_sequences, output_result_masks):
                # logger.debug(f"len(output_ids): {len(output_ids)}, len(result_mask): {len(result_mask)}, output_ids: {output_ids}, result_mask: {result_mask}")

                assert len(output_ids) == len(
                    result_mask
                ), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"

                response = torch.tensor(output_ids)
                response = pad_sequence_to_length(
                    response, self.config.response_length, self.pad_token_id
                )

                result_mask_tensor = torch.tensor(result_mask)
                result_mask_tensor = pad_sequence_to_length(
                    result_mask_tensor, self.config.response_length, 0
                )

                padded_response_list.append(response)
                padded_result_mask_list.append(result_mask_tensor)

            response = torch.stack(padded_response_list, dim=0).to(input_ids.device)
            loss_mask = torch.stack(padded_result_mask_list, dim=0).to(input_ids.device)

            non_tensor_batch = deepcopy(prompts.non_tensor_batch)
            if num_samples > 1 and do_sample:
                input_ids = _repeat_interleave(input_ids, num_samples)
                attention_mask = _repeat_interleave(attention_mask, num_samples)
                position_ids = _repeat_interleave(position_ids, num_samples)
                if non_tensor_batch:
                    for key, value in non_tensor_batch.items():
                        if isinstance(value, np.ndarray):
                            non_tensor_batch[key] = np.repeat(
                                value, num_samples, axis=0
                            )
                        elif isinstance(value, list):
                            non_tensor_batch[key] = [
                                item for item in value for _ in range(num_samples)
                            ]

            final_batch_size = input_ids.size(0)
            seq = torch.cat([input_ids, response], dim=-1)

            response_length = response.size(1)
            delta_position_id = (
                torch.arange(1, response_length + 1, device=position_ids.device)
                .unsqueeze(0)
                .expand(final_batch_size, -1)
            )

            if position_ids.dim() == 3:  # for RoPE scaling like qwen2vl mrope
                delta_position_id = delta_position_id.view(
                    final_batch_size, 1, -1
                ).expand(final_batch_size, position_ids.size(1), -1)
                response_position_ids = (
                    position_ids[..., -1:].expand(-1, position_ids.size(1), -1)
                    + delta_position_id
                )
            else:
                response_position_ids = position_ids[..., -1:] + delta_position_id

            final_position_ids = torch.cat(
                [position_ids, response_position_ids], dim=-1
            )

            response_attention_mask = get_response_mask(
                response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
            )
            final_attention_mask = torch.cat(
                (attention_mask, response_attention_mask), dim=-1
            )

            loss_mask = loss_mask * response_attention_mask

            # Calculate average execution time
            if tool_metrics["tools/total_calls"] > 0:
                tool_metrics["tools/avg_execution_time"] = (
                    tool_metrics["tools/total_execution_time"]
                    / tool_metrics["tools/total_calls"]
                )

            # Calculate average execution time and success rate for each tool
            tool_specific_metrics = {}
            for tag in self.tools.keys():
                calls = calls_per_tool[tag]
                if calls > 0:
                    tool_specific_metrics[f"tools/{tag}/calls"] = calls
                    tool_specific_metrics[f"tools/{tag}/avg_time"] = (
                        total_time_per_tool[tag] / calls
                    )
                    tool_specific_metrics[f"tools/{tag}/success_rate"] = (
                        success_per_tool[tag] / calls
                    )
                else:
                    tool_specific_metrics[f"tools/{tag}/calls"] = 0
                    tool_specific_metrics[f"tools/{tag}/avg_time"] = 0
                    tool_specific_metrics[f"tools/{tag}/success_rate"] = 0

            batch = TensorDict(
                {
                    "prompts": input_ids,
                    "responses": response,
                    "input_ids": seq,
                    "attention_mask": final_attention_mask,
                    "loss_mask": loss_mask,
                    "position_ids": final_position_ids,
                },
                batch_size=final_batch_size,
            )

        if vllm_version in ("0.5.4", "0.6.3") and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        # Merge all metrics
        all_metrics = {**tool_metrics, **tool_specific_metrics}

        # Add metrics to meta_info
        meta_info = deepcopy(prompts.meta_info) if prompts.meta_info else {}
        meta_info["metrics"] = all_metrics

        data_proto = DataProto(
            batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info
        )

        return data_proto
