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

import logging
import os
from copy import deepcopy
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask
from verl.workers.agent import create_agent
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout, _pre_process_inputs, _repeat_interleave

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class vLLMAgentRollout(vLLMRollout):
    """
    A generic vLLM rollout worker that drives a stateful, Gym-like agent
    through a `reset` and `step` loop to generate sequences.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)

        if not self.config.agent.activate_agent:
            self.agent = None
        else:
            self.agent = create_agent(
                config=config,
                vllm_engine=self.inference_engine,
                tokenizer=tokenizer
            )
        
        self.eos_token_id = tokenizer.eos_token_id

    @GPUMemoryLogger(role="vllm_agent_rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generates sequences by driving the agent through a full rollout.
        """
        if not self.agent:
            return super().generate_sequences(prompts, **kwargs)
        
        if vllm_version in ('0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        non_tensor_batch = prompts.non_tensor_batch
        input_ids = prompts.batch["input_ids"]
        batch_size = input_ids.size(0)

        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, input_ids[i]) for i in range(batch_size)],
                dtype=object
            )

        # --- 1. Prepare Sampling Parameters ---
        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        
        sampling_kwargs = {}
        if not do_sample:
            sampling_kwargs.update({'best_of': 1, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0, 'temperature': 0, 'n': 1})
        elif is_validate:
            sampling_kwargs.update({
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1
            })
        sampling_kwargs.update(kwargs)

        with self.update_sampling_params(**sampling_kwargs):
            # --- 2. Drive the agent's reset/step loop ---
            self.agent.reset(prompts, self.sampling_params)

            while True:
                dones = self.agent.step()
                if torch.all(dones):
                    break
            
            # --- 3. Get final results from the agent ---
            agent_output_batch = self.agent.get_final_responses()

        # --- 4. Finalize the DataProto ---
        responses = agent_output_batch.pop('responses')
        loss_mask = agent_output_batch.pop('loss_mask')

        input_ids = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = self.eos_token_id
        

        if self.sampling_params.n > 1 and do_sample:
            input_ids = _repeat_interleave(input_ids, self.sampling_params.n)  
            attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
            position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
            if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)
            if "interaction_kwargs" in non_tensor_batch.keys():
                non_tensor_batch["interaction_kwargs"] = _repeat_interleave(non_tensor_batch["interaction_kwargs"], self.sampling_params.n)
            if "raw_prompt" in non_tensor_batch.keys():
                non_tensor_batch["raw_prompt"] = _repeat_interleave(non_tensor_batch["raw_prompt"], self.sampling_params.n)


        final_batch_size = input_ids.size(0)
        seq = torch.cat([input_ids, responses], dim=-1)

        response_length = responses.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device).unsqueeze(0).expand(final_batch_size, -1)

        if position_ids.dim() == 3:
            response_position_ids = position_ids[..., -1:].expand(-1, -1, response_length) + delta_position_id.unsqueeze(1)
        else:
            response_position_ids = position_ids[..., -1:] + delta_position_id

        final_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=responses, eos_token=eos_token_id, dtype=attention_mask.dtype)
        final_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        final_loss_mask = loss_mask * response_attention_mask

        batch = TensorDict({
            "prompts": input_ids,
            "responses": responses,
            "input_ids": seq,
            "attention_mask": final_attention_mask,
            "loss_mask": final_loss_mask,
            "position_ids": final_position_ids,
        }, batch_size=final_batch_size)

        meta_info = deepcopy(prompts.meta_info)
        if "meta_info" in agent_output_batch:
            meta_info.update(agent_output_batch.get("meta_info"))

        if vllm_version in ('0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)