import logging
import os
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams

from verl import DataProto

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class BaseAgent(ABC):
    """
    Abstract base class for agent implementations, following the OpenAI Gym paradigm.

    This class defines a stateful, step-by-step interaction model required for
    Reinforcement Learning. Instead of a single `run_loop`, it uses `reset` and `step`
    methods, allowing an external training loop to control the rollout process.
    """

    def __init__(self, config: DictConfig, vllm_engine: LLM, tokenizer: Any, **kwargs):
        """
        Initializes the agent's static components.

        Args:
            config: The main configuration object.
            vllm_engine: The initialized vLLM engine instance for generation.
            tokenizer: The tokenizer instance.
        """
        self.config = config
        self.agent_config = config.agent
        self.vllm_engine = vllm_engine
        self.tokenizer = tokenizer
        logger.info(f"Initialized agent of type: {self.__class__.__name__}")

    @abstractmethod
    def reset(self, prompts: DataProto, sampling_params: SamplingParams) -> DataProto:
        """
        Resets the agent's state for a new batch of episodes.

        Args:
            prompts: A DataProto containing the initial prompts for each episode
                     in the batch.
            sampling_params: The vLLM sampling parameters for generation.

        Returns:
            The initial observation for the batch of episodes.
        """
        raise NotImplementedError("Each agent must implement its own `reset` method.")

    @abstractmethod
    def step(self) -> Tuple[DataProto, torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Executes one step of interaction for the entire batch of episodes.
        The "action" is implicitly the generation from the LLM.

        Returns:
            A tuple containing:
            - next_observation (DataProto): The observation after the step.
            - rewards (torch.Tensor): A tensor of rewards for each episode.
            - dones (torch.Tensor): A boolean tensor indicating if each episode is finished.
            - infos (List[Dict]): A list of dictionaries with diagnostic information.
        """
        raise NotImplementedError("Each agent must implement its own `step` method.")

    @abstractmethod
    def get_final_responses(self) -> TensorDict:
        """
        Retrieves the final results after the episodes are done.
        This is used by the rollout worker to construct the final DataProto.

        Returns:
            A TensorDict containing at least 'responses' and 'loss_mask'.
        """
        raise NotImplementedError("Each agent must implement `get_final_responses`.")