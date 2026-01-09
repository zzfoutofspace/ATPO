from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from vllm import LLM
    from .base import BaseAgent

logger = logging.getLogger(__name__)

# A registry of available agent classes.
# The key is the agent's name as it would appear in the config,
# and the value is the path to the agent class.
AGENT_REGISTRY = {
    "tool_agent": "verl.workers.agent.tool_agent.ToolAgent",
    # Future agents can be registered here, for example:
    # "react_agent": "verl.workers.agent.react_agent.ReactAgent",
}


def create_agent(config: "DictConfig", vllm_engine: "LLM", tokenizer: "Any", **kwargs) -> "BaseAgent":
    """
    Factory function to create an agent instance based on the configuration.

    This function dynamically imports and instantiates the agent class
    specified in the `config.agent.name`.

    Args:
        config: The global configuration object.
        vllm_engine: The vLLM engine instance.
        tokenizer: The tokenizer instance.
        **kwargs: Additional keyword arguments to pass to the agent's constructor.

    Returns:
        An instance of a class that inherits from BaseAgent.

    Raises:
        ValueError: If the specified agent name is not found in the registry
                    or if the agent configuration is missing.
        ImportError: If the agent class cannot be imported.
    """
    agent_config = config.get("agent")
    if not agent_config:
        raise ValueError("Agent configuration (`agent`) not found in the main config.")

    agent_name = agent_config.get("name")
    if not agent_name:
        raise ValueError("Agent name (`agent.name`) must be specified in the config.")

    if agent_name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent name: '{agent_name}'. "
            f"Available agents are: {list(AGENT_REGISTRY.keys())}"
        )

    class_path = AGENT_REGISTRY[agent_name]
    module_path, class_name = class_path.rsplit('.', 1)

    try:
        # Dynamically import the module
        agent_module = __import__(module_path, fromlist=[class_name])
        # Get the class from the module
        agent_class = getattr(agent_module, class_name)
    except ImportError as e:
        logger.error(f"Failed to import agent module {module_path}: {e}")
        raise
    except AttributeError:
        logger.error(f"Could not find class {class_name} in module {module_path}.")
        raise

    return agent_class(config=config, vllm_engine=vllm_engine, tokenizer=tokenizer, **kwargs)


__all__ = ["create_agent", "BaseAgent"] 