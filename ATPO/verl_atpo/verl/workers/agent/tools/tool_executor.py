import importlib
import logging
import os
import time
from typing import Dict, Any

from omegaconf import DictConfig, OmegaConf

# It's better to have a clear contract for what a "tool" is.
# Assuming BaseTool exists and has `trigger_tag` and `execute` methods.
# If it doesn't exist, we might need to define a Protocol here.
# from verl.workers.rollout.tools.base_tool import BaseTool # This path is assumed
from verl.workers.agent.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _load_tool_from_config(tool_config: DictConfig) -> BaseTool:
    """Dynamically loads a tool from its configuration."""
    module_path, class_name = tool_config.class_path.rsplit('.', 1)
    try:
        module = importlib.import_module(module_path)
        tool_class = getattr(module, class_name)
        tool_params = OmegaConf.to_container(tool_config.get('params', {}), resolve=True)
        tool_instance = tool_class(**tool_params)
        return tool_instance
    except ImportError as e:
        logger.error(f"Failed to import module {module_path}: {e}")
        raise
    except AttributeError as e:
        logger.error(f"Failed to find class {class_name} in module {module_path}: {e}")
        raise
    except TypeError as e:
        logger.error(f"Failed to instantiate {class_name} with provided parameters: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading tool from {tool_config.class_path}: {e}")
        raise


class ToolExecutor:
    """
    Manages the lifecycle of tools, including loading from configuration,
    executing them with retry logic, and extracting content from model outputs.
    """

    def __init__(self, tools_config: DictConfig):
        """
        Initializes the ToolExecutor.

        Args:
            tools_config: The configuration section for tools.
        """
        self.tool_retry_count = tools_config.get("retry_count", 3)
        self.fail_on_error = tools_config.get("fail_on_error", False)
        self.tools: Dict[str, BaseTool] = self._load_tools(tools_config)
        self.stop_sequences = [f"</{tag}>" for tag in self.tools.keys()]

    def _load_tools(self, tools_config: DictConfig) -> Dict[str, BaseTool]:
        """Loads all tool instances from the configuration."""
        loaded_tools: Dict[str, BaseTool] = {}
        if "tool_instances" not in tools_config:
            logger.warning("No 'tool_instances' found in tools_config. No tools will be loaded.")
            return loaded_tools

        for tool_name, tool_config in tools_config.tool_instances.items():
            logger.info(f"Loading tool '{tool_name}' from {tool_config.class_path}")
            try:
                tool_instance = _load_tool_from_config(tool_config)
                if tool_instance.trigger_tag in loaded_tools:
                    logger.warning(f"Duplicate trigger tag '{tool_instance.trigger_tag}' for tool '{tool_name}'. "
                                   f"The previous tool will be overwritten.")
                loaded_tools[tool_instance.trigger_tag] = tool_instance
            except Exception as e:
                logger.error(f"Could not initialize tool '{tool_name}'. Error: {e}")
                if self.fail_on_error:
                    raise
        return loaded_tools

    def execute_with_retry(self, tool: BaseTool, content: str) -> Dict[str, Any]:
        """
        Executes a tool with a configured number of retries upon failure.

        Args:
            tool: The tool instance to execute.
            content: The content to pass to the tool's execute method.

        Returns:
            A dictionary containing the execution result.
        """
        retry_count = 0
        start_time = time.time()
        
        while retry_count < self.tool_retry_count:
            try:
                result_text = tool.execute(content)
                if result_text is not None:  # Allow empty strings as valid results
                    execution_time = time.time() - start_time
                    logger.debug(f"Tool {tool.trigger_tag} executed successfully.")
                    return {
                        "success": True,
                        "retry_count": retry_count,
                        "execution_time": execution_time,
                        "result": result_text
                    }
                else:
                    logger.warning(
                        f"Tool({tool.trigger_tag}) returned None. Retrying {retry_count + 1}/{self.tool_retry_count}")
                    retry_count += 1
            except Exception as e:
                logger.error(
                    f"Tool({tool.trigger_tag}) execution failed. Retrying {retry_count + 1}/{self.tool_retry_count}: {e}")
                retry_count += 1
        
        execution_time = time.time() - start_time
        logger.error(f"Tool({tool.trigger_tag}) execution failed after {self.tool_retry_count} retries.")
        return {
            "success": False,
            "retry_count": retry_count,
            "execution_time": execution_time,
            "result": f"Tool({tool.trigger_tag}) failed after multiple retries."
        }

    @staticmethod
    def extract_content(text: str, tag: str) -> str:
        """
        Extracts content from within the last <tag>...</tag> block in a string.

        Args:
            text: The text containing the tagged content.
            tag: The tag to look for (e.g., 'tool_code').

        Returns:
            The extracted content, or an empty string if not found.
        """
        try:
            start_tag = f"<{tag}>"
            end_tag = f"</{tag}>"
            # Find the last occurrence of the end tag first
            end_pos = text.rindex(end_tag)
            # Then find the last start tag before that end tag
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            logger.warning(f"Could not extract content for tag '{tag}' from text.")
            return ""

    def has_tools(self) -> bool:
        """Checks if any tools are loaded."""
        return bool(self.tools)

    def get_tool(self, tag: str) -> BaseTool:
        """Retrieves a tool instance by its trigger tag."""
        return self.tools[tag] 