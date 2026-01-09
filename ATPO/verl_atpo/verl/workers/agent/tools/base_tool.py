from abc import ABC, abstractmethod


class BaseTool(ABC):
    """抽象基类，为所有工具定义通用接口"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass
    
    @property
    @abstractmethod
    def trigger_tag(self) -> str:
        """触发该工具的标签"""
        pass
    
    @abstractmethod
    def execute(self, content: str, **kwargs) -> str:
        """执行工具操作"""
        pass