"""
模块名称：prompts
功能描述：Prompt 管理子包的初始化模块，对外暴露 prompt_manager 单例实例，
         用于统一管理和渲染 YAML 格式的 Prompt 模板。
"""

from .prompt_manager import prompt_manager

__all__ = ["prompt_manager"]
