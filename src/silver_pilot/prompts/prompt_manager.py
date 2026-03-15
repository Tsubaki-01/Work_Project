"""
模块名称：prompt_manager
功能描述：Prompt 模板管理器，基于 Jinja2 和 YAML 实现 Prompt 的加载、缓存与渲染，
         支持路径自动解析、模型配置提取以及模板变量动态填充。

用法示例：
    假设有一个 YAML 模板文件 `templates/hello.yaml`:
    ```yaml
    model_config:
      temperature: 0.7
      max_tokens: 1024
    messages:
      - role: system
        content: "你是一个叫做 {{ name }} 的人工智能助手。"
      - role: user
        content: "你好，{{ name }}！"
    ```

    1. 获取单例并加载 Prompt:
    ```python
    from silver_pilot.prompts.prompt_manager import prompt_manager

    # 构建 Prompt 消息
    messages = prompt_manager.build_prompt("hello", name="Silver")
    # 输出:
    # [
    #     {"role": "system", "content": "你是一个叫做 Silver 的人工智能助手。"},
    #     {"role": "user", "content": "你好，Silver！"}
    # ]

    # 获取模型配置
    config = prompt_manager.get_model_config("hello")
    # 输出: {"temperature": 0.7, "max_tokens": 1024}
    ```
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape


class PromptManager:
    """Prompt 模板管理器

    用于管理基于 YAML 的 Prompt 模板，支持通过 Jinja2 语法进行变量渲染。
    支持缓存已读取的配置文件以提高性能。
    """

    def __init__(self, template_dir: str | Path | None = None):
        """初始化 Prompt 模板管理器

        Args:
            template_dir: 模板文件所在的根目录。如果未提供，默认使用当前文件
                同级目录下的 `templates` 文件夹。
        """
        self.template_dir = (
            Path(template_dir) if template_dir else Path(__file__).parent / "templates"
        )

        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)), autoescape=select_autoescape()
        )

    def _resolve_path(self, prompt_path: str | Path) -> Path:
        """统一路径解析逻辑：处理绝对路径、相对路径和后缀

        如果传入的是相对路径，将相对于 `template_dir` 进行解析。
        如果路径没有后缀，默认添加 `.yaml` 后缀。

        Args:
            prompt_path: 相对或绝对路径，可以不带后缀名（默认补充 .yaml）

        Returns:
            Path: 解析后的绝对或完整 Path 对象
        """
        p = Path(prompt_path)
        if not p.is_absolute():
            p = self.template_dir / p

        return p.with_suffix(".yaml") if not p.suffix else p

    @lru_cache(maxsize=128)
    def _get_raw_config(self, full_path: str) -> dict[str, Any]:
        """读取并解析 YAML 原始数据（带 LRU 缓存）

        Args:
            full_path: YAML 文件的绝对路径字符串

        Raises:
            FileNotFoundError: 当指定路径的文件不存在时抛出

        Returns:
            dict[str, Any]: 解析后的 YAML 字典数据
        """
        if not Path(full_path).exists():
            raise FileNotFoundError(f"Prompt file not found: {full_path}")

        with open(full_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_model_config(self, prompt_path: str | Path) -> dict[str, Any]:
        """获取该 Prompt 对应的模型配置

        读取指定 YAML 文件中的 `model_config` 字段。
        可以用于配置对应的大语言模型参数，如 `temperature` 等。

        Args:
            prompt_path: Prompt 文件的名称或路径。

        Returns:
            dict[str, Any]: 解析出的模型配置字典。如果配置不存在则返回空字典 {}。
        """
        # 1. 解析路径
        p = self._resolve_path(prompt_path)

        # 2. 从缓存（或磁盘）获取原始配置
        config = self._get_raw_config(str(p))

        # 3. 返回配置项
        return config.get("model_config", {})

    def build_prompt(self, prompt_path: str | Path, **kwargs: Any) -> list[dict[str, str]]:
        """渲染 Prompt 消息列表

        读取指定 YAML 文件中的 `messages` 字段，并使用 Jinja2 渲染其中的变量。

        Args:
            prompt_path: Prompt 文件的名称或路径。
            **kwargs: 用于在 Jinja2 模板中渲染的变量键值对。

        Returns:
            list[dict[str, str]]: 渲染后的消息列表，每个消息包含 `role` 和 `content` 两个键。
        """
        p = self._resolve_path(prompt_path)
        config = self._get_raw_config(str(p))
        messages = config.get("messages", [])

        rendered_messages = []
        for msg in messages:
            # 复用 jinja_env 的从字符串创建模板功能
            template = self.jinja_env.from_string(msg["content"])
            rendered_messages.append({"role": msg["role"], "content": template.render(**kwargs)})

        return rendered_messages


# 实例化一个单例对象，供外部直接导入使用
prompt_manager = PromptManager()
