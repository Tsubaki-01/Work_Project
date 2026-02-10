from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape


class PromptManager:
    def __init__(self, template_dir: str | Path | None = None):
        self.template_dir = (
            Path(template_dir) if template_dir else Path(__file__).parent / "templates"
        )
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)), autoescape=select_autoescape()
        )

    def _resolve_path(self, prompt_path: str | Path) -> Path:
        """统一路径解析逻辑：处理绝对路径、相对路径和后缀"""
        p = Path(prompt_path)
        if not p.is_absolute():
            p = self.template_dir / p
        return p.with_suffix(".yaml") if not p.suffix else p

    @lru_cache(maxsize=128)
    def _get_raw_config(self, full_path: str) -> dict[str, Any]:
        """读取并解析 YAML 原始数据（带 LRU 缓存）"""
        if not Path(full_path).exists():
            raise FileNotFoundError(f"Prompt file not found: {full_path}")
        with open(full_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_model_config(self, prompt_path: str | Path) -> dict[str, Any]:
        """获取该 Prompt 对应的模型配置"""
        # 1. 解析路径
        p = self._resolve_path(prompt_path)
        # 2. 从缓存（或磁盘）获取原始配置
        config = self._get_raw_config(str(p))
        # 3. 返回配置项
        return config.get("model_config", {})

    def render(self, prompt_path: str | Path, **kwargs) -> list[dict[str, str]]:  # type: ignore[no-untyped-def]
        """渲染 Prompt 消息列表"""
        p = self._resolve_path(prompt_path)
        config = self._get_raw_config(str(p))
        messages = config.get("messages", [])

        rendered_messages = []
        for msg in messages:
            # 复用 jinja_env 的从字符串创建模板功能
            template = self.jinja_env.from_string(msg["content"])
            rendered_messages.append({"role": msg["role"], "content": template.render(**kwargs)})
        return rendered_messages


# 实例化一个单例对象
prompt_manager = PromptManager()
