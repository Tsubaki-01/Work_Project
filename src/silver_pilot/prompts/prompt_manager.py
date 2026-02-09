from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template


class PromptManager:
    def __init__(self, template_dir: str | Path | None = None):
        # 默认定位到当前文件所在目录下的 templates 文件夹
        if template_dir is None:
            self.template_dir: Path = Path(__file__).parent / "templates"
        else:
            self.template_dir = Path(template_dir)

        self._cache: dict[str, dict[str, Any]] = {}  # 可选：用于缓存读取过的 yaml 文件

    def load_prompt(self, relative_path: str | Path) -> dict[str, Any]:
        """读取原始 YAML 配置"""
        # 如果已经是绝对路径，直接使用；否则与 template_dir 拼接
        input_path = Path(relative_path)
        if input_path.is_absolute():
            path = input_path
        else:
            path = self.template_dir / input_path

        # 自动补全后缀
        if not path.suffix:
            path = path.with_suffix(".yaml")

        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        # 简单的缓存机制，避免频繁 IO
        if str(path) in self._cache:
            return self._cache[str(path)]  # type: ignore[no-any-return]

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            self._cache[str(path)] = data
            return data  # type: ignore[no-any-return]

    def render(self, prompt_path: str | Path, **kwargs) -> list[dict[str, str]]:  # type: ignore[no-untyped-def]
        """
        读取并渲染 Prompt。
        :param prompt_path: 相对路径，例如 'agent/reasoning'
        :param kwargs: 注入到 Jinja2 模板中的变量
        :return: 格式化好的 messages 列表，可直接传给 LLM API
        """
        config = self.load_prompt(prompt_path)
        messages = config.get("messages", [])

        rendered_messages = []
        for msg in messages:
            template = Template(msg["content"])
            rendered_content = template.render(**kwargs)
            rendered_messages.append({"role": msg["role"], "content": rendered_content})

        return rendered_messages

    def get_model_config(self, prompt_path: str) -> dict[str, Any]:
        """获取该 Prompt 对应的模型配置"""
        config = self.load_prompt(prompt_path)
        return config.get("model_config", {})  # type: ignore[no-any-return]


# 实例化一个单例对象
prompt_manager = PromptManager()
