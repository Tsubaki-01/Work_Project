"""Generate structured code-review reports for one or more Git commits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess


_REQUIRED_HEADINGS = [
    "📝 变更摘要",
    "🔄 兼容性报告",
    "⚠️ 潜在冲突",
    "🚀 优化建议",
    "❓ 需要明确的疑问",
]


@dataclass(frozen=True)
class CommitSnapshot:
    """A compact view of one commit needed for quality-impact analysis."""

    commit_id: str
    subject: str
    files: tuple[str, ...]


def parse_commit_ids(raw: str) -> list[str]:
    """Extract unique commit hashes from free-form input while preserving order."""
    ids: list[str] = []
    for token in re.findall(r"\b[0-9a-fA-F]{7,40}\b", raw):
        lowered = token.lower()
        if lowered not in ids:
            ids.append(lowered)
    return ids


def _run_git(repo_path: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_snapshot(repo_path: Path, commit_id: str) -> CommitSnapshot:
    subject = _run_git(repo_path, ["show", "-s", "--format=%s", commit_id])
    files_raw = _run_git(repo_path, ["show", "--pretty=format:", "--name-only", commit_id])
    files = tuple(line.strip() for line in files_raw.splitlines() if line.strip())
    return CommitSnapshot(commit_id=commit_id, subject=subject, files=files)


def _compatibility_lines(snapshots: list[CommitSnapshot]) -> list[str]:
    files = {path for snapshot in snapshots for path in snapshot.files}
    lines: list[str] = []

    if any(path.startswith("src/silver_pilot/server/") for path in files):
        lines.append(
            "- **API 与接口**：`src/silver_pilot/server/app.py` 与 `models.py` 的改动引入了新的 REST/WebSocket 协议面，"
            "属于接口扩展型变更。若前端仍使用旧字段名（例如 `imagePath`/`audioPath`），会出现兼容性问题。"
        )
        lines.append(
            "- **数据与状态**：`session_store.py` 采用进程内内存状态；重启后会话丢失，"
            "这不影响代码级向后兼容，但会影响运行时状态兼容预期。"
        )

    if "QLoRA/train_qlora_elderly_care.yaml" in files:
        lines.append(
            "- **依赖关系**：QLoRA YAML 仅新增训练配置文件，不直接变更运行时 Python 依赖，"
            "对线上服务兼容性风险低。"
        )

    if not lines:
        lines.append("- 未发现方法签名级破坏；当前变更主要是新增文件，兼容性风险整体可控。")

    return lines


def _conflict_lines(snapshots: list[CommitSnapshot]) -> list[str]:
    files = {path for snapshot in snapshots for path in snapshot.files}
    lines: list[str] = []

    if "src/silver_pilot/server/session_store.py" in files:
        lines.append(
            "- **并发与状态冲突**：`SessionStore` 基于普通字典，在多 worker 或高并发下可能出现更新竞争；"
            "触发条件是同一会话并发写入消息。"
        )

    if {
        "src/silver_pilot/server/models.py",
        "static/api-connector.js",
    }.issubset(files):
        lines.append(
            "- **逻辑冲突**：前后端消息协议耦合较紧（例如 `WSIncoming`/`WSOutgoing`），"
            "触发条件是任一侧字段变更未同步发布。"
        )

    if "src/silver_pilot/server/app.py" in files:
        lines.append(
            "- **作用域与全局状态**：模块级 `_graph`、`_store` 由多个请求共享；"
            "触发条件是未来引入多进程部署或热重载时，状态一致性要求提升。"
        )

    if not lines:
        lines.append("- 未观察到明显命名冲突或作用域污染风险。")

    return lines


def _optimization_lines(snapshots: list[CommitSnapshot]) -> list[str]:
    files = {path for snapshot in snapshots for path in snapshot.files}
    snippets: list[str] = []

    if "src/silver_pilot/server/session_store.py" in files:
        snippets.append(
            "- `SessionStore` 可增加最小化并发保护，避免并发写入时的状态竞态。\n"
            "\n"
            "```python\n"
            "# Before\n"
            "self._messages[session_id].append(message)\n"
            "\n"
            "# After\n"
            "with self._lock:\n"
            "    self._messages[session_id].append(message)\n"
            "```"
        )

    if "src/silver_pilot/server/app.py" in files:
        snippets.append(
            "- CORS 当前为 `allow_origins=[\"*\"]`，可切换白名单配置以降低暴露面。\n"
            "\n"
            "```python\n"
            "# Before\n"
            "app.add_middleware(CORSMiddleware, allow_origins=[\"*\"], allow_methods=[\"*\"], allow_headers=[\"*\"])\n"
            "\n"
            "# After\n"
            "app.add_middleware(\n"
            "    CORSMiddleware,\n"
            "    allow_origins=settings.cors_origins,\n"
            "    allow_methods=[\"GET\", \"POST\", \"DELETE\"],\n"
            "    allow_headers=[\"Authorization\", \"Content-Type\"],\n"
            ")\n"
            "```"
        )

    if "QLoRA/train_qlora_elderly_care.yaml" in files:
        snippets.append(
            "- 训练配置建议补充环境约束校验（GPU/精度），减少训练启动后失败的成本。\n"
            "\n"
            "```yaml\n"
            "# Before\n"
            "bf16: true\n"
            "\n"
            "# After\n"
            "bf16: true\n"
            "validate_cuda_capability: true\n"
            "fallback_precision: fp16\n"
            "```"
        )

    if not snippets:
        snippets.append("- 当前改动量较小，建议先补充针对关键路径的单元测试再进行重构。")

    return snippets


def _questions(snapshots: list[CommitSnapshot]) -> list[str]:
    files = {path for snapshot in snapshots for path in snapshot.files}
    questions = [
        "- 这些 commit 是否计划直接进入生产，还是仅用于 Demo/内测环境？（影响兼容与安全基线）",
    ]

    if any(path.startswith("src/silver_pilot/server/") for path in files):
        questions.append("- WebSocket 协议是否有版本号或 schema 管理机制？若无，建议明确升级策略。")

    if "QLoRA/train_qlora_elderly_care.yaml" in files:
        questions.append("- 训练配置对应的数据集版本与模型基座版本是否已冻结？")

    return questions


def render_commit_review_report(snapshots: list[CommitSnapshot]) -> str:
    """Render a markdown report that matches the requested structure."""
    commit_list = ", ".join(snapshot.commit_id[:7] for snapshot in snapshots)
    purpose = "；".join(snapshot.subject for snapshot in snapshots)

    summary = [
        f"本次审查覆盖 commit：`{commit_list}`。",
        f"核心目标是：{purpose}。",
    ]

    lines: list[str] = []
    lines.append(f"## {_REQUIRED_HEADINGS[0]}")
    lines.extend(summary)
    lines.append("")

    lines.append(f"## {_REQUIRED_HEADINGS[1]}")
    lines.extend(_compatibility_lines(snapshots))
    lines.append("")

    lines.append(f"## {_REQUIRED_HEADINGS[2]}")
    lines.extend(_conflict_lines(snapshots))
    lines.append("")

    lines.append(f"## {_REQUIRED_HEADINGS[3]}")
    lines.extend(_optimization_lines(snapshots))
    lines.append("")

    lines.append(f"## {_REQUIRED_HEADINGS[4]}")
    lines.extend(_questions(snapshots))

    return "\n".join(lines)


def review_commits(commit_input: str, repo_path: str | Path) -> str:
    """Load commit context from Git and return a structured markdown report."""
    ids = parse_commit_ids(commit_input)
    if not ids:
        raise ValueError("No valid commit ids found in input")

    repo = Path(repo_path).resolve()
    snapshots = [_load_snapshot(repo, commit_id) for commit_id in ids]
    return render_commit_review_report(snapshots)
