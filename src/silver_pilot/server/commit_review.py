"""
模块名称：commit_review
功能描述：基于给定 Git Commit 列表生成结构化 Markdown 审查报告。
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


@dataclass(slots=True)
class CommitStats:
    """单个提交的基础统计信息。"""

    commit_hash: str
    subject: str
    files: list[str]
    insertions: int
    deletions: int


def _run_git(repo_root: Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_commit_hash(commit_hash: str) -> str:
    normalized = commit_hash.strip().lower()
    if not _COMMIT_RE.fullmatch(normalized):
        raise ValueError(f"非法 commit hash: {commit_hash}")
    return normalized


def _collect_commit_stats(repo_root: Path, commit_hash: str) -> CommitStats:
    normalized = _validate_commit_hash(commit_hash)
    _run_git(repo_root, ["cat-file", "-e", f"{normalized}^{{commit}}"])

    subject = _run_git(repo_root, ["show", "-s", "--format=%s", normalized])
    numstat = _run_git(repo_root, ["show", "--numstat", "--format=", normalized])

    files: list[str] = []
    insertions = 0
    deletions = 0
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        add_str, del_str, path = parts
        files.append(path)
        if add_str.isdigit():
            insertions += int(add_str)
        if del_str.isdigit():
            deletions += int(del_str)

    return CommitStats(
        commit_hash=normalized,
        subject=subject or "(无提交标题)",
        files=files,
        insertions=insertions,
        deletions=deletions,
    )


def build_commit_review_report(repo_root: Path, commit_hashes: list[str]) -> str:
    if not commit_hashes:
        raise ValueError("commit_hashes 不能为空")

    stats_list = [_collect_commit_stats(repo_root, commit_hash) for commit_hash in commit_hashes]
    all_files = [file for stats in stats_list for file in stats.files]

    touches_server_api = any(file.startswith("src/silver_pilot/server/") for file in all_files)
    touches_frontend = any(file.startswith("static/") for file in all_files)
    touches_agent_context = any(
        file
        in {
            "src/silver_pilot/agent/nodes/helpers.py",
            "src/silver_pilot/agent/nodes/chat_agent.py",
            "src/silver_pilot/agent/nodes/medical_agent.py",
            "src/silver_pilot/agent/nodes/supervisor.py",
        }
        for file in all_files
    )
    touches_dependencies = any(file == "pyproject.toml" for file in all_files)
    touches_session_store_internal = any(
        file in {"src/silver_pilot/server/app.py", "src/silver_pilot/server/session_store.py"}
        for file in all_files
    )

    summary_items = [f"`{stats.commit_hash[:7]}` {stats.subject}" for stats in stats_list]
    total_files = len(set(all_files))
    total_insertions = sum(stats.insertions for stats in stats_list)
    total_deletions = sum(stats.deletions for stats in stats_list)
    changed_domains = []
    if touches_server_api:
        changed_domains.append("服务端 API")
    if touches_frontend:
        changed_domains.append("前端静态资源")
    if touches_agent_context:
        changed_domains.append("Agent 对话链路")
    if touches_dependencies:
        changed_domains.append("依赖配置")
    if not changed_domains:
        changed_domains.append("通用代码实现")

    compatibility_points = [
        "- **API 与接口**："
        + (
            "涉及 `src/silver_pilot/server/` 与 `static/` 的协同改动，前后端契约变更风险为 **中等**。"
            if touches_server_api or touches_frontend
            else "本次变更主要集中于内部实现，外部接口破坏风险为 **低**。"
        ),
        "- **数据与状态**："
        + (
            "新增对话上下文提取逻辑会影响 prompt 输入内容，属于行为兼容性变更；需确认旧会话回放结果是否可接受。"
            if touches_agent_context
            else "未发现明显的数据结构版本迁移动作，前后向兼容风险较低。"
        ),
        "- **依赖关系**："
        + (
            "检测到依赖文件变更，请确认锁定版本与运行环境的一致性。"
            if touches_dependencies
            else "未检测到依赖升级/替换，模块兼容性主要取决于代码逻辑。"
        ),
    ]

    conflict_points = ["- **逻辑冲突**：未发现明确互斥逻辑；建议重点验证新增逻辑在边界输入下的行为一致性。"]
    if touches_agent_context:
        conflict_points[0] = (
            "- **逻辑冲突**：对话上下文截断策略变更可能影响意图识别和回答连贯性"
            "（触发条件：多轮对话、混合模态或消息顺序异常）。"
        )
    if touches_server_api:
        conflict_points.append(
            "- **并发与状态**：会话/消息读写若缺少并发保护，可能出现状态覆盖或读取不一致"
            "（触发条件：同一 session 并发请求）。"
        )
    else:
        conflict_points.append(
            "- **并发与状态**：本次变更未直接引入新并发原语；需确认是否影响已有异步流程中的共享状态。"
        )
    conflict_points.append(
        "- **命名空间与作用域**：未发现直接命名冲突；建议保持新增字段和函数语义单一，避免跨模块同名不同义。"
    )

    optimization_blocks: list[str] = []
    if touches_agent_context:
        optimization_blocks.append(
            """1. 将上下文窗口参数化并显式复用配置，降低 magic number 扩散。

**Before**
```python
conversation_summary = get_conversation_context(state.get("messages", []))
```

**After**
```python
max_turns = state.get("context_window_turns", 6)
conversation_summary = get_conversation_context(state.get("messages", []), max_turns=max_turns)
```"""
        )
    if touches_session_store_internal:
        optimization_blocks.append(
            """2. 尽量通过公开方法维护会话状态，避免直接依赖内部属性，减少实现耦合。

**Before**
```python
store._sessions[session_id] = session
store._messages[session_id] = []
```

**After**
```python
store.create(name="新对话", user_id="default_user")
```"""
        )
    if not optimization_blocks:
        optimization_blocks.append(
            """1. 对重复条件判断进行提取，减少分支重复并提升可读性。

**Before**
```python
if cond_a and cond_b:
    do_x()
if cond_a and cond_b:
    do_y()
```

**After**
```python
is_valid = cond_a and cond_b
if is_valid:
    do_x()
    do_y()
```"""
        )

    questions: list[str] = []
    if touches_agent_context:
        questions.append(
            "- 对话上下文是否需要包含“本轮用户输入”？该策略是否经过离线回放或 A/B 验证？"
        )
    if touches_server_api and touches_frontend:
        questions.append("- 前后端接口字段变更是否有版本兼容策略（灰度、回滚或双写阶段）？")
    if touches_dependencies:
        questions.append("- 依赖升级是否已经在目标部署环境完成兼容性验证（含锁文件和镜像构建）？")
    if not questions:
        questions.append("- 当前上下文未包含业务验收标准，是否可补充关键场景与回归用例范围？")

    report = f"""📝 **变更摘要**
本次评审覆盖以下提交：{", ".join(summary_items)}。变更主要涉及：{", ".join(changed_domains)}，共涉及 **{total_files}** 个文件（+{total_insertions}/-{total_deletions}）。

🔄 **兼容性报告**
{chr(10).join(compatibility_points)}

⚠️ **潜在冲突**
{chr(10).join(conflict_points)}

🚀 **优化建议**
{chr(10).join(optimization_blocks)}

❓ **需要明确的疑问**
{chr(10).join(questions)}
"""
    return report
