from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from silver_pilot.server.commit_review import build_commit_review_report


class CommitReviewReportTests(unittest.TestCase):
    def test_build_report_contains_required_sections(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        latest_commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
        report = build_commit_review_report(
            repo_root,
            [latest_commit],
        )

        self.assertIn("📝 **变更摘要**", report)
        self.assertIn("🔄 **兼容性报告**", report)
        self.assertIn("⚠️ **潜在冲突**", report)
        self.assertIn("🚀 **优化建议**", report)
        self.assertIn("❓ **需要明确的疑问**", report)
        self.assertIn(latest_commit[:7], report)

    def test_invalid_commit_hash_raises(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            build_commit_review_report(repo_root, ["not-a-commit"])


if __name__ == "__main__":
    unittest.main()
