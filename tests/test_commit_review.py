from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from silver_pilot.server.commit_review import (
    SECTION_COMPATIBILITY,
    SECTION_CONFLICTS,
    SECTION_OPTIMIZATION,
    SECTION_QUESTIONS,
    SECTION_SUMMARY,
    build_commit_review_report,
)


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

        self.assertIn(SECTION_SUMMARY, report)
        self.assertIn(SECTION_COMPATIBILITY, report)
        self.assertIn(SECTION_CONFLICTS, report)
        self.assertIn(SECTION_OPTIMIZATION, report)
        self.assertIn(SECTION_QUESTIONS, report)
        self.assertIn(latest_commit[:7], report)

    def test_invalid_commit_hash_raises(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            build_commit_review_report(repo_root, ["not-a-commit"])


if __name__ == "__main__":
    unittest.main()
