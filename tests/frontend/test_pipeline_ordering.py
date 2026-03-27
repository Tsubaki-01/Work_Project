from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    shutil.which("node") is None,
    reason="node is required for frontend ordering test",
)
def test_pipeline_ordering_node_contract() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    js = """
const assert = require('node:assert/strict');
const util = require('./static/pipeline-ordering.js');

const items = [
  { id: 'b', group_id: '1-parallel', event_seq: 4, _insert_idx: 2 },
  { id: 'a', group_id: '0-serial', event_seq: 10, _insert_idx: 1 },
  { id: 'd', group_id: '2-post', event_seq: 1, _insert_idx: 4 },
  { id: 'c', group_id: '1-parallel', event_seq: 3, _insert_idx: 3 },
];

util.normalizePipelineOrder(items);
assert.deepEqual(items.map(x => x.id), ['a', 'c', 'b', 'd']);
assert.equal(util.groupRank('0-serial'), 0);
assert.equal(util.groupRank('1-parallel'), 1);
assert.equal(util.groupRank('2-post'), 2);
assert.equal(util.groupRank('x-other'), 99);
console.log('ok');
"""

    result = subprocess.run(
        ["node", "-e", js],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok" in result.stdout
