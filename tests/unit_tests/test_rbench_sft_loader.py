# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression checks for mixing task-local RBench language and action windows."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("openpi")
pytest.importorskip("lerobot")

from openpi.training.config import DataConfig  # noqa: E402

from rlinf.data.datasets.openpi import rbench  # noqa: E402


def test_rbench_mixes_task_local_instruction_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read both LeRobot roots and keep their local task-index mappings."""
    roots = [tmp_path / task for task in ("task-a", "task-b")]
    for root in roots:
        (root / "meta").mkdir(parents=True)
        (root / "meta/info.json").touch()

    def metadata(repo_id: str, *, root: Path) -> SimpleNamespace:
        assert repo_id == root.name
        return SimpleNamespace(fps=20, tasks={0: root.name})

    def dataset(
        repo_id: str, *, root: Path, delta_timestamps: dict[str, list[float]]
    ) -> list[dict]:
        assert repo_id == root.name
        assert delta_timestamps == {"actions": [0.0, 0.05, 0.1]}
        return [{"task_index": np.int64(0)}]

    monkeypatch.setattr(rbench, "LeRobotDatasetMetadata", metadata)
    monkeypatch.setattr(rbench, "LeRobotDataset", dataset)
    mixed = rbench._load_tasks(roots, DataConfig(), horizon=3)
    assert len(mixed) == 2
    assert [sample["prompt"] for sample in mixed] == ["task-a", "task-b"]
