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

"""Load RBench's task-local LeRobot repositories through the OpenPI SFT pipeline."""

from pathlib import Path

import openpi.transforms as transforms
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from omegaconf import DictConfig, ListConfig, OmegaConf
from openpi.training import data_loader
from openpi.training.config import DataConfig
from torch.utils.data import ConcatDataset, DistributedSampler

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


def _load_tasks(paths: list[Path], data: DataConfig, horizon: int) -> ConcatDataset:
    task_metadata = []
    for path in paths:
        if not (path / "meta/info.json").is_file():
            raise FileNotFoundError(f"RBench LeRobot metadata not found: {path}")
        metadata = LeRobotDatasetMetadata(path.name, root=path)
        if not metadata.tasks or any(
            not prompt.strip() or prompt.strip().startswith("__PENDING__")
            for prompt in metadata.tasks.values()
        ):
            raise ValueError(f"RBench language instructions are incomplete: {path}")
        if metadata.fps != 20:
            raise ValueError(f"Expected 20 Hz for RBench task {path}")
        task_metadata.append((path, metadata))

    datasets = []
    for path, metadata in task_metadata:
        dataset = LeRobotDataset(
            path.name,
            root=path,
            delta_timestamps={
                key: [step / metadata.fps for step in range(horizon)]
                for key in data.action_sequence_keys
            },
        )
        datasets.append(
            data_loader.TransformedDataset(
                dataset, [transforms.PromptFromLeRobotTask(metadata.tasks)]
            )
        )
    return ConcatDataset(datasets)


def build_rbench_sft_dataloader(
    cfg: DictConfig,
    world_size: int,
    rank: int,
    data_paths: ListConfig | list[str] | tuple[str, ...],
    eval_dataset: bool = False,
) -> tuple[data_loader.DataLoaderImpl, DataConfig]:
    """Build a rank-aware SFT loader over every configured RBench task."""
    if not isinstance(data_paths, (ListConfig, list, tuple)) or not data_paths:
        raise ValueError("RBench SFT expects a non-empty list of task dataset paths")
    paths = [Path(path).resolve() for path in data_paths]
    if len(paths) != len(set(paths)):
        raise ValueError("RBench SFT task dataset paths must be unique")

    model = cfg.actor.model
    data_kwargs = OmegaConf.to_container(model.openpi_data, resolve=True)
    if not Path(data_kwargs["norm_stats_path"]).is_file():
        raise FileNotFoundError(data_kwargs["norm_stats_path"])
    config = get_openpi_config(
        model.openpi.config_name, model_path=model.model_path, data_kwargs=data_kwargs
    )
    horizon = config.model.action_horizon
    if (
        model.action_dim != 8
        or model.openpi.model_action_dim != config.model.action_dim
        or model.openpi.action_horizon != horizon
        or model.openpi.discrete_state_input != config.model.discrete_state_input
    ):
        raise ValueError("RBench SFT model and dataset action/state settings differ")
    data = config.data.create(config.assets_dirs, config.model)
    if data.norm_stats is None:
        raise ValueError("RBench SFT requires normalization statistics")

    dataset = data_loader.transform_dataset(_load_tasks(paths, data, horizon), data)
    seed = int(cfg.actor.seed)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=not eval_dataset,
            seed=seed,
        )
        if world_size > 1
        else None
    )
    batch_size = (
        cfg.actor.get("eval_batch_size", cfg.actor.micro_batch_size)
        if eval_dataset
        else cfg.actor.micro_batch_size
    )
    loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        shuffle=sampler is None and not eval_dataset,
        sampler=sampler,
        num_workers=int(cfg.data.get("num_workers", config.num_workers)),
        seed=seed,
        framework="pytorch",
    )
    return data_loader.DataLoaderImpl(data, loader), data
