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

"""OpenPI transforms for RBench's single-arm, two-camera demonstrations."""

import dataclasses
from pathlib import Path

import openpi.transforms as transforms
from openpi.models.model import BaseModelConfig
from openpi.policies.libero_policy import LiberoInputs
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override


def _output_rbench_actions(data: transforms.DataDict) -> transforms.DataDict:
    return {"actions": data["actions"][..., :8]}


@dataclasses.dataclass(frozen=True)
class LeRobotRBenchDataConfig(DataConfigFactory):
    """Use relative joint angles and an absolute gripper width for RBench."""

    @override
    def create(self, assets_dirs: Path, model_config: BaseModelConfig) -> DataConfig:
        mask = transforms.make_bool_mask(7, -1)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            prompt_from_task=True,
            repack_transforms=transforms.Group(
                inputs=[
                    transforms.RepackTransform(
                        {
                            "observation/image": "image",
                            "observation/wrist_image": "wrist_image",
                            "observation/state": "state",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            data_transforms=transforms.Group(
                inputs=[
                    LiberoInputs(model_type=model_config.model_type),
                    transforms.DeltaActions(mask),
                ],
                outputs=[transforms.AbsoluteActions(mask), _output_rbench_actions],
            ),
            model_transforms=ModelTransformFactory()(model_config),
        )
