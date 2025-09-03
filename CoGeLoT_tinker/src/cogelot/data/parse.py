import pickle
from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, get_args

import numpy as np
import torch
from PIL import Image

from cogelot.structures.common import Observation, Observations, PromptAssets
from cogelot.structures.vima import (
    ObjectMetadata,
    Partition,
    PoseAction,
    PoseActionType,
    Task,
    VIMAInstance,
)

POSE_ACTION_KEYS: tuple[PoseActionType, ...] = get_args(PoseActionType)
ACTIONS_FILE_NAME = "action.pkl"
TRAJECTORY_METADATA_FILE_NAME = "trajectory.pkl"
OBSERVATIONS_FILE_NAME = "obs.pkl"
RGB_PATH_PER_VIEW: Mapping[str, str] = MappingProxyType({"top": "rgb_top", "front": "rgb_front"})


def _load_data_from_pickle(pickled_file: Path) -> Any:
    """Load the data from a pickle file."""
    return pickle.load(pickled_file.open("rb"))  # noqa: S301


def get_all_raw_instance_directories(
    raw_data_dir: Path, *, task_filter: Task | None = None
) -> Iterator[Path]:
    """Get all the instance directories."""
    # If there is a task filter, then we need to get all the possible task names to iterate from,
    # and then we get all the instance paths from that directory
    if task_filter is not None:
        task_dir_names = Task.names_for_value(task_filter.value)
        path_generators = (raw_data_dir.glob(f"{dir_name}/*/") for dir_name in task_dir_names)
        return (path for path_globber in path_generators for path in path_globber)

    # Otherwise, we just get all of the instance paths
    return raw_data_dir.glob("*/*/")


def load_rgb_observation_image(
    *, instance_dir: Path, view: Literal["top", "front"], frame_idx: int
) -> torch.Tensor:
    """Load the RGB image of the observation for the given view."""
    image_path = instance_dir.joinpath(RGB_PATH_PER_VIEW[view], f"{frame_idx}.jpg")
    with Image.open(image_path) as image:
        image.draft(image.mode, image.size)
        # Also move the axes to be in the same structure as the prompt assets
        return torch.from_numpy(np.asarray(image).copy()).moveaxis(-1, 0).contiguous()


def parse_object_metadata(trajectory_metadata: dict[str, Any]) -> list[ObjectMetadata]:
    """Extract and parse object metadata from the trajectory metadata."""
    object_metadata: list[ObjectMetadata] = []

    for object_id, object_info in trajectory_metadata["obj_id_to_info"].items():
        object_metadata.append(
            ObjectMetadata(
                obj_id=object_id,
                obj_name=object_info["obj_name"],
                obj_asset_name=object_info["obj_assets"],
                texture_name=object_info["texture_name"],
            )
        )

    return object_metadata


def parse_pose_actions(instance_dir: Path) -> list[PoseAction]:
    """Parse the pose actions."""
    raw_action_data = _load_data_from_pickle(instance_dir.joinpath(ACTIONS_FILE_NAME))
    num_actions = len(raw_action_data[POSE_ACTION_KEYS[0]])
    actions_dict = {key: torch.from_numpy(raw_action_data[key]) for key in POSE_ACTION_KEYS}
    actions = [
        PoseAction(
            index=action_idx,
            pose0_position=actions_dict["pose0_position"][action_idx],
            pose1_position=actions_dict["pose1_position"][action_idx],
            pose0_rotation=actions_dict["pose0_rotation"][action_idx],
            pose1_rotation=actions_dict["pose1_rotation"][action_idx],
        )
        for action_idx in range(num_actions)
    ]
    return actions


def parse_observations(instance_dir: Path) -> list[Observation]:
    """Parse observations from raw data."""
    raw_obs_data = _load_data_from_pickle(instance_dir.joinpath(OBSERVATIONS_FILE_NAME))
    raw_segmentation_data = raw_obs_data["segm"]
    num_obserations = len(raw_segmentation_data["top"])

    observations: list[Observation] = [
        Observation.model_validate(
            {
                "index": obs_idx,
                "rgb": {
                    "front": load_rgb_observation_image(
                        instance_dir=instance_dir, view="front", frame_idx=obs_idx
                    ),
                    "top": load_rgb_observation_image(
                        instance_dir=instance_dir, view="top", frame_idx=obs_idx
                    ),
                },
                "segm": {
                    "front": torch.from_numpy(raw_segmentation_data["front"][obs_idx]),
                    "top": torch.from_numpy(raw_segmentation_data["top"][obs_idx]),
                },
            }
        )
        for obs_idx in range(num_obserations)
    ]

    return observations


class VIMAInstanceParser:
    """Parser for VIMA instances.

    Since the logic can get complicated, this class makes it easier to control how we make variants
    from the original dataset, without needing to change everything-else in all the other places.

    The logic can be a little confusing, but this seemed like the best way to do it given that
    things are handled a little confusingly in the original data. For more detail, check out the
    tests.
    """

    def __init__(self, *, keep_null_action: bool) -> None:
        super().__init__()
        self.keep_null_action = keep_null_action

    def __call__(self, instance_dir: Path) -> VIMAInstance:
        """Parse the instance from the instance directory."""
        return self.parse_from_instance_dir(instance_dir)

    def create_partial(self) -> Callable[[Path], VIMAInstance]:
        """Create a partial function that parses the instances from the directory.

        Note: This creates new objects to serve as the function, but it's not a problem because
        this is a lightweight class.
        """
        return deepcopy(self).parse_from_instance_dir

    def parse_from_instance_dir(self, instance_dir: Path) -> VIMAInstance:
        """Create a VIMAInstance from their instance dir."""
        trajectory_metadata = _load_data_from_pickle(
            instance_dir.joinpath(TRAJECTORY_METADATA_FILE_NAME)
        )

        pose_actions = self._parse_pose_actions(instance_dir)
        observations = self._parse_observations(instance_dir, num_pose_actions=len(pose_actions))

        if len(observations) != len(pose_actions):
            raise ValueError(
                f"Number of observations ({len(observations)}) does not match number of pose actions "
                f"({len(pose_actions)}) for instance {instance_dir}"
            )

        success_per_step = self._parse_success_per_step(
            num_actions=len(pose_actions), is_successful_at_end=trajectory_metadata["success"]
        )
        return VIMAInstance(
            partition=Partition.training,
            index=int(instance_dir.stem),
            task=Task[instance_dir.parent.stem],
            total_steps=trajectory_metadata["steps"],
            prompt=trajectory_metadata["prompt"],
            prompt_assets=PromptAssets.from_raw_prompt_assets(
                trajectory_metadata["prompt_assets"]
            ),
            end_effector_type=trajectory_metadata["end_effector_type"],
            object_metadata=parse_object_metadata(trajectory_metadata),
            pose_actions=pose_actions,
            observations=Observations(observations),
            generation_seed=trajectory_metadata["seed"],
            is_successful_at_end=trajectory_metadata["success"],
            success_per_step=success_per_step,
            # TODO: This is currently wrong for parsed datasets and just a placeholder to prevent breaking.
            minimum_steps=0,
        )

    def _parse_pose_actions(self, instance_dir: Path) -> list[PoseAction]:
        """Parse the pose actions for the instance."""
        pose_actions = parse_pose_actions(instance_dir)

        # If we are keeping the null action, then we need to add it on since the data does not come
        # with it.
        if self.keep_null_action:
            pose_actions.append(PoseAction.get_null_action())

        return pose_actions

    def _parse_observations(
        self, instance_dir: Path, *, num_pose_actions: int
    ) -> list[Observation]:
        """Parse the observations for the instance."""
        observations = parse_observations(instance_dir)

        # If we are not keeping the null action, then we need to make sure that we do not include
        # any observations that are past the number of pose actions, since the original data
        # contains the correct number of observations + 1 for after the final action took place.
        if not self.keep_null_action:
            observations = observations[:num_pose_actions]

        return observations

    def _parse_success_per_step(
        self, *, num_actions: int, is_successful_at_end: bool
    ) -> list[bool]:
        """Create the success tracker.

        Since these were created from expert demonstrations, we assume that the assumption held
        during eval also holds --- that the task is successful if the last action is successful.
        """
        tracker = [False for _ in range(num_actions)]
        if not is_successful_at_end:
            return tracker

        if is_successful_at_end:
            tracker[-1] = True
        if is_successful_at_end and self.keep_null_action:
            tracker[-2] = True
        return tracker
