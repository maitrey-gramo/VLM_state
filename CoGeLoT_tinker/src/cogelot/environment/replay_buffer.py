import torch
from loguru import logger

from cogelot.data.collate import collate_variable_ndim_batch
from cogelot.structures.common import Observation, Observations
from cogelot.structures.model import ModelInstance
from cogelot.structures.vima import PoseAction, PoseActionType, VIMAInstance


class ReplayBuffer:
    """Buffer for the current episode to allow replay."""

    def __init__(self) -> None:
        self._observations: list[Observation] = []
        self._actions: list[PoseAction] = []

        self._success_per_step: list[bool] = []

        self._encoded_prompt: list[torch.Tensor] = []
        self._encoded_prompt_mask: list[torch.Tensor] = []
        self._encoded_observations: list[torch.Tensor] = []
        self._encoded_observation_masks: list[torch.Tensor] = []
        self._encoded_actions: list[torch.Tensor] = []
        self._encoded_action_masks: list[torch.Tensor] = []

    def __len__(self) -> int:
        """Get the number of steps taken."""
        return self.num_actions

    @property
    def success_per_step(self) -> list[bool]:
        """Get the success per step."""
        return self._success_per_step

    @property
    def is_successful(self) -> bool:
        """Get whether the task is successful, as per the last step."""
        try:
            return self._success_per_step[-1]
        except IndexError:
            return False

    def update_success_tracker(self, *, is_successful: bool) -> None:
        """Update the success tracker."""
        self._success_per_step.append(is_successful)

    @property
    def num_observations(self) -> int:
        """Get the number of observations."""
        return len(self._encoded_observations)

    @property
    def num_actions(self) -> int:
        """Get the number of actions."""
        return len(self._encoded_actions)

    @property
    def observations(self) -> list[Observation]:
        """Get the observations."""
        return self._observations

    @property
    def encoded_prompt(self) -> torch.Tensor:
        """Get the encoded prompt."""
        return self._encoded_prompt[0]

    @encoded_prompt.setter
    def encoded_prompt(self, encoded_prompt: torch.Tensor) -> None:
        """Set the encoded prompt."""
        if self._encoded_prompt:
            raise ValueError("Cannot set the encoded prompt twice. Need to reset buffer first.")
        self._encoded_prompt.append(encoded_prompt)

    @property
    def encoded_prompt_mask(self) -> torch.Tensor:
        """Get the encoded prompt mask."""
        return self._encoded_prompt_mask[0]

    @encoded_prompt_mask.setter
    def encoded_prompt_mask(self, encoded_prompt_mask: torch.Tensor) -> None:
        """Set the encoded prompt."""
        if self._encoded_prompt_mask:
            raise ValueError("Cannot set the encoded prompt twice. Need to reset buffer first.")
        self._encoded_prompt_mask.append(encoded_prompt_mask)

    @property
    def encoded_observations(self) -> torch.Tensor:
        """Get the encoded observations."""
        # Firstly, we need to remove the batch dim since this has a batch size of 1, and we need to
        # remove the observation dim, since each tensor has only one observation. Each tensor is
        # now 2D
        observations = [tensor.squeeze(0).squeeze(0) for tensor in self._encoded_observations]

        # Then we can use the collate function to make sure it is all padded correctly, and that
        # gives us the observation dimension back, making it 3D
        collated_observations = collate_variable_ndim_batch(observations)

        # Then we just need to add the batch dimension back to make it 4D
        collated_observations = collated_observations.unsqueeze(0)
        return collated_observations

    @property
    def encoded_observation_masks(self) -> torch.Tensor:
        """Get the encoded observation masks."""
        # Firstly, remove the batch and observation dimensions since they are just 1, leaving us a
        # 1D tensor
        masks = [tensor.squeeze(0).squeeze(0) for tensor in self._encoded_observation_masks]

        # Then we can use the collate function to make sure it is all padded correctly
        collated_masks = collate_variable_ndim_batch(masks)

        # Then we just need to add the batch dim back
        collated_masks = collated_masks.unsqueeze(0)
        return collated_masks

    @property
    def encoded_actions(self) -> torch.Tensor | None:
        """Get the encoded actions."""
        if not self._encoded_actions:
            return None

        # Remove the batch dims and the actions dim
        actions = [tensor.squeeze(0).squeeze(0) for tensor in self._encoded_actions]

        # Then we can use the collate function to make sure it is all padded correctly
        collated_actions = collate_variable_ndim_batch(actions)

        # Then we just need to add the batch dim back
        collated_actions = collated_actions.unsqueeze(0)
        return collated_actions

    @property
    def encoded_actions_mask(self) -> torch.Tensor | None:
        """Get the encoded actions mask."""
        if not self._encoded_action_masks:
            return None

        masks = [tensor.squeeze(0).squeeze(0) for tensor in self._encoded_action_masks]
        collated_masks = collate_variable_ndim_batch(masks)
        collated_masks = collated_masks.unsqueeze(0)
        return collated_masks

    def add_observation(self, observation: Observation) -> None:
        """Add an observation to the buffer."""
        self._observations.append(observation)

    def add_action(self, continuous_actions: dict[PoseActionType, torch.Tensor]) -> None:
        """Add the continuous actions to the buffer."""
        continuous_actions = {
            k: tensor.clone().detach() for k, tensor in continuous_actions.items()
        }
        parsed_actions = PoseAction.model_validate(
            {"index": len(self._actions), **continuous_actions}
        )
        self._actions.append(parsed_actions)

    def add_next_encoded_observation(
        self, encoded_observation: torch.Tensor, encoded_observation_mask: torch.Tensor
    ) -> None:
        """Add an encoded observation to the buffer."""
        self._encoded_observations.append(encoded_observation)
        self._encoded_observation_masks.append(encoded_observation_mask)

    def add_next_encoded_action(
        self, encoded_action: torch.Tensor, encoded_action_mask: torch.Tensor
    ) -> None:
        """Add an encoded action to the buffer."""
        self._encoded_actions.append(encoded_action)
        self._encoded_action_masks.append(encoded_action_mask)

    def to_model_instance(self) -> ModelInstance:
        """Convert the state into a ModelInstance."""
        return ModelInstance(
            encoded_prompt=self.encoded_prompt,
            encoded_prompt_mask=self.encoded_prompt_mask,
            encoded_observations=self.encoded_observations,
            encoded_observations_mask=self.encoded_observation_masks,
            encoded_actions=self.encoded_actions,
            encoded_actions_mask=self.encoded_actions_mask,
        )

    def update_vima_instance(self, vima_instance: VIMAInstance) -> VIMAInstance:
        """Update a VIMA Instance with the output of the buffer."""
        vima_instance.observations = Observations(self._observations)
        vima_instance.pose_actions = self._actions
        vima_instance.total_steps = len(self)
        vima_instance.is_successful_at_end = self.is_successful
        vima_instance.success_per_step = self.success_per_step
        return vima_instance

    def reset(self) -> None:
        """Reset the buffer."""
        logger.info("Resetting the state")

        self._encoded_prompt = []
        self._encoded_prompt_mask = []
        self._encoded_observations = []
        self._encoded_observation_masks = []
        self._encoded_actions = []
        self._encoded_action_masks = []

        self._success_per_step = []
        self._observations = []
        self._actions = []
