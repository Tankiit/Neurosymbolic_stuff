"""Training utilities for learning the action selector from saved rollouts."""

from __future__ import annotations

from typing import Iterable, List

from ..controller import ActionValueController
from ..reasoning import RolloutResult


def train_controller_from_rollouts(rollouts: Iterable[RolloutResult]) -> ActionValueController:
    """Fit the supervised controller to saved rollout outcomes."""
    controller = ActionValueController()
    return controller.fit(rollouts)
