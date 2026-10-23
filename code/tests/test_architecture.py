from __future__ import annotations

import pytest

from cot_control.controllers import DSPyController, FixedController, TorchRLController
from cot_control.reasoning import Action

VALID_ACTIONS = {action.value for action in Action}


@pytest.mark.parametrize(
    "controller_cls", [FixedController, DSPyController, TorchRLController]
)
def test_controllers_return_valid_actions(controller_cls):
    controller = controller_cls()
    for state_text in [
        "Problem: x | History: empty",
        "Problem: x | History: error in step 2",
        "Problem: x | History: branch point",
        "Problem: x | History: answer is ready",
    ]:
        assert controller.choose_action(state_text) in VALID_ACTIONS


def test_dspy_signatures_declare_expected_fields():
    pytest.importorskip("dspy")
    from cot_control.programs.signatures import ContinueStep, SelectAction

    assert set(ContinueStep.input_fields) == {"question", "subgoals", "history"}
    assert set(ContinueStep.output_fields) == {"next_step"}
    assert "action" in SelectAction.output_fields
