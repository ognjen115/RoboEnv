"""The lab cell (two-arm xArm 7, MuJoCo) as a RoboEnv backend.

Typed devices; the task's goal is the private verdict.
"""

from roboenv_runtime.adapters.native import NativeDeviceAdapter
from roboenv_runtime.core.adapter import register_adapter
from roboenv_runtime.core.types import ScoreReport

from .rig import rig_of


@register_adapter
class LabAdapter(NativeDeviceAdapter):
    name = "lab"
    embodiment = "lab-xarm7"

    def __init__(
        self, task="open_drawer", rig=None, *, stop_on_success=False, max_steps=None, **kwargs
    ):
        if kwargs:
            raise ValueError(f"unsupported lab backend arguments: {sorted(kwargs)}")
        self._set_stop_on_success(stop_on_success)
        from .devices import LabDevices

        rig = rig or rig_of(task)
        self.embodiment = rig
        self._env = LabDevices(task, rig, max_steps=max_steps)

    def goal(self):
        return str(self._env.spec.get("sentence") or "")

    def task_context(self):
        return {}

    def score(self):
        success = self._env.task_completed()
        return ScoreReport(
            reward=float(success),
            raw_score=float(success),
            completed=success,
            detail={
                "control_protocol": "absolute_joint_position_20hz",
                "task": self._env.task_name,
            },
        )
