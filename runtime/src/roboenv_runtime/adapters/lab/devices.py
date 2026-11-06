"""The lab cell's MuJoCo world behind RoboEnv's native device contract.

The world is built exactly as lab-sim's runner builds it (sim.runner.build_world: the task's
scene, seed and goal). RoboEnv's world executor owns time: every control interval DeviceControl
applies all arms' latest targets together and advances the cell one 20 Hz control tick (its fixed
number of physics steps). The task's goal predicate is the private verdict; nothing here exposes
object poses to devices.
"""

import hashlib
import json
import sys

import os
from pathlib import Path

import numpy as np
import yaml

from roboenv_runtime.adapters.device_control import DeviceControl

from .rig import RIGS, describe, verify


class LabDevices:
    unbounded = False

    def __init__(self, task, rig, *, max_steps=None):
        from roboenv_runtime.paths import lab_root

        if rig not in RIGS or task not in RIGS[rig]:
            raise ValueError(f"lab rig {rig!r} does not declare task {task!r}")
        if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
            raise ValueError("max_steps must be positive or null for the task's budget")
        self.root = lab_root()
        self.task_name, self.embodiment = task, rig
        self.task_dir = self.root / "tasks" / task
        self.spec = yaml.safe_load((self.task_dir / "task.yaml").read_text())
        self.spec["_dir"] = self.task_dir
        self.requested_frame_limit = max_steps
        self.frame_limit = max_steps
        self.asset_roots = {}  # meshes live in the pinned lab_sim source (sources_sha256)
        self.cell = None
        self.control = None
        self.device_facts, self.rig_frames = {}, {}
        self.initialization = {}
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        # the static facts are known before the first reset (catalog generation reads them)
        self.reset(seed=1)

    # -- lifecycle ---------------------------------------------------------------------------------
    def reset(self, *, seed=None):
        from cell import spec as tspec
        from sim import runner

        self.close()
        seed = 1 if seed is None else int(seed)
        self.seed = seed
        self.cell = runner.build_world(self.spec, seed)
        self.cell.budget_t = float("inf")  # RoboEnv's executor owns the horizon (frame_limit)
        if os.environ.get("LAB_RECORD_DIR"):  # owner-side recording: the trajectory for the film, every tick
            self.cell.record_every = 2
        self._jaw_half_max = float(tspec.JAW_HALF_MAX)
        self.device_facts, self.rig_frames = describe(self.cell, self.embodiment)
        self.control_interval = 1.0 / self.cell.ctrl_hz
        budget_t = float((self.spec.get("sim") or {}).get("budget_t", 300.0))
        self.task_frame_limit = int(round(budget_t * self.cell.ctrl_hz))
        self.frame_limit = (
            None if self.unbounded else (self.requested_frame_limit or self.task_frame_limit)
        )
        limits, targets = {}, {}
        for name, f in self.rig_frames.items():
            joints = self.device_facts[f["arm"]]["intrinsics"]["model"]["joints"]
            limits[f["arm"]] = [(j["limits"]["min"], j["limits"]["max"]) for j in joints]
            targets[f["arm"]] = [float(v) for v in self.cell.q_cmd[name]]
            limits[f["tool_device"]] = [(0.0, 1.0)]
            targets[f["tool_device"]] = [self._tool_target(name)]
        self.control = DeviceControl(
            limits,
            targets,
            self._apply,
            self._advance,
            exhausted=self._exhausted,
            interval_s=self.control_interval,
        )
        verify(self.cell, self.device_facts, self.rig_frames)
        self.initialization = {
            "rig": self.embodiment,
            "task": self.task_name,
            "seed": seed,
            "control_frequency_hz": float(self.cell.ctrl_hz),
            "physics_steps_per_control": int(self.cell.decimation),
            "physics_timestep_s": float(self.cell.model.opt.timestep),
            "max_steps": self.frame_limit,
            "task_budget_s": budget_t,
            "protocol": "absolute_joint_position_20hz",
            "loaded_state_sha256": self._state_hash(),
        }
        from roboenv_runtime.adapters.native_record import initialize_record

        initialize_record(
            self,
            {
                "task": self.task_name,
                "rig": self.embodiment,
                "seed": seed,
                "max_steps": self.frame_limit,
            },
            "lab",
        )

    def close(self):
        if self.cell is not None:
            if os.environ.get("LAB_RECORD_DIR") and getattr(self.cell, "_traj", None):
                self._write_recording(Path(os.environ["LAB_RECORD_DIR"]))
            self.cell.close()
        self.cell = None
        self.control = None

    def _write_recording(self, out):
        """The episode as lab-sim records a run, for the owner only (LAB_RECORD_DIR, never the agent's devices):
        telemetry.npz (every control tick), episode.yaml (what the seed drew) and film.mp4 (the task's film
        cameras)."""
        import yaml

        out.mkdir(parents=True, exist_ok=True)
        tele = self.cell.telemetry()
        if tele:
            np.savez_compressed(out / "telemetry.npz", **tele)
        draw = getattr(self.cell, "draw", None) or {}
        if draw:
            (out / "episode.yaml").write_text(
                yaml.safe_dump({"task": self.task_name, **draw}, sort_keys=False)
            )
        try:
            fixed = list((self.spec.get("sim") or {}).get("film_cameras") or [])
            self.cell.render_film(out / "film.mp4", fps=10, fixed=fixed)
        except Exception as exc:  # a film that fails leaves the rest
            (out / "film.error.txt").write_text(f"{type(exc).__name__}: {exc}\n")

    # -- control -----------------------------------------------------------------------------------
    def _tool_target(self, name):
        cmd = self.cell.tool_cmd[name]
        if self.rig_frames[name]["tool"] == "jaw":
            return float(np.clip(float(cmd) / self._jaw_half_max, 0.0, 1.0))
        return 1.0 if bool(cmd) else 0.0

    def _apply(self, goals):
        for name, f in self.rig_frames.items():
            self.cell.q_cmd[name] = np.asarray(goals[f["arm"]], float).reshape(7)
            value = float(goals[f["tool_device"]][0])
            if f["tool"] == "jaw":
                self.cell.tool_cmd[name] = float(np.clip(value, 0.0, 1.0)) * self._jaw_half_max
            else:
                self.cell.tool_cmd[name] = value >= 0.5

    def _advance(self):
        self.cell._advance()

    def _exhausted(self):
        return (
            self.frame_limit is not None
            and self.control is not None
            and self.control.frame >= self.frame_limit
        )

    def device_write(self, name, target):
        return self.control.write(name, target)

    def device_stop(self, name):
        if self.device_facts.get(name, {}).get("type") == "joint":
            target = self.device_read(name)["positions"]
        else:
            target = self.control.targets[name]
        return self.control.stop(name, target)

    # -- sensing -----------------------------------------------------------------------------------
    @property
    def timestamp(self):
        return self.cell.tick / self.cell.ctrl_hz

    def _arm_of(self, device):
        for name, f in self.rig_frames.items():
            if device in (f["arm"], f["tool_device"]):
                return name, f
        raise ValueError("Unknown native actuator")

    def device_read(self, name):
        arm, f = self._arm_of(name)
        st = self.cell.arm_state(arm)
        base = {"faults": [], "timestamp": self.timestamp}
        if name == f["arm"]:
            return {
                **base,
                "positions": [float(v) for v in st["qpos"]],
                "velocities": [float(v) for v in st["qvel"]],
                "efforts": [],
            }
        if f["tool"] == "jaw":
            return {**base, "aperture_m": max(0.0, float(st["jaw_opening"]))}
        return {**base, "positions": [], "velocities": [], "object_detected": bool(st["held"] > 0)}

    def device_image(self, name):
        pub = next(
            (f["camera"] for arm, f in self.rig_frames.items() if name == f"{arm}_camera"), name
        )
        if pub not in self.cell._cams:
            raise ValueError("Unknown native camera")
        frame = self.cell.frame(pub)
        rgb = np.asarray(frame["rgb"])
        if rgb.dtype != np.uint8:
            rgb = (
                (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                if rgb.max() <= 1.0
                else rgb.astype(np.uint8)
            )
        return rgb[..., :3], np.asarray(frame["depth"], float)

    # -- owner-only evidence and verdict -----------------------------------------------------------
    def owner_state(self):
        d = self.cell.data
        return {
            "tick": int(self.cell.tick),
            "qpos": d.qpos.astype(float).tolist(),
            "qvel": d.qvel.astype(float).tolist(),
            "aborted": self.cell.aborted,
        }

    def _state_hash(self):
        return hashlib.sha256(json.dumps(self.owner_state(), sort_keys=True).encode()).hexdigest()

    def goal_state(self):
        return self.cell.goal_state()

    def task_completed(self):
        return bool(self.cell is not None and self.cell.goal_state().get("ok"))

    def goal_detail(self):
        """The task goal's own result, every check of it (owner-only: the verdict's detail, never a device)."""
        if self.cell is None:
            return {}
        return json.loads(json.dumps(self.cell.goal_state(), default=float))
