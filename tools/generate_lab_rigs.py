"""Deterministically generate the lab cell's device catalogs, binding defaults and task files.

Run with the lab interpreter (it builds each task's MuJoCo world to read the rig):

    MUJOCO_GL=egl .venv-sim/lab/bin/python tools/generate_lab_rigs.py \
        --bindings-out src/roboenv/devices/bindings/defaults

Every task of a rig must produce identical device facts; a difference (a scene camera moved,
another gripper) is an error that calls for a new rig in roboenv_runtime/adapters/lab/rig.py.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from generate_native_rigs import NATIVE_LIMIT_NOTES, write_rig  # noqa: E402

NATIVE_LIMIT_NOTES["lab"] = (
    "MuJoCo position actuators clip arm joint targets to their control ranges; joint ranges are "
    "the model's. Gripper profiles reject commands outside [0, 1] (1 open, 0 closed); suction is "
    "on (1) or off (0). "
    "Arm velocity and effort ratings are unspecified; no additional command speed caps apply."
)


def task_file(task, rig, sentence):
    return {
        "id": f"lab/{task.replace('_', '-')}",
        "robot": f"lab/{rig}",
        "instruction": sentence,
        "seed": 1,
        "budget_mode": "practice",
        "execution_mode": "stepped",
        "stop_on_success": False,
        "backend_kwargs": {"task": task, "rig": rig},
    }


def main():
    from roboenv_runtime.adapters.lab.devices import LabDevices
    from roboenv_runtime.adapters.lab.rig import RIGS

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=ROOT / "catalog", help="the public catalog root")
    ap.add_argument("--bindings-out", type=Path, default=None)
    args = ap.parse_args()
    for rig, tasks in RIGS.items():
        facts = None
        for task in tasks:
            world = LabDevices(task, rig)
            try:
                mine = json.loads(json.dumps(world.device_facts, sort_keys=True))
                sentence = str(world.spec.get("sentence") or "")
            finally:
                world.close()
            if facts is None:
                facts = mine
            elif mine != facts:
                differ = sorted(k for k in set(mine) | set(facts) if mine.get(k) != facts.get(k))
                raise SystemExit(f"rig {rig}: task {task} differs from {tasks[0]} in {differ}")
            path = args.out / "tasks" / "lab" / task.replace("_", "-") / "task.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                yaml.safe_dump(task_file(task, rig, sentence), sort_keys=False, width=110)
            )
        write_rig(args.out, rig, "lab", rig, facts, bindings_out=args.bindings_out)
        print(f"{rig}: {len(facts)} devices, tasks {tasks}")


if __name__ == "__main__":
    main()
