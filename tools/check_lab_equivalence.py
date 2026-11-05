"""Check that the lab backend serves the same physics as lab-sim's own world.

Run with the lab interpreter, with no other simulator on the port:

    MUJOCO_GL=egl .venv-sim/lab/bin/python tools/check_lab_equivalence.py [--task open_drawer]

A: the task's world built as lab-sim's runner builds it (sim.runner.build_world), driven tick by
   tick: every 20 Hz control tick the arms' joint targets and tool commands are set, then the cell
   advances one control tick.
B: the same task and seed launched with roboenv (stepped), driven through the public device API:
   the same targets written to the arm and tool devices, then world.wait for the segment.
After every segment both arms' joints, velocities, tool readings and the clock are compared, and at
the end one wrist camera's colour image. Exit status 1 on any difference beyond --atol.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLI = str(ROOT / ".venv/bin/roboenv")
PORT = 8123
ENDPOINT = ["--endpoint", f"http://127.0.0.1:{PORT}"]
HZ = 20


def plan(arms, tools):
    """Segments of (duration_s, {device: target}): joint offsets, then the tools closing."""
    home = [0.0, -0.2, 0.0, 0.6, 0.0, 0.8, 0.0]
    segments = []
    for k, (d1, d2, d4) in enumerate(
        [(0.3, 0.1, 0.2), (0.5, 0.2, 0.4), (0.2, 0.0, 0.1), (0.0, 0.1, 0.3)]
    ):
        targets = {}
        for i, arm in enumerate(arms):
            sign = 1 if i == 0 else -1
            q = list(home)
            q[0] += sign * d1
            q[1] += d2
            q[3] += d4
            targets[f"{arm}_arm"] = q
            tool = f"{arm}_gripper" if tools[arm] == "jaw" else f"{arm}_suction"
            targets[tool] = [1.0 if k < 2 else 0.0] if tools[arm] == "jaw" else [0 if k < 2 else 1]
        segments.append((0.5 + 0.25 * k, targets))
    return segments


def run_a(task, seed, segments):
    sys.path.insert(0, str(ROOT / "third_party/lab_sim"))
    from cell import spec as tspec
    from sim import runner

    task_dir = ROOT / "third_party/lab_sim/tasks" / task
    spec = yaml.safe_load((task_dir / "task.yaml").read_text())
    spec["_dir"] = task_dir
    cell = runner.build_world(spec, seed)
    cell.budget_t = float("inf")
    tools = {name: a.tool for name, a in cell.arms.items()}
    states = []
    try:
        for duration, targets in segments:
            for _ in range(int(round(duration * HZ))):
                for arm in cell.arms:
                    cell.q_cmd[arm] = np.asarray(targets[f"{arm}_arm"], float)
                    if tools[arm] == "jaw":
                        cell.tool_cmd[arm] = (
                            float(targets[f"{arm}_gripper"][0]) * tspec.JAW_HALF_MAX
                        )
                    else:
                        cell.tool_cmd[arm] = float(targets[f"{arm}_suction"][0]) >= 0.5
                cell._advance()
            states.append(read_a(cell, tools))
        image = np.asarray(cell.frame(f"{list(cell.arms)[0]}_wrist")["rgb"])
    finally:
        cell.close()
    return list(cell.arms), tools, states, image


def read_a(cell, tools):
    out = {"time_s": cell.tick / cell.ctrl_hz}
    for arm in cell.arms:
        st = cell.arm_state(arm)
        out[f"{arm}_q"] = st["qpos"]
        out[f"{arm}_qd"] = st["qvel"]
        out[f"{arm}_tool"] = st["jaw_opening"] if tools[arm] == "jaw" else float(st["held"] > 0)
    return out


def call(*args):
    out = subprocess.run([CLI, *args, *ENDPOINT], capture_output=True, text=True, cwd=ROOT)
    if out.returncode:
        raise SystemExit(f"roboenv {' '.join(args[:3])}: {out.stdout}{out.stderr}")
    data = json.loads(out.stdout)
    return data.get("data", data) if isinstance(data, dict) else data


def run_b(task, seed, arms, tools, segments):
    import tempfile

    spec = yaml.safe_load(
        (ROOT / "catalog/tasks/lab" / task.replace("_", "-") / "task.yaml").read_text()
    )
    path = Path(tempfile.mkdtemp(prefix="lab-eq-")) / "task.yaml"
    path.write_text(yaml.safe_dump({**spec, "seed": seed}, sort_keys=False))
    launched = subprocess.run(
        [CLI, "launch", str(path), "--execution-mode", "stepped", "--port", str(PORT), "--detach"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=600,
    )
    text = launched.stdout + launched.stderr
    if "Ready:" not in text:
        raise SystemExit(f"launch failed:\n{text[-1500:]}")
    run = next(
        line.split("/runs/")[1].split("/")[0] for line in text.splitlines() if "/runs/" in line
    )
    states = []
    try:
        for duration, targets in segments:
            for device, target in targets.items():
                action = target if device.endswith("_arm") else target[0]
                call(
                    "device",
                    device,
                    "write",
                    json.dumps({"actions": [action], "duration_s": duration}),
                )
            call("device", "world", "wait", json.dumps({"duration_s": duration}))
            out = {"time_s": call("device", "world", "read", "{}")["time_s"]}
            for arm in arms:
                r = call("device", f"{arm}_arm", "read", "{}")
                out[f"{arm}_q"], out[f"{arm}_qd"] = r["positions"], r["velocities"]
                if tools[arm] == "jaw":
                    out[f"{arm}_tool"] = call("device", f"{arm}_gripper", "read", "{}")[
                        "aperture_m"
                    ]
                else:
                    out[f"{arm}_tool"] = float(
                        call("device", f"{arm}_suction", "read", "{}")["object_detected"]
                    )
            states.append(out)
        from PIL import Image

        image = np.asarray(
            Image.open(call("device", f"{arms[0]}_camera", "read", "{}")["file"]).convert("RGB")
        )
    finally:
        subprocess.run([CLI, "stop", run], capture_output=True, cwd=ROOT)
    return states, image


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", default="open_drawer")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--atol", type=float, default=1e-9)
    args = ap.parse_args()
    t0 = time.time()
    # the plan needs the rig's arms and tools: build A once to read them, then run both
    probe_arms, probe_tools, _, _ = run_a(args.task, args.seed, [])
    segments = plan(probe_arms, probe_tools)
    arms, tools, a_states, a_image = run_a(args.task, args.seed, segments)
    b_states, b_image = run_b(args.task, args.seed, arms, tools, segments)
    worst = 0.0
    for k, (sa, sb) in enumerate(zip(a_states, b_states, strict=True)):
        diffs = {
            key: float(np.max(np.abs(np.asarray(sa[key], float) - np.asarray(sb[key], float))))
            for key in sa
        }
        worst = max(worst, max(diffs.values()))
        flagged = {key: f"{v:.2e}" for key, v in diffs.items() if v > args.atol}
        print(
            f"segment {k + 1}: t={sa['time_s']:.2f}s  max diff {max(diffs.values()):.2e}"
            + (f"  DIFFER: {flagged}" if flagged else "")
        )
    # renders are deterministic; a state difference at the 1e-15 level can move an edge pixel by one
    # intensity level, so the image passes within 2 levels on at most 0.1% of the pixels
    if a_image.shape != b_image.shape:
        raise SystemExit(f"image shapes differ: {a_image.shape} vs {b_image.shape}")
    delta = np.abs(a_image.astype(int) - b_image.astype(int))
    pixels = int(np.count_nonzero(delta.max(axis=-1)))
    image_ok = int(delta.max()) <= 2 and pixels <= 0.001 * delta.shape[0] * delta.shape[1]
    print(
        f"{arms[0]} wrist image: {a_image.shape}, differing pixels {pixels}, "
        f"max level diff {int(delta.max())}"
    )
    ok = worst <= args.atol and image_ok
    print(
        f"{'EQUIVALENT' if ok else 'NOT EQUIVALENT'}: {args.task} seed {args.seed}, "
        "worst difference "
        f"{worst:.2e} (atol {args.atol:g}), {time.time() - t0:.0f}s"
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
