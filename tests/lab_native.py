"""Native acceptance checks of the lab backend (run with the lab interpreter, port 8123 free).

    MUJOCO_GL=egl .venv-sim/lab/bin/python tests/lab_native.py

Static: the catalog regenerates byte-identically; suction, gripper and camera profiles follow the
device contracts. Kinematics: each arm profile's serial chain matches MuJoCo's flange pose at random
joints. Live (roboenv launch, public device API): invalid targets are rejected before any effect;
reads never advance time, world.wait does; stop holds; camera files match the profile (PNG and
16-bit millimetre PGM depth); public reads carry no private scene state; an exhausted budget ends
the episode with episode_finished. Prints PASS/FAIL per check; exit status 1 on any failure.
"""

import filecmp
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLI = str(ROOT / ".venv/bin/roboenv")
ENDPOINT = ["--endpoint", "http://127.0.0.1:8123"]
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def call(*args, ok=True):
    out = subprocess.run([CLI, *args, *ENDPOINT], capture_output=True, text=True, cwd=ROOT)
    data = json.loads(out.stdout) if out.stdout.strip().startswith("{") else {"raw": out.stdout}
    if ok and out.returncode:
        raise RuntimeError(f"roboenv {' '.join(args[:3])}: {out.stdout}{out.stderr}")
    return (data.get("data", data) if isinstance(data, dict) else data), out.returncode


def static_checks():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                sys.executable,
                "tools/generate_lab_rigs.py",
                "--out",
                tmp,
                "--bindings-out",
                f"{tmp}/bindings",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        cmp = filecmp.dircmp(f"{tmp}/robots/lab", ROOT / "catalog/robots/lab")

        def differ(c):
            return (
                c.diff_files
                + c.left_only
                + c.right_only
                + [f for sub in c.subdirs.values() for f in differ(sub)]
            )

        check("catalog regenerates byte-identically", not differ(cmp), str(differ(cmp)[:5]))
    for path in sorted((ROOT / "catalog/robots/lab").glob("*/devices/*/profile.yaml")):
        p = yaml.safe_load(path.read_text())
        rel = "/".join(path.parts[-4:-1])
        if p["type"] == "suction":
            control = p["intrinsics"]["control"]
            check(
                f"{rel}: suction control is binary 0/1, strict keys",
                control == {"mode": "binary", "off_target": 0, "on_target": 1},
            )
        if p["type"] == "gripper":
            control = p["intrinsics"]["control"]
            check(
                f"{rel}: gripper normalized [0, 1], open 1",
                control["bounds"] == {"min": 0, "max": 1}
                and control["open_target"] == 1
                and control["close_target"] == 0,
            )
        if p["type"] == "rgbd":
            depth = p["intrinsics"]["depth"]
            check(
                f"{rel}: depth pgm16 millimetres, 0 invalid",
                depth["image_format"] == "pgm16_be"
                and depth["scale_m_per_unit"] == 0.001
                and depth["invalid_value"] == 0,
            )
        if p["type"] != "world":
            check(
                f"{rel}: has a driver",
                (path.parent / "driver.py").is_file() and "driver" in p.get("access", {}),
            )


def kinematics_checks():
    from roboenv_runtime.adapters.lab.devices import LabDevices
    from roboenv_runtime.adapters.lab.rig import RIGS, _body_world, chain_fk

    rng = np.random.default_rng(0)
    for rig, tasks in RIGS.items():
        world = LabDevices(tasks[0], rig)
        try:
            import mujoco

            cell, worst = world.cell, 0.0
            for f in world.rig_frames.values():
                fact = world.device_facts[f["arm"]]
                mnt = fact["extrinsics"]["mount"]
                base = [*mnt["translation_m"], *mnt["rotation_xyzw"]]
                for _ in range(5):
                    q = [
                        rng.uniform(j["limits"]["min"], j["limits"]["max"]) * 0.8
                        for j in fact["intrinsics"]["model"]["joints"]
                    ]
                    m, d = cell.model, cell.data
                    for i, v in enumerate(q):
                        d.qpos[m.jnt_qposadr[m.joint(f"{f['prefix']}joint{i + 1}").id]] = v
                    mujoco.mj_kinematics(m, d)
                    fk = chain_fk(base, fact["intrinsics"]["model"], q)
                    worst = max(
                        worst, float(np.abs(fk - _body_world(m, d, f"{f['prefix']}flange")).max())
                    )
            check(
                f"{rig}: profile FK equals MuJoCo flange at random joints",
                worst < 1e-9,
                f"{worst:.1e}",
            )
        finally:
            world.close()


def live_checks():
    spec = yaml.safe_load((ROOT / "catalog/tasks/lab/place-in-drawer/task.yaml").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "task.yaml"  # jaw left, suction right; a 3 s budget for the end
        spec["backend_kwargs"] = {**spec["backend_kwargs"], "max_steps": 60}
        path.write_text(yaml.safe_dump(spec, sort_keys=False))
        out = subprocess.run(
            [CLI, "launch", str(path), "--execution-mode", "stepped", "--port", "8123", "--detach"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=600,
        )
    text = out.stdout + out.stderr
    run = next(
        (line.split("/runs/")[1].split("/")[0] for line in text.splitlines() if "/runs/" in line),
        None,
    )
    check("launch place_in_drawer reaches Ready", "Ready:" in text)
    if "Ready:" not in text:
        return
    try:
        q0, _ = call("device", "left_arm", "read", "{}")
        t0 = call("device", "world", "read", "{}")[0]["time_s"]
        for label, device, args in [
            ("arm target of wrong length", "left_arm", {"actions": [[0.0] * 6], "duration_s": 0.5}),
            (
                "arm target with NaN",
                "left_arm",
                {"actions": [[float("nan")] * 7], "duration_s": 0.5},
            ),
            ("gripper beyond [0, 1]", "left_gripper", {"actions": [1.5], "duration_s": 0.5}),
            ("suction neither 0 nor 1", "right_suction", {"actions": [0.5], "duration_s": 0.5}),
        ]:
            payload = json.dumps(args).replace("NaN", "1e999")
            _, code = call("device", device, "write", payload, ok=False)
            check(f"rejected: {label}", code != 0)
        q1, _ = call("device", "left_arm", "read", "{}")
        t1 = call("device", "world", "read", "{}")[0]["time_s"]
        check(
            "rejections had no effect (joints, clock)",
            q1["positions"] == q0["positions"] and t1 == t0,
        )
        check("reads never advance the clock", t1 == t0 == 0.0, f"t={t1}")
        target = list(q0["positions"])
        target[0] += 0.4
        call("device", "left_arm", "write", json.dumps({"actions": [target], "duration_s": 2.0}))
        call("device", "world", "wait", json.dumps({"duration_s": 0.25}))
        call("device", "left_arm", "stop", "{}")
        held = call("device", "left_arm", "read", "{}")[0]["positions"][0]
        call("device", "world", "wait", json.dumps({"duration_s": 1.0}))
        after = call("device", "left_arm", "read", "{}")[0]
        check(
            "world.wait advances the clock",
            abs(after["timestamp"] - 1.25) < 1e-9,
            f"t={after['timestamp']}",
        )
        check(
            "stop holds the arm (no further travel to the old target)",
            abs(after["positions"][0] - held) < 0.002 and after["positions"][0] < target[0] - 0.01,
            f"held {held:.3f}, after {after['positions'][0]:.3f}, old target {target[0]:.3f}",
        )
        cam, _ = call("device", "left_camera", "read", "{}")
        profile = yaml.safe_load(
            (
                ROOT / "catalog/robots/lab/lab-xarm7-tray/devices/left_camera/profile.yaml"
            ).read_text()
        )["intrinsics"]
        from PIL import Image

        png = Image.open(cam["file"])
        raw = Path(cam["depth_file"]).read_bytes()
        header = raw.split(b"\n", 3)
        w, h = (int(v) for v in header[1].split())
        depth = np.frombuffer(header[3], dtype=">u2").reshape(h, w)
        check(
            "camera colour size matches the profile",
            png.size == (profile["color"]["width"], profile["color"]["height"]),
        )
        check(
            "depth is 16-bit PGM, millimetres within the scene",
            header[0] == b"P5"
            and header[2] == b"65535"
            and (w, h) == png.size
            and 0.05 < np.median(depth[depth > 0]) / 1000 < 2.0,
            f"median {np.median(depth[depth > 0]):.0f} mm",
        )
        public = json.dumps(
            [
                call("device", d, "read", "{}")[0]
                for d in ("left_arm", "left_gripper", "right_suction", "world")
            ]
            + [call("describe")[0]]
        )
        leaks = [
            w
            for w in ("qpos", "puck", "part_pose", "goal", "verdict", "drawer_drawer")
            if w in public
        ]
        check("public reads and describe carry no private scene state", not leaks, str(leaks))
        call("device", "world", "wait", json.dumps({"duration_s": 5.0}), ok=False)
        _, code = call(
            "device",
            "left_arm",
            "write",
            json.dumps({"actions": [target], "duration_s": 0.5}),
            ok=False,
        )
        status, _ = call("device", "world", "read", "{}", ok=False)
        check(
            "an exhausted budget ends the episode (writes refused)",
            code != 0,
            f"budget {status.get('budget') if isinstance(status, dict) else status}",
        )
    finally:
        subprocess.run([CLI, "stop", run], capture_output=True, cwd=ROOT)


def main():
    static_checks()
    kinematics_checks()
    live_checks()
    failed = RESULTS.count(False)
    print(f"{len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
