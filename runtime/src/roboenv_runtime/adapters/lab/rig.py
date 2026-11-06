"""Rig facts of the lab cell, read from the compiled MuJoCo model of a task's world (no URDF).

Two UFACTORY xArm 7 arms on one desk, each with a jaw gripper or a vacuum cup and a wrist RGB-D
camera, plus the standard cell's fixed camera (right_fixed). Every task stands in the standard cell, so a rig is
one pair of grippers; tasks with the same pair share a rig and its device catalog.
"""

import numpy as np
from roboenv.devices.model import local_frame
from roboenv.tasks.lab import RIGS, rig_of  # noqa: F401  (the rigs are shared with task discovery)

from roboenv_runtime.adapters.urdf_geometry import (
    gripper_intrinsics,
    joint_intrinsics,
    matrix_pose,
    mount,
)
from roboenv_runtime.bindings.backends.robocasa.camera import intrinsics as camera_intrinsics

# MuJoCo camera axes (x right, y up, looking -z) -> OpenCV optical (x right, y down, z forward)
OPTICAL = np.diag([1.0, -1.0, -1.0, 1.0])
JAW_DESCRIPTION = (
    "UFACTORY xArm parallel jaw: two fingers on four-bar linkages close symmetrically along the "
    "base's y axis. "
    "Aperture is the measured separation of the finger pads; it does not indicate a grasp."
)


def _quat_xyzw(q):
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def _body_world(model, data, name):
    b = model.body(name).id
    T = np.eye(4)
    T[:3, :3] = data.xmat[b].reshape(3, 3)
    T[:3, 3] = data.xpos[b]
    return T


def _site_world(model, data, name):
    s = model.site(name).id
    T = np.eye(4)
    T[:3, :3] = data.site_xmat[s].reshape(3, 3)
    T[:3, 3] = data.site_xpos[s]
    return T


def arm_chain(model, prefix):
    """The arm as RoboEnv's serial chain: link_base -> link1..link7 -> flange, joint1..joint7."""
    joints, chain = [], []
    for i in range(1, 8):
        body = model.body(f"{prefix}link{i}").id
        joint = model.joint(f"{prefix}joint{i}").id
        if model.jnt_bodyid[joint] != body or model.jnt_type[joint] != 3:  # mjJNT_HINGE
            raise ValueError("lab arm joints must be hinges on link1..link7")
        axis = np.asarray(model.jnt_axis[joint], float)
        limited = bool(model.jnt_limited[joint])
        lo, hi = (float(v) for v in model.jnt_range[joint])
        chain.append(
            {
                "origin": [*map(float, model.body_pos[body]), *_quat_xyzw(model.body_quat[body])],
                "joint": i - 1,
                "axis": (axis / np.linalg.norm(axis)).tolist(),
                "anchor": [float(v) for v in model.jnt_pos[joint]],
                "reference": 0.0,
            }
        )
        joints.append(
            {
                "name": f"joint{i}",
                "type": "revolute" if limited else "continuous",
                "limits": {"min": lo, "max": hi} if limited else None,
                "max_velocity": None,
                "max_effort": None,
            }
        )
    flange = model.body(f"{prefix}flange").id
    chain.append(
        {"origin": [*map(float, model.body_pos[flange]), *_quat_xyzw(model.body_quat[flange])]}
    )
    return {"format": "serial_chain", "joints": joints, "chain": chain}


def _rotation(axis, angle):
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def chain_fk(base_pose, model, q):
    """Forward kinematics of a serial_chain (origin, then the joint's rotation about its anchor)."""
    from roboenv_runtime.adapters.urdf_geometry import matrix

    T = matrix(base_pose)
    for link in model["chain"]:
        T = T @ matrix(link["origin"])
        if "joint" in link:
            R = np.eye(4)
            R[:3, :3] = _rotation(link["axis"], q[link["joint"]] - link["reference"])
            a = np.asarray(link["anchor"], float)
            A, Ai = np.eye(4), np.eye(4)
            A[:3, 3], Ai[:3, 3] = a, -a
            T = T @ A @ R @ Ai
    return T


def _camera(cell, pub, parent, parent_T):
    w, h = cell.frame_size(pub) if hasattr(cell, "frame_size") else _wh(cell, pub)
    fovy = float(cell.model.cam_fovy[cell.model.camera(cell._cams[pub]).id])
    intr = camera_intrinsics(w, h, fovy, True)
    K = cell.camera_K(pub)
    intr["color"]["projection"].update(
        fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2])
    )
    optical = cell.camera_pose(pub) @ OPTICAL
    return {
        "type": "rgbd",
        "intrinsics": intr,
        "extrinsics": mount(
            parent,
            matrix_pose(np.linalg.inv(parent_T) @ optical),
            "flange" if parent != "world" else None,
        ),
    }


def _wh(cell, pub):
    from sim import world as W

    return W._wh(pub)


def describe(cell, rig):
    """(facts, frames) of the rig as built in `cell`; facts are what the catalog publishes."""
    from cell import spec as tspec

    m, d = cell.model, cell.data
    facts = {
        "world": {
            "type": "world",
            "intrinsics": {
                "source": "default",
                "method": "unknown",
                "calibrated_at": None,
                "bounds": None,
                "surfaces": [],
            },
            "extrinsics": {},
        }
    }
    frames = {}
    for name, arm in cell.arms.items():
        p = arm.prefix
        arm_dev = f"{name}_arm"
        model = arm_chain(m, p)
        base_T = _body_world(m, d, f"{p}link_base")
        flange_T = _body_world(m, d, f"{p}flange")
        facts[arm_dev] = {
            "type": "joint",
            "intrinsics": joint_intrinsics(model, [float(v) for v in tspec.ARM_HOME], arm_dev),
            "extrinsics": mount("world", matrix_pose(base_T)),
        }
        grip_T = _site_world(m, d, f"{p}grip")
        if arm.tool == "jaw":
            dev = f"{name}_gripper"
            gb_T = _body_world(m, d, f"{p}jaw_gripper_base")
            intr = gripper_intrinsics(dev)
            intr["aperture_m"] = {"min": 0.0, "max": float(2 * tspec.JAW_HALF_MAX)}
            intr["frames"] = {"tcp": local_frame(matrix_pose(np.linalg.inv(gb_T) @ grip_T))}
            intr["geometry"] = {
                "description": JAW_DESCRIPTION,
                "approach_axis": [0.0, 0.0, 1.0],
                "closing_axis": [0.0, 1.0, 0.0],
            }
            facts[dev] = {
                "type": "gripper",
                "intrinsics": intr,
                "extrinsics": mount(arm_dev, matrix_pose(np.linalg.inv(flange_T) @ gb_T), "flange"),
            }
        else:
            dev = f"{name}_suction"
            facts[dev] = {
                "type": "suction",
                "intrinsics": {
                    "source": "introspected",
                    "method": "introspection",
                    "calibrated_at": None,
                    "frames": {},
                    "control": {"mode": "binary", "off_target": 0, "on_target": 1},
                    "feedback": {
                        "positions": [],
                        "velocities": False,
                        "sensors": ["object_detected"],
                        "aperture_model": None,
                    },
                    "stop": {"behavior": "hold", "affected_devices": [dev]},
                },
                "extrinsics": mount(
                    arm_dev, matrix_pose(np.linalg.inv(flange_T) @ grip_T), "flange"
                ),
            }
        facts[f"{name}_camera"] = _camera(cell, f"{name}_wrist", arm_dev, flange_T)
        frames[name] = {
            "prefix": p,
            "tool": arm.tool,
            "arm": arm_dev,
            "tool_device": dev,
            "camera": f"{name}_wrist",
        }
    for pub in cell._cams:
        if not pub.endswith("_wrist"):
            facts[pub] = _camera(cell, pub, "world", np.eye(4))
    return facts, frames


def verify(cell, facts, frames, *, atol=1e-6):
    """The live world matches the rig: arm mounts, the chain's FK, camera projections."""
    from roboenv_runtime.adapters.urdf_geometry import matrix

    m, d = cell.model, cell.data
    for name, f in frames.items():
        arm = facts[f["arm"]]
        mnt = arm["extrinsics"]["mount"]
        base = [*mnt["translation_m"], *mnt["rotation_xyzw"]]
        if not np.allclose(
            matrix(base), _body_world(m, d, f"{f['prefix']}link_base"), atol=atol, rtol=0
        ):
            raise ValueError(f"{name}: native arm mount differs from the rig")
        q = cell.arm_state(name)["qpos"]
        fk = chain_fk(base, arm["intrinsics"]["model"], q)
        if not np.allclose(fk, _body_world(m, d, f"{f['prefix']}flange"), atol=1e-5, rtol=0):
            raise ValueError(f"{name}: the rig's kinematic chain differs from the native arm")
    for dev, fact in facts.items():
        if fact["type"] == "rgbd":
            pub = next(
                (
                    fr["camera"]
                    for fr in frames.values()
                    if f"{fr['arm'].rsplit('_', 1)[0]}_camera" == dev
                ),
                dev,
            )
            K = cell.camera_K(pub)
            proj = fact["intrinsics"]["color"]["projection"]
            if not np.allclose(
                [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
                [proj[k] for k in ("fx", "fy", "cx", "cy")],
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError(f"{dev}: native camera projection differs from the rig")
