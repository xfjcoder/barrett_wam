"""Stage 1 verification: does the vendored Barrett WAM 7-DOF arm + BHand URDF
(assets/wam7_bhand.urdf) actually load into MuJoCo and move the way a real WAM should?

Pure kinematics, no physics stepping and no actuators yet (the raw URDF import has none --
those come with Stage 2's controller) -- this just sets qpos directly and calls mj_forward,
sweeping each arm joint through its own range one at a time (so a bug in one joint's axis/
limits is visible in isolation) then closing/opening the BHand's three fingers together.

Usage:
    python3 demo_stage1_kinematic_sweep.py --record media/stage1_kinematic_sweep.gif
"""
import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import mujoco as mj
import numpy as np
from PIL import Image

ARM_JOINTS = [
    "wam/base_yaw_joint",
    "wam/shoulder_pitch_joint",
    "wam/shoulder_yaw_joint",
    "wam/elbow_pitch_joint",
    "wam/wrist_yaw_joint",
    "wam/wrist_pitch_joint",
    "wam/palm_yaw_joint",
]
HAND_JOINTS = [
    "wam/bhand/finger_1/prox_joint", "wam/bhand/finger_1/med_joint", "wam/bhand/finger_1/dist_joint",
    "wam/bhand/finger_2/prox_joint", "wam/bhand/finger_2/med_joint", "wam/bhand/finger_2/dist_joint",
    "wam/bhand/finger_3/med_joint", "wam/bhand/finger_3/dist_joint",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=str, default="assets/wam7_bhand.urdf")
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--frames-per-joint", type=int, default=15)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(args.urdf)
    data = mj.MjData(model)
    qadr = {name: model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)]
            for name in ARM_JOINTS + HAND_JOINTS}
    jid = {name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS + HAND_JOINTS}

    renderer = mj.Renderer(model, height=480, width=640)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 1.7]
    cam.distance = 2.2
    cam.elevation = -15

    frames = []

    def render(azimuth):
        cam.azimuth = azimuth
        mj.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    azimuth = 50.0
    # Sweep each arm joint, one at a time, through its own [low, high] range and back to 0.
    for name in ARM_JOINTS:
        lo, hi = model.jnt_range[jid[name]]
        sweep = np.concatenate([
            np.linspace(0.0, hi, args.frames_per_joint),
            np.linspace(hi, lo, 2 * args.frames_per_joint),
            np.linspace(lo, 0.0, args.frames_per_joint),
        ])
        for val in sweep:
            data.qpos[qadr[name]] = val
            azimuth += 0.3
            render(azimuth)
        data.qpos[qadr[name]] = 0.0

    # Close then open all three BHand fingers together (prox+med+dist in lockstep).
    for target_frac in (1.0, 0.0):
        for t in np.linspace(0.0, 1.0, args.frames_per_joint * 2):
            frac = t if target_frac == 1.0 else (1.0 - t)
            for name in HAND_JOINTS:
                lo, hi = model.jnt_range[jid[name]]
                data.qpos[qadr[name]] = lo + frac * (hi - lo)
            azimuth += 0.3
            render(azimuth)

    print(f"Rendered {len(frames)} frames. Final qpos (should be all back near 0/open): "
          f"{np.round(data.qpos, 3)}")

    if args.record:
        from pathlib import Path
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=33, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
