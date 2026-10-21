"""Stage 2 demo: the arm reaching a sequence of commanded Cartesian targets.

Each target is sampled the same way eval_stage2_reach_target.py validates them (a random
valid joint config, forward-kinematics'd to a guaranteed-reachable point -- not an arbitrary
Cartesian point that might be unreachable), solved via IK from the arm's current pose, then
driven there in closed loop under gravity (control.py's PD + gravity-compensation). The green
sphere is the commanded target (a mocap body, assets/wam7_bhand.xml); the red dot on the palm
is the end-effector site actually being driven to it.

Usage:
    python3 demo_stage2_reach_target.py --record media/stage2_reach_target.gif
"""
import argparse

import mujoco as mj
import numpy as np
from PIL import Image

from control import pd_gravity_comp_step
from eval_stage2_reach_target import random_target, URDF, SITE
from ik import solve_ik_multistart, N_ARM

# seed=1 used to be one of the 3 trials eval_stage2_reach_target.py's original sweep couldn't
# solve at all (a joint-limit deadlock from home -- see README's Stage 2 writeup and ik.py's
# solve_ik_multistart docstring for the fix); kept here deliberately to show it now works.
DEMO_SEEDS = [1, 2, 15, 18]
SECONDS_PER_TARGET = 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=str, default=None)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(URDF)
    data = mj.MjData(model)
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, SITE)

    renderer = mj.Renderer(model, height=480, width=640)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 1.9]
    cam.distance = 2.5
    cam.elevation = -10
    azimuth = 40.0

    frames = []

    def render():
        nonlocal azimuth
        azimuth += 0.15
        cam.azimuth = azimuth
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    q_current = np.zeros(N_ARM)
    data.qpos[:N_ARM] = q_current
    mj.mj_forward(model, data)

    restart_rng = np.random.default_rng(30_000)
    for seed in DEMO_SEEDS:
        rng = np.random.default_rng(seed)
        target_pos, _ = random_target(model, rng)
        data.mocap_pos[0] = target_pos
        q_ik, ik_ok, _, n_restarts = solve_ik_multistart(model, target_pos, q_current,
                                                          restart_rng=restart_rng)

        n_steps = int(SECONDS_PER_TARGET / model.opt.timestep)
        for step in range(n_steps):
            pd_gravity_comp_step(model, data, q_ik)
            if step % 20 == 0:
                render()

        q_current = data.qpos[:N_ARM].copy()
        final_dist = float(np.linalg.norm(data.site_xpos[site_id] - target_pos))
        print(f"seed={seed} ik_ok={ik_ok} n_restarts={n_restarts} "
              f"target={np.round(target_pos, 3)} final_dist={final_dist:.4f}m")

    if args.record:
        from pathlib import Path
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=33, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
