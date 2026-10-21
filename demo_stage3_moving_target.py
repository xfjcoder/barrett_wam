"""Stage 3 demo: the arm tracking a moving (circling) target in closed loop.

Same controller (control.py) as Stage 2, with IK (solve_ik_multistart, see ik.py) warm-started
each step from the arm's current pose and re-solved continuously against
trajectory.circle_target's moving position, instead of solving once for a static target. The
green sphere is the moving target; the red dot on the palm is the end-effector site chasing
it, with a small visible tracking lag -- see eval_stage3_moving_target.py for the measured lag.

Usage:
    python3 demo_stage3_moving_target.py --record media/stage3_moving_target.gif
"""
import argparse

import mujoco as mj
import numpy as np
from PIL import Image

from control import pd_gravity_comp_step
from eval_stage2_reach_target import random_target, URDF, SITE
from ik import solve_ik_multistart, N_ARM
from trajectory import circle_target

# seed=2 from eval_stage3_moving_target.py's own sweep: tracked (steady mean error ~4cm), with
# enough lag to actually be visible in the GIF rather than looking like a static reach.
SEED = 2
RADIUS = 0.15
OMEGA = 0.5
SECONDS = 16.0  # ~1.25 circles after the ~2s settle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=str, default=None)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(URDF)
    data = mj.MjData(model)
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, SITE)

    rng = np.random.default_rng(SEED)
    center, _ = random_target(model, rng)

    data.qpos[:N_ARM] = 0.0
    mj.mj_forward(model, data)

    renderer = mj.Renderer(model, height=480, width=640)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 1.9]
    cam.distance = 2.5
    cam.elevation = -10
    azimuth = 40.0

    frames = []
    dt = model.opt.timestep
    n_steps = int(SECONDS / dt)
    restart_rng = np.random.default_rng(20_000 + SEED)
    errs = []
    for i in range(n_steps):
        t = i * dt
        target = circle_target(t, center, RADIUS, OMEGA)
        data.mocap_pos[0] = target
        q_ik, _, _, _ = solve_ik_multistart(model, target, data.qpos[:N_ARM], max_iters=20,
                                             tol=0.003, n_restarts=5, restart_rng=restart_rng)
        pd_gravity_comp_step(model, data, q_ik)
        errs.append(float(np.linalg.norm(data.site_xpos[site_id] - target)))
        if i % 20 == 0:
            azimuth += 0.3
            cam.azimuth = azimuth
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    errs = np.array(errs)
    steady = errs[int(2.0 / dt):]
    print(f"center={np.round(center, 3)} steady_mean_err={steady.mean():.4f}m "
          f"steady_max_err={steady.max():.4f}m")

    if args.record:
        from pathlib import Path
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=33, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
