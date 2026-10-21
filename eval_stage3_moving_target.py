"""Stage 3 eval: can the arm track a moving target, not just reach a static one?

Target is a horizontal circle (trajectory.circle_target) around a random reachable center
(sampled the same way Stage 2's eval does: a random valid joint config, forward-kinematics'd
to a point). Tracking uses the same IK (ik.py) and controller (control.py) as Stage 2, but IK
is now warm-started from the arm's current pose each step with few iterations and a loose
tolerance -- a continuous resolved-rate tracking loop, not a one-shot solve.

The first 2s (arm moving from home to the circle) are excluded from the error stats as a
settle transient, matching Stage 2's own distinction between IK convergence and tracking.

Usage:
    python3 eval_stage3_moving_target.py --trials 10 --seconds 10.0
"""
import argparse

import mujoco as mj
import numpy as np

from control import pd_gravity_comp_step
from eval_stage2_reach_target import random_target, URDF, SITE
from ik import solve_ik, N_ARM
from trajectory import circle_target

RADIUS = 0.15
OMEGA = 0.5
SETTLE_SECONDS = 2.0
TRACK_TOL = 0.05  # 5cm mean steady-state error


def run_trial(model, center, seconds):
    data = mj.MjData(model)
    data.qpos[:N_ARM] = 0.0
    mj.mj_forward(model, data)
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, SITE)

    dt = model.opt.timestep
    n_steps = int(seconds / dt)
    n_settle = int(SETTLE_SECONDS / dt)
    errs = []
    for i in range(n_steps):
        t = i * dt
        target = circle_target(t, center, RADIUS, OMEGA)
        q_ik, _, _ = solve_ik(model, target, data.qpos[:N_ARM], max_iters=20, tol=0.003)
        pd_gravity_comp_step(model, data, q_ik)
        errs.append(float(np.linalg.norm(data.site_xpos[site_id] - target)))

    steady = np.array(errs[n_settle:])
    return dict(mean_err=steady.mean(), max_err=steady.max(), tracked=steady.mean() < TRACK_TOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--base-seed", type=int, default=0)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(URDF)

    results = []
    for i in range(args.trials):
        rng = np.random.default_rng(args.base_seed + i)
        center, _ = random_target(model, rng)
        r = run_trial(model, center, args.seconds)
        results.append(r)
        print(f"trial {i:2d} seed={args.base_seed + i:4d} center={np.round(center, 3)} "
              f"steady_mean_err={r['mean_err']:.4f}m steady_max_err={r['max_err']:.4f}m "
              f"tracked={r['tracked']}")

    n_tracked = sum(r["tracked"] for r in results)
    means = [r["mean_err"] for r in results]
    print(f"\nSummary over {args.trials} trials: tracked {n_tracked}/{args.trials} "
          f"(mean steady-state error < {TRACK_TOL}m), "
          f"mean of means={np.mean(means):.4f}m max of means={np.max(means):.4f}m")


if __name__ == "__main__":
    main()
