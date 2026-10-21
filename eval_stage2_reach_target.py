"""Stage 2 eval: can the arm actually reach a commanded target, not just solve IK for one?

For each trial: sample a random valid joint config, forward-kinematics it to get a
guaranteed-reachable Cartesian target, solve IK from the arm's current pose to get a joint
target, then run closed-loop position-actuator control under gravity for a fixed duration and
check the end-effector site's final distance to the target. Judged over multiple random seeds
(not a single run), per this project's inherited eval convention.

Usage:
    python3 eval_stage2_reach_target.py --trials 12 --seconds 3.0
"""
import argparse

import mujoco as mj
import numpy as np

from control import pd_gravity_comp_step
from ik import solve_ik, N_ARM

URDF = "assets/wam7_bhand.xml"
SITE = "wam/ee_site"
TOL = 0.02  # 2cm


def random_target(model, rng):
    lo = model.jnt_range[:N_ARM, 0]
    hi = model.jnt_range[:N_ARM, 1]
    # Shrink 15% in from each joint's limits -- targets drawn from the extreme edge of the
    # workspace are disproportionately likely to be right at a singularity.
    margin = 0.15 * (hi - lo)
    q_rand = rng.uniform(lo + margin, hi - margin)
    data = mj.MjData(model)
    data.qpos[:N_ARM] = q_rand
    mj.mj_forward(model, data)
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, SITE)
    return data.site_xpos[site_id].copy(), q_rand


def run_trial(model, target_pos, q_start, seconds):
    q_ik, ik_ok, ik_iters = solve_ik(model, target_pos, q_start)

    data = mj.MjData(model)
    data.qpos[:N_ARM] = q_start
    mj.mj_forward(model, data)

    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, SITE)
    n_steps = int(seconds / model.opt.timestep)
    for _ in range(n_steps):
        pd_gravity_comp_step(model, data, q_ik)

    final_dist = float(np.linalg.norm(data.site_xpos[site_id] - target_pos))
    return dict(ik_ok=ik_ok, ik_iters=ik_iters, final_dist=final_dist,
                reached=final_dist < TOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--base-seed", type=int, default=0)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(URDF)

    results = []
    for i in range(args.trials):
        rng = np.random.default_rng(args.base_seed + i)
        target_pos, _ = random_target(model, rng)
        q_start = np.zeros(N_ARM)  # every trial starts from the same home pose
        r = run_trial(model, target_pos, q_start, args.seconds)
        results.append(r)
        print(f"trial {i:2d} seed={args.base_seed + i:4d} target={np.round(target_pos, 3)} "
              f"ik_ok={r['ik_ok']} ik_iters={r['ik_iters']:3d} "
              f"final_dist={r['final_dist']:.4f}m reached={r['reached']}")

    n_reached = sum(r["reached"] for r in results)
    n_ik_ok = sum(r["ik_ok"] for r in results)
    dists = [r["final_dist"] for r in results]
    print(f"\nSummary over {args.trials} trials: "
          f"reached {n_reached}/{args.trials}, IK converged {n_ik_ok}/{args.trials}, "
          f"final_dist mean={np.mean(dists):.4f}m max={np.max(dists):.4f}m")


if __name__ == "__main__":
    main()
