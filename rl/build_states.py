"""Precomputes a pool of "ready-to-close" states: the arm already parked at the object via
Stage 4's scripted pregrasp+descend (eval_stage4_grasp.py, unchanged), fingers still open.

Why precompute instead of re-running pregrasp+descend on every RL episode reset: pregrasp
alone is 4 simulated seconds (2000 steps at dt=0.002) and descend another 1.5s (750 steps),
all before the part an RL episode actually needs to learn from (closing the hand, ~1-2s).
Re-simulating that every reset would make the approach-phase overhead 3-4x the learned-phase
cost. Since pregrasp+descend is deterministic given a seed, it only needs running once per
distinct table position; this script runs it N_STATES times and saves the resulting (arm
qpos/qvel, object qpos/qvel, original object xy) snapshots to states.npz for grasp_env.py to
sample from on every reset.

Usage (from the repo root):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 rl/build_states.py --n-states 128
"""
import argparse
import sys
from pathlib import Path

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from control import pd_gravity_comp_step
from eval_stage4_grasp import (URDF, STANDOFF, PREGRASP_SECONDS, DESCEND_SECONDS,
                                random_table_xy, best_oriented_ik)
from hand_control import HandController, SPREAD_PARALLEL
from ik import solve_ik_multistart, N_ARM

OUT = Path(__file__).parent / "states.npz"


def build_one_state(model, seed):
    data = mj.MjData(model)
    hand = HandController(model)
    obj_joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "grasp_object_joint")
    obj_qadr = model.jnt_qposadr[obj_joint]
    obj_dof = model.jnt_dofadr[obj_joint]
    obj_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "grasp_object_geom")
    table_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "table")
    sphere_radius = model.geom_size[obj_geom][0]
    table_top_z = model.geom_pos[table_geom][2] + model.geom_size[table_geom][2]

    rng = np.random.default_rng(seed)
    obj_xy = random_table_xy(rng)
    obj_pos0 = np.array([obj_xy[0], obj_xy[1], table_top_z + sphere_radius])
    data.qpos[:N_ARM] = 0.0
    data.qpos[obj_qadr:obj_qadr + 3] = [2.0, 2.0, -1.0]
    data.qpos[obj_qadr + 3:obj_qadr + 7] = [1, 0, 0, 0]
    mj.mj_forward(model, data)

    dt = model.opt.timestep
    restart_rng = np.random.default_rng(40_000 + seed)

    above = obj_pos0 + np.array([0, 0, STANDOFF])
    q_pregrasp, orient_score = best_oriented_ik(model, above, restart_rng)
    n = int(PREGRASP_SECONDS / dt)
    for _ in range(n):
        hand.set_targets(data, SPREAD_PARALLEL, 0.0)
        pd_gravity_comp_step(model, data, q_pregrasp)
    q_arm = q_pregrasp.copy()

    data.qpos[obj_qadr:obj_qadr + 3] = obj_pos0
    data.qpos[obj_qadr + 3:obj_qadr + 7] = [1, 0, 0, 0]
    data.qvel[obj_dof:obj_dof + 6] = 0.0
    mj.mj_forward(model, data)

    n = int(DESCEND_SECONDS / dt)
    for i in range(n):
        frac = (i + 1) / n
        target = above + (obj_pos0 - above) * frac
        q_arm, _, _, _ = solve_ik_multistart(model, target, q_arm, restart_rng=restart_rng,
                                              max_iters=30, tol=0.01)
        hand.set_targets(data, SPREAD_PARALLEL, 0.0)
        data.ctrl[:N_ARM] = q_arm
        mj.mj_forward(model, data)
        lo = model.actuator_forcerange[N_ARM:2 * N_ARM, 0]
        hi = model.actuator_forcerange[N_ARM:2 * N_ARM, 1]
        data.ctrl[N_ARM:2 * N_ARM] = np.clip(data.qfrc_bias[:N_ARM], lo, hi)
        mj.mj_step(model, data)

    return dict(
        arm_qpos=data.qpos[:N_ARM].copy(),
        arm_qvel=data.qvel[:N_ARM].copy(),
        obj_qpos=data.qpos[obj_qadr:obj_qadr + 7].copy(),
        obj_qvel=data.qvel[obj_dof:obj_dof + 6].copy(),
        obj_xy=obj_xy.copy(),
        orient_score=orient_score,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=128)
    ap.add_argument("--base-seed", type=int, default=0)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(URDF)
    fields = ["arm_qpos", "arm_qvel", "obj_qpos", "obj_qvel", "obj_xy", "orient_score"]
    out = {f: [] for f in fields}
    for i in range(args.n_states):
        seed = args.base_seed + i
        state = build_one_state(model, seed)
        for f in fields:
            out[f].append(state[f])
        print(f"{i + 1}/{args.n_states} seed={seed} orient_score={state['orient_score']:.3f}")

    np.savez(OUT, **{f: np.array(out[f]) for f in fields})
    print(f"wrote {OUT} ({args.n_states} states)")


if __name__ == "__main__":
    main()
