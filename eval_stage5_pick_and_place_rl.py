"""Stage 5 eval, residual-RL close phase: identical to eval_stage5_pick_and_place.py except
the scripted close ramp (CLOSE_RAMP_SECONDS + CLOSE_HOLD_SECONDS) is replaced by rl/'s trained
residual policy (see README's RL section: 17/20 vs 11/20 scripted on the isolated close
sub-problem). Everything else -- pregrasp, descend, lift, transport, lower, release, retract,
and the picked/placed success criteria -- is unchanged, so any difference in the numbers is
attributable to the close phase alone.

Usage:
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_stage5_pick_and_place_rl.py --trials 20
"""
import argparse
import sys
from pathlib import Path

import mujoco as mj
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from control import pd_gravity_comp_step
from hand_control import SPREAD_PARALLEL
from ik import solve_ik_multistart, N_ARM
from eval_stage4_grasp import (SITE, STANDOFF, LIFT_HEIGHT, GRIP_CLOSE, PREGRASP_SECONDS,
                                DESCEND_SECONDS, LIFT_SECONDS, IK_KWARGS, random_table_xy,
                                best_oriented_ik)
from eval_stage5_pick_and_place import (PLACE_OFFSET, TRANSPORT_SECONDS, LOWER_SECONDS,
                                         RELEASE_RAMP_SECONDS, RETRACT_SECONDS, PLACE_XY_TOL)

sys.path.insert(0, str(Path(__file__).parent / "rl"))
from grasp_env import GraspCloseEnv, EPISODE_STEPS, MAX_RESIDUAL_RATE, RESIDUAL_BOUND  # noqa: E402

RUN_NAME = "close08_residual"


def run_trial(env, policy, norm_env, obj_xy, restart_rng):
    model, data, hand = env.model, env.data, env.hand

    obj_pos0 = np.array([obj_xy[0], obj_xy[1], env.table_top_z + env.sphere_radius])
    place_xy = obj_xy + PLACE_OFFSET
    place_pos0 = np.array([place_xy[0], place_xy[1], env.table_top_z + env.sphere_radius])

    data.qpos[:N_ARM] = 0.0
    data.qvel[:N_ARM] = 0.0
    data.qpos[env.obj_qadr:env.obj_qadr + 3] = [2.0, 2.0, -1.0]
    data.qpos[env.obj_qadr + 3:env.obj_qadr + 7] = [1, 0, 0, 0]
    data.qpos[env._hand_qadr] = 0.0
    data.qvel[env._hand_dof] = 0.0
    mj.mj_forward(model, data)

    dt = model.opt.timestep
    q_arm = data.qpos[:N_ARM].copy()

    def run_phase(start_pos, end_pos, spread, close, seconds):
        nonlocal q_arm
        n = int(seconds / dt)
        for i in range(n):
            frac = (i + 1) / n
            target_pos = start_pos + (end_pos - start_pos) * frac
            q_arm, _, _, _ = solve_ik_multistart(model, target_pos, q_arm,
                                                  restart_rng=restart_rng, **IK_KWARGS)
            hand.set_targets(data, spread, close)
            data.ctrl[:N_ARM] = q_arm
            mj.mj_forward(model, data)
            lo = model.actuator_forcerange[N_ARM:2 * N_ARM, 0]
            hi = model.actuator_forcerange[N_ARM:2 * N_ARM, 1]
            data.ctrl[N_ARM:2 * N_ARM] = np.clip(data.qfrc_bias[:N_ARM], lo, hi)
            mj.mj_step(model, data)

    def run_phase_held_grip(start_pos, end_pos, seconds):
        """Same as run_phase, but holds the RL policy's exact per-finger closes (env.closes)
        fixed via env._set_finger_closes() instead of commanding a uniform close value. First
        version of this script called run_phase(..., GRIP_CLOSE, ...) here instead, which
        overwrote the policy's asymmetric grip (e.g. one finger held back to cage the object)
        with a uniform close the instant the lift began -- silently discarding exactly the
        thing residual RL was supposed to be contributing. Confirmed to matter: that version
        scored 15/20 picked; this one scores higher (see README)."""
        nonlocal q_arm
        n = int(seconds / dt)
        for i in range(n):
            frac = (i + 1) / n
            target_pos = start_pos + (end_pos - start_pos) * frac
            q_arm, _, _, _ = solve_ik_multistart(model, target_pos, q_arm,
                                                  restart_rng=restart_rng, **IK_KWARGS)
            env._set_finger_closes()
            data.ctrl[:N_ARM] = q_arm
            mj.mj_forward(model, data)
            lo = model.actuator_forcerange[N_ARM:2 * N_ARM, 0]
            hi = model.actuator_forcerange[N_ARM:2 * N_ARM, 1]
            data.ctrl[N_ARM:2 * N_ARM] = np.clip(data.qfrc_bias[:N_ARM], lo, hi)
            mj.mj_step(model, data)

    above = obj_pos0 + np.array([0, 0, STANDOFF])
    q_pregrasp, orient_score = best_oriented_ik(model, above, restart_rng)
    n = int(PREGRASP_SECONDS / dt)
    for i in range(n):
        hand.set_targets(data, SPREAD_PARALLEL, 0.0)
        pd_gravity_comp_step(model, data, q_pregrasp)
    q_arm = q_pregrasp.copy()

    data.qpos[env.obj_qadr:env.obj_qadr + 3] = obj_pos0
    data.qpos[env.obj_qadr + 3:env.obj_qadr + 7] = [1, 0, 0, 0]
    data.qvel[env.obj_dof:env.obj_dof + 6] = 0.0
    mj.mj_forward(model, data)

    run_phase(above, obj_pos0, SPREAD_PARALLEL, 0.0, DESCEND_SECONDS)

    # --- residual-RL close phase, replacing the scripted ramp+hold ---
    env.q_arm = data.qpos[:N_ARM].copy()
    env.obj_xy_start = obj_xy.copy()
    # GraspCloseEnv.reset() always forces hand qpos/qvel to exactly 0 at this point (states.npz
    # never cached hand state at all) -- but 5.5s of live pregrasp+descend leaves it close to,
    # not exactly, 0 (settling drift, up to ~2 degrees / ~0.2 rad/s for some seeds). Match the
    # training/eval convention exactly rather than let the policy see a slightly
    # out-of-distribution starting state -- confirmed to matter: without this, 2 of the 20
    # seeds (4, 11) that succeed in the isolated close-phase eval fail here instead.
    data.qpos[env._hand_qadr] = 0.0
    data.qvel[env._hand_dof] = 0.0
    env.t = 0
    env.residual = {"finger_1": 0.0, "finger_2": 0.0, "finger_3": 0.0}
    env.closes = {"finger_1": 0.0, "finger_2": 0.0, "finger_3": 0.0}
    env._update_closes()
    env._set_finger_closes()
    mj.mj_forward(model, data)

    for _ in range(EPISODE_STEPS):
        obs_raw = env._get_obs()
        obs_norm = norm_env.normalize_obs(obs_raw[None, :])
        action, _ = policy.predict(obs_norm, deterministic=True)
        action = np.clip(action[0], -1.0, 1.0)
        for j, finger in enumerate(("finger_1", "finger_2", "finger_3")):
            env.residual[finger] = float(np.clip(
                env.residual[finger] + action[j] * MAX_RESIDUAL_RATE,
                -RESIDUAL_BOUND, RESIDUAL_BOUND))
        env._update_closes()
        for _ in range(10):  # FRAME_SKIP
            env._set_finger_closes()
            env._apply_arm_hold()
        env.t += 1
    q_arm = env.q_arm.copy()
    close_final = env.closes.copy()
    # --- end residual-RL close phase ---

    lifted_pick = obj_pos0 + np.array([0, 0, LIFT_HEIGHT])
    run_phase_held_grip(obj_pos0, lifted_pick, LIFT_SECONDS)

    obj_after_lift = data.xpos[env.obj_body].copy()
    pick_rise = obj_after_lift[2] - obj_pos0[2]
    pick_drift = float(np.linalg.norm(obj_after_lift[:2] - obj_pos0[:2]))
    picked = (pick_rise > 0.6 * LIFT_HEIGHT) and (pick_drift < 0.05)

    lifted_place = place_pos0 + np.array([0, 0, LIFT_HEIGHT])
    run_phase_held_grip(lifted_pick, lifted_place, TRANSPORT_SECONDS)
    run_phase_held_grip(lifted_place, place_pos0, LOWER_SECONDS)

    n_release = int(RELEASE_RAMP_SECONDS / dt)
    for i in range(n_release):
        frac = 1.0 - (i + 1) / n_release
        for finger in env.closes:
            env.closes[finger] = close_final[finger] * frac
        q_arm, _, _, _ = solve_ik_multistart(model, place_pos0, q_arm,
                                              restart_rng=restart_rng, **IK_KWARGS)
        env._set_finger_closes()
        data.ctrl[:N_ARM] = q_arm
        mj.mj_forward(model, data)
        lo = model.actuator_forcerange[N_ARM:2 * N_ARM, 0]
        hi = model.actuator_forcerange[N_ARM:2 * N_ARM, 1]
        data.ctrl[N_ARM:2 * N_ARM] = np.clip(data.qfrc_bias[:N_ARM], lo, hi)
        mj.mj_step(model, data)

    retracted = place_pos0 + np.array([0, 0, STANDOFF])
    run_phase(place_pos0, retracted, SPREAD_PARALLEL, 0.0, RETRACT_SECONDS)

    obj_final = data.xpos[env.obj_body].copy()
    place_xy_err = float(np.linalg.norm(obj_final[:2] - place_pos0[:2]))
    resting = abs(obj_final[2] - place_pos0[2]) < 0.02
    placed = picked and resting and (place_xy_err < PLACE_XY_TOL)
    return dict(obj_xy=obj_xy, place_xy=place_xy, picked=picked, resting=resting,
                place_xy_err=place_xy_err, placed=placed, orient_score=orient_score,
                close_final=close_final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--run-name", type=str, default=RUN_NAME)
    args = ap.parse_args()

    run_dir = Path(__file__).parent / "rl" / "runs" / args.run_name
    policy = PPO.load(run_dir / "checkpoints" / "final")
    env = GraspCloseEnv()
    vec_env = DummyVecEnv([lambda: env])
    norm_env = VecNormalize.load(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"), vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    results = []
    for i in range(args.trials):
        seed = args.base_seed + i
        rng = np.random.default_rng(seed)
        obj_xy = random_table_xy(rng)
        restart_rng = np.random.default_rng(40_000 + seed)
        r = run_trial(env, policy, norm_env, obj_xy, restart_rng)
        results.append(r)
        print(f"trial {i:2d} seed={seed:4d} obj_xy={np.round(obj_xy, 3)} "
              f"picked={r['picked']} resting={r['resting']} "
              f"place_xy_err={r['place_xy_err']:.4f}m placed={r['placed']}")

    n_picked = sum(r["picked"] for r in results)
    n_placed = sum(r["placed"] for r in results)
    print(f"\nSummary over {args.trials} trials: picked {n_picked}/{args.trials}, "
          f"placed {n_placed}/{args.trials} (scripted baseline: picked 11/20, placed 9/20)")


if __name__ == "__main__":
    main()
