"""Stage 5 demo, residual-RL close phase: same nine phases as demo_stage5_pick_and_place.py,
but the scripted close ramp is replaced by rl/'s trained residual policy (see
eval_stage5_pick_and_place_rl.py and README's RL section: picked 17/20, placed 14/20, vs the
scripted baseline's 11/20 picked, 9/20 placed). seed=1 is one of the eval sweep's placed
successes.

Usage:
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 demo_stage5_pick_and_place_rl.py \
        --record media/stage5_pick_and_place_rl.gif
"""
import argparse
import sys
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from control import pd_gravity_comp_step
from eval_stage4_grasp import (SITE, STANDOFF, LIFT_HEIGHT, PREGRASP_SECONDS, DESCEND_SECONDS,
                                LIFT_SECONDS, IK_KWARGS, random_table_xy, best_oriented_ik)
from eval_stage5_pick_and_place import (PLACE_OFFSET, TRANSPORT_SECONDS, LOWER_SECONDS,
                                         RELEASE_RAMP_SECONDS, RETRACT_SECONDS)
from hand_control import SPREAD_PARALLEL
from ik import solve_ik_multistart, N_ARM

sys.path.insert(0, str(Path(__file__).parent / "rl"))
from grasp_env import GraspCloseEnv, EPISODE_STEPS, MAX_RESIDUAL_RATE, RESIDUAL_BOUND  # noqa: E402

SEED = 1
RUN_NAME = "close08_residual"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--run-name", type=str, default=RUN_NAME)
    args = ap.parse_args()

    run_dir = Path(__file__).parent / "rl" / "runs" / args.run_name
    policy = PPO.load(run_dir / "checkpoints" / "final")
    env = GraspCloseEnv()
    vec_env = DummyVecEnv([lambda: env])
    norm_env = VecNormalize.load(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"), vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    model, data, hand = env.model, env.data, env.hand
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, SITE)

    rng = np.random.default_rng(SEED)
    obj_xy = random_table_xy(rng)
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
    restart_rng = np.random.default_rng(40_000 + SEED)

    renderer = mj.Renderer(model, height=480, width=640)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    midpoint = (obj_pos0[:2] + place_pos0[:2]) / 2
    cam.lookat[:] = [midpoint[0], midpoint[1], 1.1]
    cam.distance = 1.3
    cam.elevation = -35
    azimuth = 20.0

    frames = []
    step_count = [0]

    def render():
        nonlocal azimuth
        azimuth += 0.08
        cam.azimuth = azimuth
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    def step(target_pos, spread, close, q_arm):
        q_arm, _, _, _ = solve_ik_multistart(model, target_pos, q_arm,
                                              restart_rng=restart_rng, **IK_KWARGS)
        hand.set_targets(data, spread, close)
        data.ctrl[:N_ARM] = q_arm
        mj.mj_forward(model, data)
        lo = model.actuator_forcerange[N_ARM:2 * N_ARM, 0]
        hi = model.actuator_forcerange[N_ARM:2 * N_ARM, 1]
        data.ctrl[N_ARM:2 * N_ARM] = np.clip(data.qfrc_bias[:N_ARM], lo, hi)
        mj.mj_step(model, data)
        step_count[0] += 1
        if step_count[0] % 60 == 0:
            render()
        return q_arm

    def step_held_grip(target_pos, q_arm):
        """Same as step(), but holds env.closes (the RL policy's per-finger grip) instead of
        commanding a uniform close -- see eval_stage5_pick_and_place_rl.py's module note on
        why overriding it with a uniform value here silently discards the learned asymmetry."""
        q_arm, _, _, _ = solve_ik_multistart(model, target_pos, q_arm,
                                              restart_rng=restart_rng, **IK_KWARGS)
        env._set_finger_closes()
        data.ctrl[:N_ARM] = q_arm
        mj.mj_forward(model, data)
        lo = model.actuator_forcerange[N_ARM:2 * N_ARM, 0]
        hi = model.actuator_forcerange[N_ARM:2 * N_ARM, 1]
        data.ctrl[N_ARM:2 * N_ARM] = np.clip(data.qfrc_bias[:N_ARM], lo, hi)
        mj.mj_step(model, data)
        step_count[0] += 1
        if step_count[0] % 60 == 0:
            render()
        return q_arm

    above = obj_pos0 + np.array([0, 0, STANDOFF])
    q_pregrasp, orient_score = best_oriented_ik(model, above, restart_rng)
    n = int(PREGRASP_SECONDS / dt)
    for i in range(n):
        hand.set_targets(data, SPREAD_PARALLEL, 0.0)
        pd_gravity_comp_step(model, data, q_pregrasp)
        step_count[0] += 1
        if step_count[0] % 60 == 0:
            render()
    q_arm = q_pregrasp.copy()

    data.qpos[env.obj_qadr:env.obj_qadr + 3] = obj_pos0
    data.qpos[env.obj_qadr + 3:env.obj_qadr + 7] = [1, 0, 0, 0]
    data.qvel[env.obj_dof:env.obj_dof + 6] = 0.0
    mj.mj_forward(model, data)

    n = int(DESCEND_SECONDS / dt)
    for i in range(n):
        frac = (i + 1) / n
        target = above + (obj_pos0 - above) * frac
        q_arm = step(target, SPREAD_PARALLEL, 0.0, q_arm)

    # --- residual-RL close phase, replacing the scripted ramp+hold ---
    env.q_arm = data.qpos[:N_ARM].copy()
    env.obj_xy_start = obj_xy.copy()
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
            step_count[0] += 1
            if step_count[0] % 60 == 0:
                render()
        env.t += 1
    q_arm = env.q_arm.copy()
    close_final = env.closes.copy()
    # --- end residual-RL close phase ---

    lifted_pick = obj_pos0 + np.array([0, 0, LIFT_HEIGHT])
    n = int(LIFT_SECONDS / dt)
    for i in range(n):
        frac = (i + 1) / n
        target = obj_pos0 + (lifted_pick - obj_pos0) * frac
        q_arm = step_held_grip(target, q_arm)

    lifted_place = place_pos0 + np.array([0, 0, LIFT_HEIGHT])
    n = int(TRANSPORT_SECONDS / dt)
    for i in range(n):
        frac = (i + 1) / n
        target = lifted_pick + (lifted_place - lifted_pick) * frac
        q_arm = step_held_grip(target, q_arm)

    n = int(LOWER_SECONDS / dt)
    for i in range(n):
        frac = (i + 1) / n
        target = lifted_place + (place_pos0 - lifted_place) * frac
        q_arm = step_held_grip(target, q_arm)

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
        step_count[0] += 1
        if step_count[0] % 60 == 0:
            render()

    retracted = place_pos0 + np.array([0, 0, STANDOFF])
    n = int(RETRACT_SECONDS / dt)
    for i in range(n):
        frac = (i + 1) / n
        target = place_pos0 + (retracted - place_pos0) * frac
        q_arm = step(target, SPREAD_PARALLEL, 0.0, q_arm)

    obj_final = data.xpos[env.obj_body].copy()
    place_xy_err = float(np.linalg.norm(obj_final[:2] - place_pos0[:2]))
    print(f"seed={SEED} obj_xy={np.round(obj_xy, 3)} place_xy={np.round(place_xy, 3)} "
          f"place_xy_err={place_xy_err:.4f}m")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=33, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
