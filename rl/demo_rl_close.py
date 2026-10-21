"""RL demo: scripted pregrasp/descend (identical to Stage 4), then the trained PPO policy
controls per-finger closing (the sub-problem it was trained on, see grasp_env.py's module
docstring), then the same scripted lift+check as Stage 4. Run once per seed to compare a
success against a failure -- see README's RL section for the held-out eval (4/20 for
close06_stage2): seeds 2/10/11/19 succeed, seed 0 is representative of the 16 failures.

Usage:
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 rl/demo_rl_close.py --seed 2 \
        --record media/rl_close_success.gif
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 rl/demo_rl_close.py --seed 0 \
        --record media/rl_close_failure.gif
"""
import argparse
import sys
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent.parent))
from control import pd_gravity_comp_step
from eval_stage4_grasp import (STANDOFF, PREGRASP_SECONDS, DESCEND_SECONDS, LIFT_HEIGHT,
                                LIFT_SECONDS, random_table_xy, best_oriented_ik)
from hand_control import SPREAD_PARALLEL
from ik import solve_ik_multistart, N_ARM
from grasp_env import GraspCloseEnv, EPISODE_STEPS, FRAME_SKIP, MAX_RESIDUAL_RATE, RESIDUAL_BOUND

RUN_NAME = "close06_stage2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--run-name", type=str, default=RUN_NAME)
    ap.add_argument("--record", type=str, default=None)
    args = ap.parse_args()
    seed = args.seed

    run_dir = Path(__file__).parent / "runs" / args.run_name
    policy = PPO.load(run_dir / "checkpoints" / "final")

    env = GraspCloseEnv(state_indices=np.array([seed]))
    vec_env = DummyVecEnv([lambda: env])
    norm_env = VecNormalize.load(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"), vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    model = env.model
    data = env.data
    hand = env.hand

    rng = np.random.default_rng(seed)
    obj_xy = random_table_xy(rng)
    obj_pos0 = np.array([obj_xy[0], obj_xy[1], env.table_top_z + env.sphere_radius])
    data.qpos[:N_ARM] = 0.0
    data.qpos[env.obj_qadr:env.obj_qadr + 3] = [2.0, 2.0, -1.0]
    data.qpos[env.obj_qadr + 3:env.obj_qadr + 7] = [1, 0, 0, 0]
    mj.mj_forward(model, data)

    dt = model.opt.timestep
    restart_rng = np.random.default_rng(40_000 + seed)

    renderer = mj.Renderer(model, height=480, width=640)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [obj_pos0[0], obj_pos0[1], 1.1]
    cam.distance = 0.9
    cam.elevation = -25
    azimuth = 30.0
    frames = []

    def render():
        nonlocal azimuth
        azimuth += 0.1
        cam.azimuth = azimuth
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    step_count = [0]

    above = obj_pos0 + np.array([0, 0, STANDOFF])
    q_pregrasp, orient_score = best_oriented_ik(model, above, restart_rng)
    n = int(PREGRASP_SECONDS / dt)
    for i in range(n):
        hand.set_targets(data, SPREAD_PARALLEL, 0.0)
        pd_gravity_comp_step(model, data, q_pregrasp)
        step_count[0] += 1
        if step_count[0] % 25 == 0:
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
        q_arm, _, _, _ = solve_ik_multistart(model, target, q_arm, restart_rng=restart_rng,
                                              max_iters=30, tol=0.01)
        hand.set_targets(data, SPREAD_PARALLEL, 0.0)
        data.ctrl[:N_ARM] = q_arm
        mj.mj_forward(model, data)
        lo = model.actuator_forcerange[N_ARM:2 * N_ARM, 0]
        hi = model.actuator_forcerange[N_ARM:2 * N_ARM, 1]
        data.ctrl[N_ARM:2 * N_ARM] = np.clip(data.qfrc_bias[:N_ARM], lo, hi)
        mj.mj_step(model, data)
        step_count[0] += 1
        if step_count[0] % 25 == 0:
            render()

    # hand control over to the RL policy for the close phase (grasp_env.py's GraspCloseEnv,
    # driven directly rather than through its reset(), so we can render the live trajectory
    # instead of teleporting to a cached post-descend snapshot). q_arm here must be the
    # ACHIEVED joint position (data.qpos), not the last commanded IK target -- that's what
    # states.npz caches and what reset() holds the arm at; the two differ by the PD
    # controller's settling error, which is enough to flip a marginal grasp outcome.
    env.q_arm = data.qpos[:N_ARM].copy()
    env.obj_xy_start = obj_xy.copy()
    env.t = 0
    env.residual = {"finger_1": 0.0, "finger_2": 0.0, "finger_3": 0.0}
    env.closes = {"finger_1": 0.0, "finger_2": 0.0, "finger_3": 0.0}
    env._update_closes()
    env._set_finger_closes()
    mj.mj_forward(model, data)

    for i in range(EPISODE_STEPS):
        obs_raw = env._get_obs()
        obs_norm = norm_env.normalize_obs(obs_raw[None, :])
        action, _ = policy.predict(obs_norm, deterministic=True)
        action = np.clip(action[0], -1.0, 1.0)
        for j, finger in enumerate(("finger_1", "finger_2", "finger_3")):
            env.residual[finger] = float(np.clip(
                env.residual[finger] + action[j] * MAX_RESIDUAL_RATE,
                -RESIDUAL_BOUND, RESIDUAL_BOUND))
        env._update_closes()
        for _ in range(FRAME_SKIP):
            env._set_finger_closes()
            env._apply_arm_hold()
            step_count[0] += 1
            if step_count[0] % 25 == 0:
                render()
        env.t += 1

    # scripted lift + success check, identical to grasp_env.py's _scripted_lift_check
    obj_pos0_lift = data.xpos[env.obj_body].copy()
    lifted = obj_pos0_lift + np.array([0, 0, LIFT_HEIGHT])
    n = int(LIFT_SECONDS / dt)
    q_arm = env.q_arm.copy()
    lift_rng = np.random.default_rng(0)
    for i in range(n):
        frac = (i + 1) / n
        target = obj_pos0_lift + (lifted - obj_pos0_lift) * frac
        q_arm, _, _, _ = solve_ik_multistart(model, target, q_arm, restart_rng=lift_rng,
                                              max_iters=30, tol=0.01)
        env._set_finger_closes()
        data.ctrl[:N_ARM] = q_arm
        mj.mj_forward(model, data)
        lo = model.actuator_forcerange[N_ARM:2 * N_ARM, 0]
        hi = model.actuator_forcerange[N_ARM:2 * N_ARM, 1]
        data.ctrl[N_ARM:2 * N_ARM] = np.clip(data.qfrc_bias[:N_ARM], lo, hi)
        mj.mj_step(model, data)
        step_count[0] += 1
        if step_count[0] % 25 == 0:
            render()

    obj_final = data.xpos[env.obj_body].copy()
    rise = obj_final[2] - obj_pos0_lift[2]
    xy_drift = float(np.linalg.norm(obj_final[:2] - obj_pos0_lift[:2]))
    success = (rise > 0.6 * LIFT_HEIGHT) and (xy_drift < 0.05)
    print(f"seed={seed} run={args.run_name} orient_score={orient_score:.3f} obj_xy={np.round(obj_xy, 3)} "
          f"rise={rise:.4f}m xy_drift={xy_drift:.4f}m success={success}")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=33, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
