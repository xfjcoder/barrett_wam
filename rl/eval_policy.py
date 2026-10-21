"""Evaluates a trained grasp-closing policy over the 20 held-out states (seeds 0-19, matching
eval_stage4_grasp.py's own 20-trial sweep exactly -- see build_states.py and train.py's
TRAIN_INDICES for why these specific indices were never shown to the policy during training),
and reports the same success criterion eval_stage4_grasp.py uses, for direct comparison
against the scripted close-ramp baseline (11/20).

Usage:
    python3 rl/eval_policy.py --run-name close01
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from grasp_env import EPISODE_STEPS, GraspCloseEnv

HERE = Path(__file__).parent
EVAL_INDICES = np.arange(0, 20)  # seeds 0-19; held out from training, see train.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default="final")
    args = ap.parse_args()

    run_dir = HERE / "runs" / args.run_name
    model = PPO.load(run_dir / "checkpoints" / args.checkpoint)

    raw_env = GraspCloseEnv(state_indices=EVAL_INDICES, seed=0)
    vec_env = DummyVecEnv([lambda: raw_env])
    vecnorm_path = run_dir / "checkpoints" / "vecnormalize_final.pkl"
    env = VecNormalize.load(str(vecnorm_path), vec_env)
    env.training = False
    env.norm_reward = False

    results = []
    for i, idx in enumerate(EVAL_INDICES):
        raw_env.rng = np.random.default_rng(0)
        # Force this specific held-out state instead of a random draw from EVAL_INDICES.
        raw_env.state_indices = np.array([idx])
        obs = env.reset()
        for step in range(EPISODE_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done[0]:
                break
        # TERMINAL_SUCCESS_BONUS (+20) vs TERMINAL_FAIL_PENALTY (-5) dwarfs the per-step
        # shaping terms accumulated in this same final env.step() call, so the terminal
        # reward's sign alone is a reliable success/failure signal.
        placed = reward[0] > 0
        results.append(placed)
        print(f"trial {i:2d} seed={idx:4d} final_reward={float(reward[0]):.3f} success={placed}")

    n_ok = sum(results)
    print(f"\nSummary over {len(EVAL_INDICES)} trials: grasped {n_ok}/{len(EVAL_INDICES)} "
          f"(scripted baseline: 11/20, see README's Stage 4 addendum)")


if __name__ == "__main__":
    main()
