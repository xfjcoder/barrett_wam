"""Trains a PPO policy for the grasp-closing sub-problem (see grasp_env.py's module docstring
for scope and why). Mirrors go1_rl_walk's conventions (PPO + SubprocVecEnv + VecNormalize +
VecMonitor, per-run args.json for reproducibility) at a scale matched to this much smaller
problem: a 3D action, a 2-second episode, and a few hundred thousand timesteps rather than
tens of millions.

Usage:
    python3 rl/train.py --run-name close01 --timesteps 300000 --n-envs 8
"""
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from grasp_env import GraspCloseEnv

HERE = Path(__file__).parent
TRAIN_INDICES = np.arange(20, 420)  # seeds 20-419; seeds 0-19 held out, see build_states.py


def linear_schedule(initial):
    def fn(progress_remaining):
        return progress_remaining * initial
    return fn


def make_env(seed, contact_force_weight):
    def _init():
        return GraspCloseEnv(state_indices=TRAIN_INDICES, seed=seed,
                              contact_force_weight=contact_force_weight)
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None,
                     help="path to a prior run's final.zip to continue training from")
    ap.add_argument("--resume-vecnormalize", type=str, default=None,
                     help="path to that prior run's vecnormalize_final.pkl")
    ap.add_argument("--contact-force-weight", type=float, default=0.02,
                     help="see README's RL section: 1.0 to discover contact behavior (stage "
                          "1), 0.02 to correct it into a real grasp without reward hacking "
                          "(stage 2, --resume from stage 1's checkpoint)")
    args = ap.parse_args()

    run_dir = HERE / "runs" / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    env_fns = [make_env(args.seed + i, args.contact_force_weight) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)
    if args.resume_vecnormalize:
        env = VecNormalize.load(args.resume_vecnormalize, env)
        env.training = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=args.gamma)

    n_steps = args.n_steps
    batch_size = args.batch_size
    if (n_steps * args.n_envs) % batch_size != 0:
        batch_size = n_steps * args.n_envs
        print(f"n_steps*n_envs not divisible by batch_size, using batch_size={batch_size}")

    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu",
                          learning_rate=linear_schedule(args.lr))
        print(f"resumed from {args.resume}")
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=linear_schedule(args.lr),
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=args.gamma,
            ent_coef=0.0,  # see go1_rl_walk's own lesson: nonzero risked policy-std blowup there
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
            device="cpu",
            tensorboard_log=str(run_dir / "logs"),
            seed=args.seed,
            verbose=1,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.timesteps // (args.n_envs * 10), 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo",
        save_vecnormalize=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb)
    model.save(run_dir / "checkpoints" / "final")
    env.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))
    print(f"done, saved to {run_dir}/checkpoints/")


if __name__ == "__main__":
    main()
