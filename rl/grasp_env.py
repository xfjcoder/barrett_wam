"""Gymnasium env for the grasp-closing sub-problem identified in README's Stage 4 addendum:
scripted pregrasp/descend/lift are already reliable (see eval_stage4_grasp.py), but the
lockstep close ramp can't react to contact -- once one finger touches the object, the others
keep sweeping toward its *original* position and miss it after it's nudged. This env hands
control of exactly that phase (and only that phase) to an RL policy: the arm holds position
(the already-solved part), and the policy outputs per-finger closing *rates* with per-finger
contact force as part of the observation, so it can in principle learn to slow down or stop a
finger that's already made contact while the others catch up -- the thing open-loop scripting
structurally can't do.

Episode: reset -> sample a cached post-descend state (see build_states.py) -> FRAME_SKIP*
EPISODE_STEPS physics steps of RL-controlled closing -> a scripted lift+check exactly like
eval_stage4_grasp.py's, whose outcome becomes the terminal reward. The arm is NOT controlled
by the policy at all; it's held at the cached pregrasp/descend joint target throughout.
"""
import sys
from pathlib import Path

import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).parent.parent))
from control import pd_gravity_comp_step
from eval_stage4_grasp import URDF, LIFT_HEIGHT, LIFT_SECONDS
from hand_control import (DIST_MAX, DIST_TO_MED_RATIO, FINGER_1, FINGER_2, FINGER_3, FINGERS,
                           MED_MAX, HandController, SPREAD_PARALLEL)
from ik import N_ARM, solve_ik_multistart

# eval_stage4_grasp.URDF is relative to the repo root (fine for scripts always run from there);
# resolve it against this file's location instead, since SubprocVecEnv worker processes don't
# inherit the launching script's cwd assumption reliably.
URDF = str(Path(__file__).parent.parent / URDF)

STATES_PATH = Path(__file__).parent / "states.npz"

FRAME_SKIP = 10
EPISODE_STEPS = 100  # 100 * FRAME_SKIP * dt(0.002) = 2.0s of RL-controlled closing
MAX_CLOSE_RATE = 0.05  # max change in a finger's close value per env step (saturates in 20)

# A first training run (500k timesteps) with a threshold-count contact bonus (+W_CONTACT per
# finger above a force threshold) got 0/20 on held-out eval -- every single episode, the exact
# same ~-5.02 terminal reward, meaning the terminal success bonus was NEVER observed once in
# ~5000 training episodes: a threshold/count bonus gives zero gradient below the threshold, so
# there's no signal pushing a near-random initial policy toward the specific, multi-step-
# coordinated behavior needed to ever trigger it even once.
#
# Switching to a continuous per-finger contact-force reward (W_CONTACT_FORCE below) didn't fix
# it either -- same result, 0/20, and the training curve showed WHY: reward briefly spiked
# positive (actual successes) around 1/5 of the way through training, then collapsed back to
# the same ~-8 floor as before. With the original W_DRIFT=2.0, a *failed* attempt to close
# (drift penalty on top of the guaranteed -5 terminal failure penalty) scored worse than just
# not moving (drift=0, same guaranteed -5) -- so once the policy drifted toward caution, there
# was no gradient pulling it back out: trying and failing looks strictly worse than not
# trying, even though trying is required to ever succeed. Lowered W_DRIFT (2.0 -> 0.5) so a
# failed attempt isn't penalized harder than inaction, and training runs longer (1M vs 500k
# timesteps) to give it more chances to find and consolidate a reliable success.
W_DRIFT = 0.5
W_ACTION = 0.02     # per-step penalty on action magnitude (discourage jittering)
# W_DRIFT=0.5 + W_CONTACT_FORCE=1.0 (above) finally produced real training-time successes
# (ep_rew_mean climbed from -8 to 40-130) -- but held-out eval was STILL 0/20, deterministic
# and stochastic alike, and per-episode cumulative reward was often 15-60 despite the final
# (terminal) reward always being negative. That's reward hacking: 3 fingers x up to 1.0 force
# each x 100 steps caps the per-step term's episode total around 300, vastly outweighing the
# +20 terminal success bonus, so the optimal policy under this reward is "squeeze as hard as
# possible," which farms proxy reward every step regardless of whether that squeeze actually
# produces a grip that survives the lift check. Rescaled down 50x so the dense term can only
# ever contribute a few points total per episode -- a gradient hint, not a competing objective.
W_CONTACT_FORCE = 0.02
CONTACT_FORCE_CAP = 1.0  # clip each finger's force before summing, so squeezing harder than
                         # needed doesn't keep paying out (that's what overshoots the grip)
TERMINAL_SUCCESS_BONUS = 20.0
TERMINAL_FAIL_PENALTY = -5.0
FALL_Z_MARGIN = 0.1  # early-terminate (object fell off/through the table) if z drops this far


class FingerContactSensor:
    """Per-finger total normal contact force against one target geom (the object). See
    README's Stage 4 addendum for why this was built: a lockstep close ramp can't react to
    per-finger contact, which is exactly the capability this observation feeds an RL policy."""

    def __init__(self, model):
        self.geom_ids = {"finger_1": [], "finger_2": [], "finger_3": []}
        for gi in range(model.ngeom):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gi)
            if not name:
                continue
            for finger in self.geom_ids:
                if name.startswith(f"wam_bhand_{finger}_"):
                    self.geom_ids[finger].append(gi)

    def forces(self, model, data, obj_geom_id):
        out = {name: 0.0 for name in self.geom_ids}
        force = np.zeros(6)
        for c in range(data.ncon):
            con = data.contact[c]
            if obj_geom_id not in (con.geom1, con.geom2):
                continue
            other = con.geom2 if con.geom1 == obj_geom_id else con.geom1
            for finger, geoms in self.geom_ids.items():
                if other in geoms:
                    mj.mj_contactForce(model, data, c, force)
                    out[finger] += abs(float(force[0]))
        return out


class GraspCloseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, states_path=STATES_PATH, state_indices=None, seed=None,
                 contact_force_weight=W_CONTACT_FORCE):
        super().__init__()
        self.contact_force_weight = contact_force_weight
        self.model = mj.MjModel.from_xml_path(str(URDF))
        self.data = mj.MjData(self.model)
        self.hand = HandController(self.model)
        self.sensor = FingerContactSensor(self.model)

        self.site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "wam/ee_site")
        self.obj_body = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "grasp_object")
        obj_joint = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "grasp_object_joint")
        self.obj_qadr = self.model.jnt_qposadr[obj_joint]
        self.obj_dof = self.model.jnt_dofadr[obj_joint]
        obj_geom = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "grasp_object_geom")
        self.obj_geom = obj_geom
        table_geom = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "table")
        self.sphere_radius = self.model.geom_size[obj_geom][0]
        self.table_top_z = self.model.geom_pos[table_geom][2] + self.model.geom_size[table_geom][2]

        states = np.load(states_path)
        self.pool = {k: states[k] for k in states.files}
        self.n_pool = len(self.pool["obj_xy"])
        self.state_indices = state_indices if state_indices is not None else np.arange(self.n_pool)

        obs_dim = 8 + 8 + 3 + 3 + 3  # hand qpos, hand qvel, contact force, obj pos rel ee, obj linvel
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        self._hand_qadr = self._hand_joint_qpos_addrs()
        self._hand_dof = self._hand_joint_dof_addrs()
        self.rng = np.random.default_rng(seed)

    def _hand_joint_qpos_addrs(self):
        addrs = []
        for finger in FINGERS:
            for j in finger:
                if j is None:
                    continue
                jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, j)
                addrs.append(self.model.jnt_qposadr[jid])
        return np.array(addrs)

    def _hand_joint_dof_addrs(self):
        addrs = []
        for finger in FINGERS:
            for j in finger:
                if j is None:
                    continue
                jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, j)
                addrs.append(self.model.jnt_dofadr[jid])
        return np.array(addrs)

    def _apply_arm_hold(self):
        pd_gravity_comp_step(self.model, self.data, self.q_arm)

    def _get_obs(self):
        hand_qpos = self.data.qpos[self._hand_qadr]
        hand_qvel = self.data.qvel[self._hand_dof]
        forces = self.sensor.forces(self.model, self.data, self.obj_geom)
        force_vec = np.array([forces["finger_1"], forces["finger_2"], forces["finger_3"]])
        obj_pos = self.data.xpos[self.obj_body]
        obj_rel = obj_pos - self.data.site_xpos[self.site_id]
        obj_linvel = self.data.qvel[self.obj_dof:self.obj_dof + 3]
        obs = np.concatenate([hand_qpos, hand_qvel, np.clip(force_vec, 0, 5.0), obj_rel, obj_linvel])
        return obs.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        idx = self.rng.choice(self.state_indices)

        self.data.qpos[:N_ARM] = self.pool["arm_qpos"][idx]
        self.data.qvel[:N_ARM] = self.pool["arm_qvel"][idx]
        self.data.qpos[self.obj_qadr:self.obj_qadr + 7] = self.pool["obj_qpos"][idx]
        self.data.qvel[self.obj_dof:self.obj_dof + 6] = self.pool["obj_qvel"][idx]
        self.q_arm = self.pool["arm_qpos"][idx].copy()
        self.obj_xy_start = self.pool["obj_xy"][idx].copy()
        self.closes = {"finger_1": 0.0, "finger_2": 0.0, "finger_3": 0.0}
        self._set_finger_closes()
        mj.mj_forward(self.model, self.data)

        self.t = 0
        return self._get_obs(), {}

    def _set_finger_closes(self):
        """hand_control.HandController only exposes the uniform set_targets -- apply per-
        finger close values directly via the same actuator ids it already resolved."""
        for finger, close in self.closes.items():
            prox_j, med_j, dist_j = {"finger_1": FINGER_1, "finger_2": FINGER_2,
                                      "finger_3": FINGER_3}[finger]
            med = close * MED_MAX
            dist = min(close * MED_MAX * DIST_TO_MED_RATIO, DIST_MAX)
            if prox_j is not None:
                self.data.ctrl[self.hand.act_ids[prox_j]] = SPREAD_PARALLEL
            self.data.ctrl[self.hand.act_ids[med_j]] = med
            self.data.ctrl[self.hand.act_ids[dist_j]] = dist

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        for i, finger in enumerate(("finger_1", "finger_2", "finger_3")):
            self.closes[finger] = float(np.clip(
                self.closes[finger] + action[i] * MAX_CLOSE_RATE, 0.0, 1.0))

        for _ in range(FRAME_SKIP):
            self._set_finger_closes()
            self._apply_arm_hold()

        forces = self.sensor.forces(self.model, self.data, self.obj_geom)
        n_contact = sum(1 for f in forces.values() if f > 0.02)
        contact_force_reward = sum(min(f, CONTACT_FORCE_CAP) for f in forces.values())
        drift = float(np.linalg.norm(self.data.xpos[self.obj_body][:2] - self.obj_xy_start))

        reward = (self.contact_force_weight * contact_force_reward
                  - W_DRIFT * drift
                  - W_ACTION * float(np.sum(action ** 2)))

        self.t += 1
        obj_z = self.data.xpos[self.obj_body][2]
        fell = obj_z < self.table_top_z - FALL_Z_MARGIN
        terminated = False
        truncated = False

        if fell:
            reward += TERMINAL_FAIL_PENALTY
            terminated = True
        elif self.t >= EPISODE_STEPS:
            success = self._scripted_lift_check()
            reward += TERMINAL_SUCCESS_BONUS if success else TERMINAL_FAIL_PENALTY
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {"n_contact": n_contact}

    def _scripted_lift_check(self):
        """Same lift phase and success criteria as eval_stage4_grasp.py's run_trial, applied
        to whatever per-finger closes the policy ended the episode holding."""
        obj_pos0 = self.data.xpos[self.obj_body].copy()
        lifted = obj_pos0 + np.array([0, 0, LIFT_HEIGHT])
        dt = self.model.opt.timestep
        n = int(LIFT_SECONDS / dt)
        q_arm = self.q_arm.copy()
        restart_rng = np.random.default_rng(0)
        for i in range(n):
            frac = (i + 1) / n
            target = obj_pos0 + (lifted - obj_pos0) * frac
            q_arm, _, _, _ = solve_ik_multistart(self.model, target, q_arm,
                                                  restart_rng=restart_rng, max_iters=30, tol=0.01)
            self._set_finger_closes()
            self.data.ctrl[:N_ARM] = q_arm
            mj.mj_forward(self.model, self.data)
            lo = self.model.actuator_forcerange[N_ARM:2 * N_ARM, 0]
            hi = self.model.actuator_forcerange[N_ARM:2 * N_ARM, 1]
            self.data.ctrl[N_ARM:2 * N_ARM] = np.clip(self.data.qfrc_bias[:N_ARM], lo, hi)
            mj.mj_step(self.model, self.data)
        obj_final = self.data.xpos[self.obj_body]
        rise = obj_final[2] - obj_pos0[2]
        xy_drift = float(np.linalg.norm(obj_final[:2] - obj_pos0[:2]))
        return (rise > 0.6 * LIFT_HEIGHT) and (xy_drift < 0.05)
