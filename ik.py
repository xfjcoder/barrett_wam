"""Damped-least-squares Jacobian IK for the WAM arm's end-effector site (wam/ee_site).

Numerical rather than closed-form: the WAM's specific link geometry would need a custom
derivation, and damped-least-squares is robust to the redundancy (7 arm joints driving a
3-DOF position target) without one.

Position-only -- an orientation-constrained variant (matching the palm's approach direction
to straight down, for Stage 4 grasping) was tried and dropped: at the table-height region
Stage 4 actually reaches into, demanding an exact "straight down" palm convincingly fights the
arm's natural geometry there (3 joints pinned at their range limits trying to satisfy it,
residual error frozen, same deadlock signature as the joint-limit issue below but *not*
fixable by multistart here -- every random restart found the same fight). Turned out
unnecessary anyway: plain position-only IK, warm-started from home, already lands on a
consistent, sensibly-downward-tilted orientation in that region on its own (no constraint
needed) -- good enough for a sphere, which looks the same from any roll angle. See Stage 4's
README section for the numbers.
"""
import mujoco as mj
import numpy as np

N_ARM = 7


def solve_ik(model, target_pos, q_init, site_name="wam/ee_site",
             max_iters=200, tol=1e-3, damping=0.05, max_step=0.2):
    """Returns (q_arm, reached, n_iters). q_arm is length-7, clipped to joint ranges."""
    data = mj.MjData(model)
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    q = np.array(q_init[:N_ARM], dtype=float)
    lo = model.jnt_range[:N_ARM, 0]
    hi = model.jnt_range[:N_ARM, 1]
    jacp = np.zeros((3, model.nv))

    for it in range(max_iters):
        data.qpos[:N_ARM] = q
        mj.mj_forward(model, data)
        err = target_pos - data.site_xpos[site_id]
        if np.linalg.norm(err) < tol:
            return q, True, it
        mj.mj_jacSite(model, data, jacp, None, site_id)
        J = jacp[:, :N_ARM]
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), err)
        step = np.clip(dq, -max_step, max_step)
        q = np.clip(q + step, lo, hi)

    data.qpos[:N_ARM] = q
    mj.mj_forward(model, data)
    err = target_pos - data.site_xpos[site_id]
    return q, np.linalg.norm(err) < tol, max_iters


def solve_ik_multistart(model, target_pos, q_init, site_name="wam/ee_site",
                         n_restarts=20, restart_rng=None, **kwargs):
    """solve_ik, falling back to random-restart if the warm-started solve gets stuck.

    Damped-least-squares IK with joint-limit clipping can deadlock: the descent direction
    that would reduce Cartesian error pushes an already-at-limit joint further out, clipping
    resets it to the limit every iteration, and if every error-reducing direction needs an
    already-pinned joint, the solver has nowhere left to go -- error stops decreasing for good,
    not just slowly. Confirmed directly (see barrett_wam's Stage 2/3 writeups): 3+ joints
    pinned exactly at their range limits, residual error frozen for 150+ iterations. A
    different starting posture almost always avoids pinning those same joints -- every
    deadlock found in practice resolved on the very first random restart.
    """
    q, ok, it = solve_ik(model, target_pos, q_init, site_name=site_name, **kwargs)
    if ok:
        return q, True, it, 0
    if restart_rng is None:
        restart_rng = np.random.default_rng(0)
    lo = model.jnt_range[:N_ARM, 0]
    hi = model.jnt_range[:N_ARM, 1]
    for r in range(n_restarts):
        q_try = restart_rng.uniform(lo, hi)
        q2, ok2, it2 = solve_ik(model, target_pos, q_try, site_name=site_name, **kwargs)
        if ok2:
            return q2, True, it2, r + 1
    return q, False, it, n_restarts
