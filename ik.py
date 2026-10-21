"""Damped-least-squares Jacobian IK for the WAM arm's end-effector site (wam/ee_site).

Numerical rather than closed-form: the WAM's specific link geometry would need a custom
derivation, and damped-least-squares is robust to the redundancy (7 arm joints driving a
3-DOF position target) without one.
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
