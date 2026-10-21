"""Stage 4 eval: pick up a sphere off a table with the BHand, over randomized object positions.

Five fixed phases per trial, each driving the arm toward the grasp-point site (wam/ee_site)
while hand_control.py commands the fingers open/closed:
  1. pregrasp -- jump to a hand-picked, well-oriented joint config above the object (see
     best_oriented_ik below), fingers open
  2. descend  -- lower the grasp point onto the object's center, fingers still open
  3. close    -- hold position, ramp fingers from open to closed
  4. lift     -- raise the grasp point, fingers held closed
  5. (verify) -- did the object rise with the hand, or get left behind / knocked away?

Usage:
    python3 eval_stage4_grasp.py --trials 10
"""
import argparse

import mujoco as mj
import numpy as np

from control import pd_gravity_comp_step
from hand_control import HandController, SPREAD_PARALLEL
from ik import solve_ik, solve_ik_multistart, N_ARM

URDF = "assets/wam7_bhand_grasp.xml"
SITE = "wam/ee_site"

TABLE_XY_RANGE = ((0.5, 0.65), (-0.12, 0.12))  # (x_lo, x_hi), (y_lo, y_hi)

STANDOFF = 0.15
LIFT_HEIGHT = 0.15

# Counterintuitively, close=1.0 (full curl) is NOT the tightest fit: med/dist's fixed coupling
# ratio (hand_control.py's DIST_TO_MED_RATIO) makes the fingertip's distance to the grasp-
# point site bottom out around close=0.7-0.75 and then swing back OUT again past that --
# confirmed by direct measurement (tip-to-ee distance at close=1.0 was *larger* than at
# close=0.75). Driving all the way to close=1.0 was quietly retracting the fingers off the
# object rather than gripping it. GRIP_CLOSE is that measured minimum, not a round number.
GRIP_CLOSE = 0.75

PREGRASP_SECONDS = 4.0  # home -> standoff is a large displacement; see run_trial's note
DESCEND_SECONDS = 1.5
CLOSE_RAMP_SECONDS = 2.5
CLOSE_HOLD_SECONDS = 1.5
LIFT_SECONDS = 2.0

# Position-only (see ik.py's module docstring for why an orientation constraint was tried and
# dropped) -- used for the continuous small-step warm-started solves (descend/close/hold/lift).
# Picking a good *initial* orientation is best_oriented_ik's job, below.
IK_KWARGS = dict(max_iters=30, tol=0.01)


def random_table_xy(rng):
    (xlo, xhi), (ylo, yhi) = TABLE_XY_RANGE
    return np.array([rng.uniform(xlo, xhi), rng.uniform(ylo, yhi)])


def best_oriented_ik(model, target_pos, rng, n_candidates=40):
    """Position-only IK (ik.py's solve_ik) has 4 redundant DOF free for a 7-joint arm driving
    a 3-DOF position target, and which one it lands on is NOT just a function of the target --
    it's path-dependent. Confirmed directly: continuously warm-starting small steps from home
    out to a table-height standoff point consistently landed on an "upside-down" branch (palm
    tilted up/back, local z's world-frame z-component *positive*), while a single-shot solve
    from a *different* random start, for the exact same target, just as often landed on a
    sensible downward-facing branch. Rather than fight that path-dependence, this tries many
    random starting postures, keeps only the ones where position IK actually converges, and
    explicitly picks whichever valid solution has the palm facing most nearly straight down --
    the thing an orientation-constrained IK was trying (and failing, see ik.py) to guarantee.
    """
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "wam/ee_site")
    data = mj.MjData(model)
    lo = model.jnt_range[:N_ARM, 0]
    hi = model.jnt_range[:N_ARM, 1]
    best_q, best_score = None, -2.0
    for i in range(n_candidates):
        q_init = np.zeros(N_ARM) if i == 0 else rng.uniform(lo, hi)
        q, ok, _ = solve_ik(model, target_pos, q_init, max_iters=150, tol=0.01)
        if not ok:
            continue
        data.qpos[:N_ARM] = q
        mj.mj_forward(model, data)
        local_z_world = data.site_xmat[site_id].reshape(3, 3)[:, 2]
        score = -local_z_world[2]  # +1 if local z points straight down (0,0,-1), -1 if up
        if score > best_score:
            best_score, best_q = score, q
    return best_q, best_score


def run_trial(model, obj_xy, restart_rng):
    data = mj.MjData(model)
    hand = HandController(model)
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, SITE)
    obj_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "grasp_object")
    obj_joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "grasp_object_joint")
    obj_qadr = model.jnt_qposadr[obj_joint]
    obj_dof = model.jnt_dofadr[obj_joint]

    # Read the object/table's actual size straight from the compiled model rather than
    # keeping a second hardcoded copy in sync with build_stage4_mjcf.py's source of truth.
    obj_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "grasp_object_geom")
    table_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "table")
    sphere_radius = model.geom_size[obj_geom][0]
    table_top_z = model.geom_pos[table_geom][2] + model.geom_size[table_geom][2]

    obj_pos0 = np.array([obj_xy[0], obj_xy[1], table_top_z + sphere_radius])
    data.qpos[:N_ARM] = 0.0
    # Parked out of the way during pregrasp, not yet at obj_pos0 -- see the note below PREGRASP.
    data.qpos[obj_qadr:obj_qadr + 3] = [2.0, 2.0, -1.0]
    data.qpos[obj_qadr + 3:obj_qadr + 7] = [1, 0, 0, 0]
    mj.mj_forward(model, data)

    dt = model.opt.timestep
    q_arm = data.qpos[:N_ARM].copy()

    def run_phase(start_pos, end_pos, spread, close, seconds):
        """Interpolates the IK target from start_pos to end_pos over the phase, instead of
        jumping straight to end_pos -- an open, straight finger_3 points right along the
        approach axis (see hand_control.py), so jumping the IK target straight down to the
        object's center let the controller's own transient path swing that finger sideways
        through the ball first (confirmed by rendering the trajectory: the object was already
        in contact, and knocked off-table, before the close phase even started). Interpolating
        keeps the commanded path a straight vertical line the whole way down.
        """
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

    above = obj_pos0 + np.array([0, 0, STANDOFF])
    q_pregrasp, orient_score = best_oriented_ik(model, above, restart_rng)
    n = int(PREGRASP_SECONDS / dt)
    for i in range(n):
        hand.set_targets(data, SPREAD_PARALLEL, 0.0)
        pd_gravity_comp_step(model, data, q_pregrasp)
    q_arm = q_pregrasp.copy()

    # best_oriented_ik's chosen joint config can be a large, unplanned joint-space jump from
    # home -- the swept Cartesian path on the way there isn't controlled at all (confirmed:
    # with the object already on the table, the hand swept through and launched it partway
    # through pregrasp, well before descend). The object is only placed on the table once the
    # arm has safely parked at the standoff pose above it -- modeling an idealized reach where
    # a real motion planner would route the approach around the table; planning that route is
    # a separate problem from grasping itself and out of scope here.
    data.qpos[obj_qadr:obj_qadr + 3] = obj_pos0
    data.qpos[obj_qadr + 3:obj_qadr + 7] = [1, 0, 0, 0]
    data.qvel[obj_dof:obj_dof + 6] = 0.0
    mj.mj_forward(model, data)

    run_phase(above, obj_pos0, SPREAD_PARALLEL, 0.0, DESCEND_SECONDS)

    n_ramp = int(CLOSE_RAMP_SECONDS / dt)
    for i in range(n_ramp):
        close = GRIP_CLOSE * (i + 1) / n_ramp
        q_arm, _, _, _ = solve_ik_multistart(model, obj_pos0, q_arm,
                                              restart_rng=restart_rng, **IK_KWARGS)
        hand.set_targets(data, SPREAD_PARALLEL, close)
        data.ctrl[:N_ARM] = q_arm
        mj.mj_forward(model, data)
        lo = model.actuator_forcerange[N_ARM:2 * N_ARM, 0]
        hi = model.actuator_forcerange[N_ARM:2 * N_ARM, 1]
        data.ctrl[N_ARM:2 * N_ARM] = np.clip(data.qfrc_bias[:N_ARM], lo, hi)
        mj.mj_step(model, data)
    run_phase(obj_pos0, obj_pos0, SPREAD_PARALLEL, GRIP_CLOSE, CLOSE_HOLD_SECONDS)

    lifted = obj_pos0 + np.array([0, 0, LIFT_HEIGHT])
    run_phase(obj_pos0, lifted, SPREAD_PARALLEL, GRIP_CLOSE, LIFT_SECONDS)

    obj_pos_final = data.xpos[obj_body].copy()
    rise = obj_pos_final[2] - obj_pos0[2]
    xy_drift = float(np.linalg.norm(obj_pos_final[:2] - obj_pos0[:2]))
    success = (rise > 0.6 * LIFT_HEIGHT) and (xy_drift < 0.05)
    return dict(obj_xy=obj_xy, rise=rise, xy_drift=xy_drift, success=success,
                orient_score=orient_score,
                final_ee_dist=float(np.linalg.norm(data.site_xpos[site_id] - lifted)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=0)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(URDF)
    results = []
    for i in range(args.trials):
        seed = args.base_seed + i
        rng = np.random.default_rng(seed)
        obj_xy = random_table_xy(rng)
        restart_rng = np.random.default_rng(40_000 + seed)
        r = run_trial(model, obj_xy, restart_rng)
        results.append(r)
        print(f"trial {i:2d} seed={seed:4d} obj_xy={np.round(obj_xy, 3)} "
              f"orient_score={r['orient_score']:.3f} "
              f"rise={r['rise']:.4f}m xy_drift={r['xy_drift']:.4f}m "
              f"ee_track_err={r['final_ee_dist']:.4f}m success={r['success']}")

    n_ok = sum(r["success"] for r in results)
    print(f"\nSummary over {args.trials} trials: grasped {n_ok}/{args.trials}")


if __name__ == "__main__":
    main()
