"""Stage 5 eval: pick a sphere up off the table and place it at a second table location.

Reuses Stage 4's pick sequence unchanged (pregrasp/descend/close -- see eval_stage4_grasp.py)
through the lift phase, then adds:
  5. transport -- carry the (still-gripped) object sideways to above the place location
  6. lower     -- descend to the place location, fingers still closed
  7. release   -- hold position, ramp fingers from closed back to open
  8. retract   -- raise the grasp point clear of the now-placed object, fingers open
  9. (verify)  -- did the object end up resting near the place target, not still in the hand
     or knocked away during release?

Usage:
    python3 eval_stage5_pick_and_place.py --trials 10
"""
import argparse

import mujoco as mj
import numpy as np

from control import pd_gravity_comp_step
from hand_control import HandController, SPREAD_PARALLEL
from ik import solve_ik_multistart, N_ARM
from eval_stage4_grasp import (URDF, SITE, STANDOFF, LIFT_HEIGHT, GRIP_CLOSE, PREGRASP_SECONDS,
                                DESCEND_SECONDS, CLOSE_RAMP_SECONDS, CLOSE_HOLD_SECONDS,
                                LIFT_SECONDS, IK_KWARGS, random_table_xy, best_oriented_ik)

# Offset from the pick location to the place location, same table (TABLE_XY_RANGE spans
# roughly x in [0.5, 0.65], y in [-0.12, 0.12]; the table itself is much bigger -- see
# build_stage4_mjcf.py's TABLE_HALF -- so this offset stays comfortably on the table surface
# for every sampled pick location).
PLACE_OFFSET = np.array([0.1, 0.15])

TRANSPORT_SECONDS = 2.5
LOWER_SECONDS = 1.5
RELEASE_RAMP_SECONDS = 1.5
RETRACT_SECONDS = 1.5

PLACE_XY_TOL = 0.05  # success: object settles within 5cm of the place target, resting on the table


def run_trial(model, obj_xy, restart_rng):
    data = mj.MjData(model)
    hand = HandController(model)
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, SITE)
    obj_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "grasp_object")
    obj_joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "grasp_object_joint")
    obj_qadr = model.jnt_qposadr[obj_joint]
    obj_dof = model.jnt_dofadr[obj_joint]

    obj_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "grasp_object_geom")
    table_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "table")
    sphere_radius = model.geom_size[obj_geom][0]
    table_top_z = model.geom_pos[table_geom][2] + model.geom_size[table_geom][2]

    obj_pos0 = np.array([obj_xy[0], obj_xy[1], table_top_z + sphere_radius])
    place_xy = obj_xy + PLACE_OFFSET
    place_pos0 = np.array([place_xy[0], place_xy[1], table_top_z + sphere_radius])

    data.qpos[:N_ARM] = 0.0
    data.qpos[obj_qadr:obj_qadr + 3] = [2.0, 2.0, -1.0]
    data.qpos[obj_qadr + 3:obj_qadr + 7] = [1, 0, 0, 0]
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

    above = obj_pos0 + np.array([0, 0, STANDOFF])
    q_pregrasp, orient_score = best_oriented_ik(model, above, restart_rng)
    n = int(PREGRASP_SECONDS / dt)
    for i in range(n):
        hand.set_targets(data, SPREAD_PARALLEL, 0.0)
        pd_gravity_comp_step(model, data, q_pregrasp)
    q_arm = q_pregrasp.copy()

    # Same reasoning as eval_stage4_grasp.py: the pregrasp jump isn't collision-aware, so the
    # object is only placed on the table once the arm has safely parked above it.
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

    lifted_pick = obj_pos0 + np.array([0, 0, LIFT_HEIGHT])
    run_phase(obj_pos0, lifted_pick, SPREAD_PARALLEL, GRIP_CLOSE, LIFT_SECONDS)

    # Did Stage 4's pick actually succeed? If not, there's nothing worth carrying -- same
    # check eval_stage4_grasp.py uses, just evaluated mid-trial instead of at the end.
    obj_after_lift = data.xpos[obj_body].copy()
    pick_rise = obj_after_lift[2] - obj_pos0[2]
    pick_drift = float(np.linalg.norm(obj_after_lift[:2] - obj_pos0[:2]))
    picked = (pick_rise > 0.6 * LIFT_HEIGHT) and (pick_drift < 0.05)

    lifted_place = place_pos0 + np.array([0, 0, LIFT_HEIGHT])
    run_phase(lifted_pick, lifted_place, SPREAD_PARALLEL, GRIP_CLOSE, TRANSPORT_SECONDS)
    run_phase(lifted_place, place_pos0, SPREAD_PARALLEL, GRIP_CLOSE, LOWER_SECONDS)

    n_release = int(RELEASE_RAMP_SECONDS / dt)
    for i in range(n_release):
        close = GRIP_CLOSE * (1.0 - (i + 1) / n_release)
        q_arm, _, _, _ = solve_ik_multistart(model, place_pos0, q_arm,
                                              restart_rng=restart_rng, **IK_KWARGS)
        hand.set_targets(data, SPREAD_PARALLEL, close)
        data.ctrl[:N_ARM] = q_arm
        mj.mj_forward(model, data)
        lo = model.actuator_forcerange[N_ARM:2 * N_ARM, 0]
        hi = model.actuator_forcerange[N_ARM:2 * N_ARM, 1]
        data.ctrl[N_ARM:2 * N_ARM] = np.clip(data.qfrc_bias[:N_ARM], lo, hi)
        mj.mj_step(model, data)

    retracted = place_pos0 + np.array([0, 0, STANDOFF])
    run_phase(place_pos0, retracted, SPREAD_PARALLEL, 0.0, RETRACT_SECONDS)

    obj_final = data.xpos[obj_body].copy()
    place_xy_err = float(np.linalg.norm(obj_final[:2] - place_pos0[:2]))
    resting = abs(obj_final[2] - place_pos0[2]) < 0.02
    placed = picked and resting and (place_xy_err < PLACE_XY_TOL)
    return dict(obj_xy=obj_xy, place_xy=place_xy, picked=picked, resting=resting,
                place_xy_err=place_xy_err, placed=placed, orient_score=orient_score)


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
              f"picked={r['picked']} resting={r['resting']} "
              f"place_xy_err={r['place_xy_err']:.4f}m placed={r['placed']}")

    n_picked = sum(r["picked"] for r in results)
    n_placed = sum(r["placed"] for r in results)
    print(f"\nSummary over {args.trials} trials: picked {n_picked}/{args.trials}, "
          f"placed {n_placed}/{args.trials}")


if __name__ == "__main__":
    main()
