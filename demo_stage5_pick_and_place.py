"""Stage 5 demo: the arm picks a sphere up off the table and places it at a second spot.

Same nine phases as eval_stage5_pick_and_place.py (pregrasp / descend / close / lift /
transport / lower / release / retract), run here for one seed with frame capture. seed=0 is
one of the eval sweep's successes (9/20 overall placed -- see README's Stage 5 writeup).

Usage:
    python3 demo_stage5_pick_and_place.py --record media/stage5_pick_and_place.gif
"""
import argparse

import mujoco as mj
import numpy as np
from PIL import Image

from control import pd_gravity_comp_step
from eval_stage4_grasp import (URDF, SITE, STANDOFF, LIFT_HEIGHT, GRIP_CLOSE, PREGRASP_SECONDS,
                                DESCEND_SECONDS, CLOSE_RAMP_SECONDS, CLOSE_HOLD_SECONDS,
                                LIFT_SECONDS, IK_KWARGS, random_table_xy, best_oriented_ik)
from eval_stage5_pick_and_place import (PLACE_OFFSET, TRANSPORT_SECONDS, LOWER_SECONDS,
                                         RELEASE_RAMP_SECONDS, RETRACT_SECONDS)
from hand_control import HandController, SPREAD_PARALLEL
from ik import solve_ik_multistart, N_ARM

SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=str, default=None)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(URDF)
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

    rng = np.random.default_rng(SEED)
    obj_xy = random_table_xy(rng)
    obj_pos0 = np.array([obj_xy[0], obj_xy[1], table_top_z + sphere_radius])
    place_xy = obj_xy + PLACE_OFFSET
    place_pos0 = np.array([place_xy[0], place_xy[1], table_top_z + sphere_radius])

    data.qpos[:N_ARM] = 0.0
    data.qpos[obj_qadr:obj_qadr + 3] = [2.0, 2.0, -1.0]
    data.qpos[obj_qadr + 3:obj_qadr + 7] = [1, 0, 0, 0]
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

    data.qpos[obj_qadr:obj_qadr + 3] = obj_pos0
    data.qpos[obj_qadr + 3:obj_qadr + 7] = [1, 0, 0, 0]
    data.qvel[obj_dof:obj_dof + 6] = 0.0
    mj.mj_forward(model, data)

    n = int(DESCEND_SECONDS / dt)
    for i in range(n):
        frac = (i + 1) / n
        target = above + (obj_pos0 - above) * frac
        q_arm = step(target, SPREAD_PARALLEL, 0.0, q_arm)

    n_ramp = int(CLOSE_RAMP_SECONDS / dt)
    for i in range(n_ramp):
        close = GRIP_CLOSE * (i + 1) / n_ramp
        q_arm = step(obj_pos0, SPREAD_PARALLEL, close, q_arm)

    n = int(CLOSE_HOLD_SECONDS / dt)
    for i in range(n):
        q_arm = step(obj_pos0, SPREAD_PARALLEL, GRIP_CLOSE, q_arm)

    lifted_pick = obj_pos0 + np.array([0, 0, LIFT_HEIGHT])
    n = int(LIFT_SECONDS / dt)
    for i in range(n):
        frac = (i + 1) / n
        target = obj_pos0 + (lifted_pick - obj_pos0) * frac
        q_arm = step(target, SPREAD_PARALLEL, GRIP_CLOSE, q_arm)

    lifted_place = place_pos0 + np.array([0, 0, LIFT_HEIGHT])
    n = int(TRANSPORT_SECONDS / dt)
    for i in range(n):
        frac = (i + 1) / n
        target = lifted_pick + (lifted_place - lifted_pick) * frac
        q_arm = step(target, SPREAD_PARALLEL, GRIP_CLOSE, q_arm)

    n = int(LOWER_SECONDS / dt)
    for i in range(n):
        frac = (i + 1) / n
        target = lifted_place + (place_pos0 - lifted_place) * frac
        q_arm = step(target, SPREAD_PARALLEL, GRIP_CLOSE, q_arm)

    n_release = int(RELEASE_RAMP_SECONDS / dt)
    for i in range(n_release):
        close = GRIP_CLOSE * (1.0 - (i + 1) / n_release)
        q_arm = step(place_pos0, SPREAD_PARALLEL, close, q_arm)

    retracted = place_pos0 + np.array([0, 0, STANDOFF])
    n = int(RETRACT_SECONDS / dt)
    for i in range(n):
        frac = (i + 1) / n
        target = place_pos0 + (retracted - place_pos0) * frac
        q_arm = step(target, SPREAD_PARALLEL, 0.0, q_arm)

    obj_final = data.xpos[obj_body].copy()
    place_xy_err = float(np.linalg.norm(obj_final[:2] - place_pos0[:2]))
    print(f"seed={SEED} obj_xy={np.round(obj_xy, 3)} place_xy={np.round(place_xy, 3)} "
          f"place_xy_err={place_xy_err:.4f}m")

    if args.record:
        from pathlib import Path
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=33, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
