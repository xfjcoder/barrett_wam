"""Stage 4 demo: the arm picks a sphere up off a table with the BHand.

Same five phases as eval_stage4_grasp.py (pregrasp / descend / close / hold / lift), run here
for one seed with frame capture. seed=0 is one of eval_stage4_grasp.py's own sweep's successes
(11/20 overall -- see README's Stage 4 writeup for the failure mode behind the other 9).

Usage:
    python3 demo_stage4_grasp.py --record media/stage4_grasp.gif
"""
import argparse

import mujoco as mj
import numpy as np
from PIL import Image

from control import pd_gravity_comp_step
from eval_stage4_grasp import (URDF, SITE, STANDOFF, GRIP_CLOSE, PREGRASP_SECONDS,
                                DESCEND_SECONDS, CLOSE_RAMP_SECONDS, CLOSE_HOLD_SECONDS,
                                LIFT_SECONDS, IK_KWARGS, random_table_xy, best_oriented_ik)
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
    data.qpos[:N_ARM] = 0.0
    data.qpos[obj_qadr:obj_qadr + 3] = [2.0, 2.0, -1.0]
    data.qpos[obj_qadr + 3:obj_qadr + 7] = [1, 0, 0, 0]
    mj.mj_forward(model, data)

    dt = model.opt.timestep
    restart_rng = np.random.default_rng(40_000 + SEED)

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
        if step_count[0] % 25 == 0:
            render()
        return q_arm

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

    lifted = obj_pos0 + np.array([0, 0, 0.15])
    n = int(LIFT_SECONDS / dt)
    for i in range(n):
        frac = (i + 1) / n
        target = obj_pos0 + (lifted - obj_pos0) * frac
        q_arm = step(target, SPREAD_PARALLEL, GRIP_CLOSE, q_arm)

    obj_pos_final = data.xpos[obj_body].copy()
    rise = obj_pos_final[2] - obj_pos0[2]
    xy_drift = float(np.linalg.norm(obj_pos_final[:2] - obj_pos0[:2]))
    print(f"seed={SEED} orient_score={orient_score:.3f} obj_xy={np.round(obj_xy, 3)} "
          f"rise={rise:.4f}m xy_drift={xy_drift:.4f}m")

    if args.record:
        from pathlib import Path
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=33, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
