"""Position-only control for the 8 BHand finger joints (no gravity-comp feedforward -- see
build_stage4_mjcf.py's module docstring for why plain kp+kv was judged sufficient here).

Mirrors the real BarrettHand's underactuation in one respect: finger_1/2's first joint is the
spread (shared plane for both fingers; finger_3 has none, it's fixed opposite them). It does
NOT model the real hand's med->dist tendon transmission -- "dist" is driven open-loop off
"med" at a fixed ratio instead of the real hardware's contact-triggered breakaway coupling,
since that transmission isn't present in this URDF import (both joints are independent
revolutes with no <mimic>). Flagged as a simplification, same spirit as Stage 1/2/3's.
"""
import mujoco as mj

FINGER_1 = ("wam/bhand/finger_1/prox_joint", "wam/bhand/finger_1/med_joint",
            "wam/bhand/finger_1/dist_joint")
FINGER_2 = ("wam/bhand/finger_2/prox_joint", "wam/bhand/finger_2/med_joint",
            "wam/bhand/finger_2/dist_joint")
FINGER_3 = (None, "wam/bhand/finger_3/med_joint", "wam/bhand/finger_3/dist_joint")
FINGERS = (FINGER_1, FINGER_2, FINGER_3)

MED_MAX = 2.443460952792061     # finger_*/med_joint upper limit (assets/wam7_bhand.urdf)
DIST_MAX = 0.8377580409572781   # finger_*/dist_joint upper limit
DIST_TO_MED_RATIO = 0.33        # approximates the real hand's ~1:3 med->dist coupling
SPREAD_PARALLEL = 0.0           # finger_1/2 prox angle for a side-by-side preshape


class HandController:
    def __init__(self, model):
        self.act_ids = {}
        for finger in FINGERS:
            for joint in finger:
                if joint is None:
                    continue
                act_name = joint.replace("joint", "pos_act")
                self.act_ids[joint] = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, act_name)

    def set_targets(self, data, spread, close):
        """spread: finger_1/2 prox angle (rad). close: 0 (open) .. 1 (fully curled)."""
        med = close * MED_MAX
        dist = min(close * MED_MAX * DIST_TO_MED_RATIO, DIST_MAX)
        for prox_j, med_j, dist_j in FINGERS:
            if prox_j is not None:
                data.ctrl[self.act_ids[prox_j]] = spread
            data.ctrl[self.act_ids[med_j]] = med
            data.ctrl[self.act_ids[dist_j]] = dist
