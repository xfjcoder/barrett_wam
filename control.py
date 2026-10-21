"""PD + gravity-compensation controller for the 7 arm joints.

Two actuator sets per joint (see assets/wam7_bhand.xml's actuator comment for why):
  - a <position> servo (kp + kv), MuJoCo's own implicit PD -- the "pos_act"s, indices [0:7].
  - a <motor> actuator fed data.qfrc_bias as pure gravity/Coriolis-compensation feedforward --
    the "ff_act"s, indices [7:14].
A hand-rolled explicit -kd*qvel torque was tried first and diverged (the wrist/palm links'
tiny rotational inertia breaks the explicit-damping stability bound); MuJoCo's native kv is
integrated implicitly and has no such issue regardless of a joint's inertia.
"""
import mujoco as mj
import numpy as np

from ik import N_ARM


def pd_gravity_comp_step(model, data, q_target):
    mj.mj_forward(model, data)  # refreshes data.qfrc_bias for the current state
    data.ctrl[:N_ARM] = q_target
    lo = model.actuator_forcerange[N_ARM:2 * N_ARM, 0]
    hi = model.actuator_forcerange[N_ARM:2 * N_ARM, 1]
    data.ctrl[N_ARM:2 * N_ARM] = np.clip(data.qfrc_bias[:N_ARM], lo, hi)
    mj.mj_step(model, data)
