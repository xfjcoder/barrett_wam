"""Shared MJCF-building helpers used by both build_stage2_mjcf.py and build_stage4_mjcf.py.

Pulled out once Stage 4 needed the same arm-actuator/site/self-collision-disable logic Stage 2
already built, rather than copy-pasting it into a second script.
"""
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco as mj

HERE = Path(__file__).parent

ARM_JOINTS = [
    "wam/base_yaw_joint", "wam/shoulder_pitch_joint", "wam/shoulder_yaw_joint",
    "wam/elbow_pitch_joint", "wam/wrist_yaw_joint", "wam/wrist_pitch_joint",
    "wam/palm_yaw_joint",
]
GAINS = {
    "wam/base_yaw_joint": (80, 8),
    "wam/shoulder_pitch_joint": (80, 8),
    "wam/shoulder_yaw_joint": (60, 6),
    "wam/elbow_pitch_joint": (40, 4),
    "wam/wrist_yaw_joint": (8, 1),
    "wam/wrist_pitch_joint": (8, 1),
    "wam/palm_yaw_joint": (4, 0.5),
}
# The wrist/palm links' own rigid-body inertia (~1e-4 kg*m^2) has no reflected motor/cable
# inertia in it -- the rigid-body URDF import only captures the physical link, not the real
# cable-driven WAM's actual motors and transmission. Without it, Coriolis coupling from the
# much heavier, fast-moving shoulder/elbow links drove these joints' velocity to 15-20 rad/s
# during a reach (confirmed by direct measurement) regardless of controller tuning. `armature`
# is MuJoCo's standard mechanism for exactly this: adds a fixed rotor-inertia term to the
# joint's effective inertia for dynamics, independent of the link's own rigid-body inertia.
ARMATURE = {
    "wam/wrist_yaw_joint": 0.05,
    "wam/wrist_pitch_joint": 0.05,
    "wam/palm_yaw_joint": 0.05,
}


def compile_urdf(urdf_path):
    """Loads+compiles a URDF via MuJoCo, returns the resulting MJCF as an ElementTree."""
    with tempfile.NamedTemporaryFile(suffix=".xml") as tmp:
        model = mj.MjModel.from_xml_path(str(urdf_path))
        mj.mj_saveLastXML(tmp.name, model)
        tree = ET.parse(tmp.name)
    return tree


def find_body(root, name):
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(name)


def add_floor_and_light(worldbody):
    light = ET.SubElement(worldbody, "light")
    light.set("pos", "1 1 3")
    light.set("dir", "-1 -1 -2")
    light.set("diffuse", "0.8 0.8 0.8")

    floor = ET.SubElement(worldbody, "geom")
    floor.set("name", "floor")
    floor.set("type", "plane")
    floor.set("size", "2 2 0.1")
    floor.set("rgba", "0.5 0.5 0.55 1")
    floor.set("pos", "0 0 0")


def add_ee_site(root):
    """wrist_palm_link + (0,0,0.06) [bhand_base_joint] + (0,0,0.12) [bhand_grasp_joint] =
    (0,0,0.18) in the wrist_palm_link frame -- the BHand's real "grasp point" reference frame.
    """
    wrist_palm = find_body(root, "wam/wrist_palm_link")
    site = ET.SubElement(wrist_palm, "site")
    site.set("name", "wam/ee_site")
    site.set("pos", "0 0 0.18")
    site.set("size", "0.015")
    site.set("rgba", "1 0.2 0.2 1")


def add_target_marker(worldbody):
    """A mocap body for demo/eval scripts to visualize the current commanded target (green
    sphere, moved by setting data.mocap_pos -- no joint, never participates in the dynamics).
    """
    target_marker = ET.SubElement(worldbody, "body")
    target_marker.set("name", "target_marker")
    target_marker.set("mocap", "true")
    marker_geom = ET.SubElement(target_marker, "geom")
    marker_geom.set("type", "sphere")
    marker_geom.set("size", "0.02")
    marker_geom.set("rgba", "0.1 0.9 0.1 0.6")
    marker_geom.set("contype", "0")
    marker_geom.set("conaffinity", "0")


def add_arm_actuators(root):
    """Two actuators per arm joint: a <position> servo (kp+kv, MuJoCo's implicit PD -- safe
    regardless of a joint's inertia, unlike a hand-rolled explicit "-kd*qvel" torque, which
    diverged in testing) plus a <motor> purely for gravity-compensation feedforward (ctrl set
    to qfrc_bias each step by control.py).
    """
    joint_range = {}
    joint_frc = {}
    for j in root.iter("joint"):
        if j.get("name") in ARM_JOINTS:
            joint_range[j.get("name")] = j.get("range")
            joint_frc[j.get("name")] = j.get("actuatorfrcrange")
        if j.get("name") in ARMATURE:
            j.set("armature", str(ARMATURE[j.get("name")]))

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for name in ARM_JOINTS:
        kp, kv = GAINS[name]
        pos = ET.SubElement(actuator, "position")
        pos.set("name", name.replace("joint", "pos_act"))
        pos.set("joint", name)
        pos.set("kp", str(kp))
        pos.set("kv", str(kv))
        pos.set("ctrlrange", joint_range[name])
        pos.set("forcerange", joint_frc[name])
    for name in ARM_JOINTS:
        mot = ET.SubElement(actuator, "motor")
        mot.set("name", name.replace("joint", "ff_act"))
        mot.set("joint", name)
        mot.set("ctrlrange", joint_frc[name])
        mot.set("forcerange", joint_frc[name])
    return actuator


def disable_wam_self_collision(root):
    """Stage 1's collision meshes are each link's own visual (_fine.stl) mesh, not the
    original convex-decomposition files (MuJoCo can't import .dae) -- MuJoCo auto-convexifies
    those for collision, and the inflated hulls of adjacent links permanently overlap at every
    joint, which locks the whole arm solid under contact forces. Disabled wholesale here;
    Stage 4 adds its own purpose-built primitive collision geoms back in for the hand only.
    """
    for body in root.iter("body"):
        if not body.get("name", "").startswith("wam/"):
            continue
        for geom in body.findall("geom"):
            if geom.get("contype") != "0":
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
