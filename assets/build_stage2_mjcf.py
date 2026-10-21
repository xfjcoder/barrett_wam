"""Builds assets/wam7_bhand.xml (Stage 2's MJCF) from assets/wam7_bhand.urdf (Stage 1's).

MuJoCo's URDF importer has no mechanism to add actuators, sites, or mocap bodies -- so this
compiles the URDF once (mj.mj_saveLastXML) and hand-edits the resulting MJCF to add them. See
the top-level README's Stage 2 section for why each addition below was needed.

Usage (run from the repo root):
    python3 assets/build_stage2_mjcf.py
"""
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco as mj

HERE = Path(__file__).parent
URDF = HERE / "wam7_bhand.urdf"
OUT = HERE / "wam7_bhand.xml"

with tempfile.NamedTemporaryFile(suffix=".xml") as tmp:
    model = mj.MjModel.from_xml_path(str(URDF))
    mj.mj_saveLastXML(tmp.name, model)
    tree = ET.parse(tmp.name)
root = tree.getroot()
worldbody = root.find("worldbody")

# Ground plane + light, for visual context (Stage 1's demo had neither; Stage 2 renders a
# standing scene so the floor gives a depth reference).
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

# End-effector site: wrist_palm_link + (0,0,0.06) [bhand_base_joint] + (0,0,0.12)
# [bhand_grasp_joint] = (0,0,0.18) in the wrist_palm_link frame -- the BHand's real
# "grasp point" reference frame on the actual robot, just reused here as the IK target site
# since Stage 2 doesn't touch the fingers.
def find_body(name):
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(name)

wrist_palm = find_body("wam/wrist_palm_link")
site = ET.SubElement(wrist_palm, "site")
site.set("name", "wam/ee_site")
site.set("pos", "0 0 0.18")
site.set("size", "0.015")
site.set("rgba", "1 0.2 0.2 1")

# A mocap body for the demo/eval scripts to visualize the current reach target (green sphere,
# moved by setting data.mocap_pos -- no joint, so it never participates in the dynamics).
target_marker = ET.SubElement(worldbody, "body")
target_marker.set("name", "target_marker")
target_marker.set("mocap", "true")
marker_geom = ET.SubElement(target_marker, "geom")
marker_geom.set("type", "sphere")
marker_geom.set("size", "0.02")
marker_geom.set("rgba", "0.1 0.9 0.1 0.6")
marker_geom.set("contype", "0")
marker_geom.set("conaffinity", "0")

# Two actuators per arm joint (Stage 2 is arm-only; the BHand stays passive until a later
# grasping stage):
#  - a <position> servo (kp + kv) for PD feedback. kv is MuJoCo's native derivative gain on
#    the position actuator, integrated implicitly -- safe regardless of a joint's inertia.
#    A hand-rolled "-kd*qvel" torque computed in Python and added as a plain <motor> force is
#    NOT safe: the wrist/palm links have tiny rotational inertia (~1e-4 kg*m^2, since the
#    rigid-body URDF import has no reflected motor/cable inertia the real cable-driven WAM's
#    distal joints actually have), and an explicit damping force at that inertia blows the
#    standard kd*dt/I stability bound -- joint velocity diverged to 15-20 rad/s within a few
#    hundred steps in testing, even though the proportional+gravity-comp terms alone were
#    stable. A plain <position> servo alone (kp only, no feedforward) was tried first too and
#    left several cm of steady-state gravity sag, too much for a 2cm reach tolerance.
#  - a <motor> actuator purely for gravity-compensation feedforward (ctrl set to qfrc_bias
#    each step by control.py) so the position servo's kp only has to correct tracking error,
#    not hold the arm up against gravity.
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
# during a reach (confirmed by direct measurement, not just visibly "jittery") regardless of
# controller tuning. `armature` is MuJoCo's standard mechanism for exactly this: adds a fixed
# rotor-inertia term to the joint's effective inertia for dynamics, independent of the link's
# own rigid-body inertia.
ARMATURE = {
    "wam/wrist_yaw_joint": 0.05,
    "wam/wrist_pitch_joint": 0.05,
    "wam/palm_yaw_joint": 0.05,
}
joint_range = {}
joint_frc = {}
for j in root.iter("joint"):
    if j.get("name") in ARM_JOINTS:
        joint_range[j.get("name")] = j.get("range")
        joint_frc[j.get("name")] = j.get("actuatorfrcrange")
    if j.get("name") in ARMATURE:
        j.set("armature", str(ARMATURE[j.get("name")]))

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

# Disable all arm/hand self-collision. Stage 1's collision meshes are each link's own
# visual (_fine.stl) mesh, not the original convex-decomposition files (MuJoCo can't import
# .dae) -- MuJoCo auto-convexifies those for collision, and the inflated hulls of adjacent
# links permanently overlap at every joint, which locks the whole arm solid under contact
# forces. Harmless for Stage 2 (free-space reaching, no contact task); revisit together with
# the same Stage-1-flagged collision-precision tradeoff if a later grasping stage needs real
# self-collision.
for body in root.iter("body"):
    if not body.get("name", "").startswith("wam/"):
        continue
    for geom in body.findall("geom"):
        if geom.get("contype") != "0":
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

ET.indent(tree, space="  ")
tree.write(OUT)
print("wrote", OUT)
