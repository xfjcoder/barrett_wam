"""Builds assets/wam7_bhand_grasp.xml (Stage 4's MJCF) from assets/wam7_bhand.urdf.

Reuses Stage 2's arm pieces via mjcf_common.py (ee site, arm actuators, armature, self-
collision disabling) and adds what grasping needs on top:
  - Primitive (capsule/box) collision geometry for the hand only, since Stage 1/2's visual-
    mesh collision proxies auto-convexify into permanently-overlapping hulls and were disabled
    wholesale rather than fixed (see mjcf_common.disable_wam_self_collision). Capsules are
    placed from each finger link's own origin to its *child* joint's origin (the physical
    "bone" of that segment, read directly off the URDF's joint tree) rather than guessed from
    the mesh's bounding box -- the mesh AABB's long axis doesn't actually track the segment's
    kinematic axis here (each finger link's knuckle/motor-housing bulk dominates its bounding
    box more than the slender arm does), so AABB-fitting would have picked the wrong axis.
  - Position actuators for the 8 BHand finger joints (no gravity-comp feedforward -- finger
    links are light enough, and grasping's tolerance is coarser than Stage 2's 2cm reach, that
    plain kp+kv suffices).
  - A static table (a bare worldbody geom, not a body -- doesn't need dynamics) and a free-
    floating sphere object to grasp. Sphere, not a box or cylinder, specifically because the
    IK's chosen approach orientation only constrains the palm to face roughly downward (see
    ik.py's target_quat); a sphere looks the same from any rotation about that axis, so the
    grasp doesn't depend on a roll angle nothing in this pipeline controls.
  - A contype/conaffinity split (object+table+floor default to 1/1; hand collision primitives
    use 2/1) that lets the hand contact the object/table/floor while leaving finger-to-finger
    and hand-to-rest-of-arm collision off, for the same lockup reason as Stage 1/2.

Usage (run from the repo root):
    python3 assets/build_stage4_mjcf.py
"""
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

import mjcf_common as common

URDF = common.HERE / "wam7_bhand.urdf"
OUT = common.HERE / "wam7_bhand_grasp.xml"

# Capsule collision geoms: (link name) -> (endpoint in the link's own frame, radius). Endpoint
# is that link's child joint's <origin xyz> straight from the URDF (the next joint down the
# kinematic chain), i.e. finger_1/prox_link's capsule runs from (0,0,0) to finger_1/med_joint's
# origin. finger_3 has no prox_link (its first joint, med, attaches directly to the palm).
FINGER_CAPSULES = {}
for finger, has_prox in [("finger_1", True), ("finger_2", True), ("finger_3", False)]:
    if has_prox:
        FINGER_CAPSULES[f"wam/bhand/{finger}/prox_link"] = ((0.05, 0.0, 0.0339), 0.013)
    FINGER_CAPSULES[f"wam/bhand/{finger}/med_link"] = ((0.06994, 0.003, 0.0), 0.011)
    FINGER_CAPSULES[f"wam/bhand/{finger}/dist_link"] = ((0.05, 0.0, 0.0), 0.010)

def palm_box_from_mesh(urdf_path):
    """bhand_palm_link has only a fixed joint to wrist_palm_link, so MuJoCo's compiler fuses
    it straight into wrist_palm_link -- it never appears as its own body, just as an extra,
    unnamed geom pair (visual+collision) on wrist_palm_link, at whatever pos/quat the fusion
    worked out. Rather than re-derive that transform by hand, load the compiled model here,
    find that geom by its mesh id, and read its pos/quat directly.

    The mesh's own AABB (from mesh_vert) is in the *raw STL vertex frame*, which isn't
    axis-aligned with the physical palm at all -- that's exactly why the compiled geom needed
    a non-trivial corrective quat in the first place. Taking the AABB in that raw frame and
    then rotating the resulting box by that quat produces an oversized, oddly-tilted box (this
    was tried and visually confirmed wrong: it ballooned out past the fingers). Instead,
    transform every vertex INTO wrist_palm_link's frame first (apply the geom's own pos/quat),
    then take the AABB of the already-rotated points -- a tight, axis-aligned (in
    wrist_palm_link's own frame) box, so the <geom> itself needs no quat.
    """
    model = mj.MjModel.from_xml_path(str(urdf_path))
    wrist_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "wam/wrist_palm_link")
    mesh_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MESH, "bhand_palm_fine")
    gi = next(g for g in range(model.ngeom)
              if model.geom_bodyid[g] == wrist_id and model.geom_dataid[g] == mesh_id)
    geom_pos = model.geom_pos[gi].copy()
    geom_quat = model.geom_quat[gi].copy()

    adr, num = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    verts = model.mesh_vert[adr:adr + num]
    rot = np.zeros(9)
    mj.mju_quat2Mat(rot, geom_quat)
    rot = rot.reshape(3, 3)
    verts_in_body = verts @ rot.T + geom_pos

    lo, hi = verts_in_body.min(axis=0), verts_in_body.max(axis=0)
    box_pos = (lo + hi) / 2
    half = (hi - lo) / 2

    # The mesh's own AABB reaches to within ~3mm of where a SPHERE_RADIUS object centered at
    # the ee_site (0.18m out, same z axis) would start -- too tight a margin: the palm ended
    # up pressing directly into the object before the fingers could close around it at all
    # (confirmed by contact force tracing: ~20N palm-object force, ~0 net finger grip). Capping
    # the box's forward (local +z) reach leaves clearance for the fingers to do the gripping.
    margin = 0.01
    max_forward_half = (0.18 - SPHERE_RADIUS - margin) - box_pos[2]
    half[2] = min(half[2], max_forward_half)
    return box_pos, np.array([1.0, 0.0, 0.0, 0.0]), half


HAND_JOINTS = [
    "wam/bhand/finger_1/prox_joint", "wam/bhand/finger_1/med_joint", "wam/bhand/finger_1/dist_joint",
    "wam/bhand/finger_2/prox_joint", "wam/bhand/finger_2/med_joint", "wam/bhand/finger_2/dist_joint",
    "wam/bhand/finger_3/med_joint", "wam/bhand/finger_3/dist_joint",
]

TABLE_TOP_Z = 1.0
TABLE_POS = (0.625, 0.0, TABLE_TOP_Z - 0.02)
TABLE_HALF = (0.3, 0.3, 0.02)
SPHERE_RADIUS = 0.025


#  solref/solimp soften contact stiffness. Default contact parameters, combined with the
# sphere's light mass (tens of grams) and a fingertip that's still moving when it first
# touches, produced repeated 4-5 m/s "pop" velocity spikes (confirmed by direct per-step
# tracing of the object's qvel during the close ramp) -- it would get launched clear across
# the table rather than gripped. Softer contacts spread the same closing force over more time.
HAND_SOLREF = "0.02 1"
HAND_SOLIMP = "0.9 0.95 0.001"


def add_hand_collision(root):
    for link_name, (endpoint, radius) in FINGER_CAPSULES.items():
        body = common.find_body(root, link_name)
        cap = ET.SubElement(body, "geom")
        cap.set("name", link_name.replace("/", "_") + "_col")
        cap.set("type", "capsule")
        cap.set("fromto", f"0 0 0 {endpoint[0]} {endpoint[1]} {endpoint[2]}")
        cap.set("size", str(radius))
        cap.set("contype", "2")
        cap.set("conaffinity", "1")
        cap.set("condim", "6")
        cap.set("friction", "1.2 0.01 0.02")
        cap.set("solref", HAND_SOLREF)
        cap.set("solimp", HAND_SOLIMP)
        cap.set("rgba", "0.2 0.2 0.2 1")

    wrist_palm = common.find_body(root, "wam/wrist_palm_link")
    box_pos, box_quat, box_half = palm_box_from_mesh(URDF)
    box = ET.SubElement(wrist_palm, "geom")
    box.set("name", "wam_palm_col")
    box.set("type", "box")
    box.set("pos", " ".join(f"{c:.6f}" for c in box_pos))
    box.set("quat", " ".join(f"{c:.6f}" for c in box_quat))
    box.set("size", " ".join(f"{h:.6f}" for h in box_half))
    box.set("contype", "2")
    box.set("conaffinity", "1")
    box.set("condim", "6")
    box.set("friction", "1.2 0.01 0.02")
    box.set("solref", HAND_SOLREF)
    box.set("solimp", HAND_SOLIMP)
    box.set("rgba", "0.2 0.2 0.2 1")


# Light finger links (tens of grams) plus a fast position servo made first contact with the
# object violently stiff (see add_hand_collision's note) -- armature adds fixed rotor inertia
# to each hand joint, same fix and same reasoning as Stage 2's wrist/palm joints.
HAND_ARMATURE = 0.01


def add_hand_actuators(root):
    joint_range = {}
    joint_frc = {}
    for j in root.iter("joint"):
        if j.get("name") in HAND_JOINTS:
            joint_range[j.get("name")] = j.get("range")
            joint_frc[j.get("name")] = j.get("actuatorfrcrange")
            j.set("armature", str(HAND_ARMATURE))

    actuator = root.find("actuator")
    for name in HAND_JOINTS:
        pos = ET.SubElement(actuator, "position")
        pos.set("name", name.replace("joint", "pos_act"))
        pos.set("joint", name)
        pos.set("kp", "1.5")
        pos.set("kv", "0.5")
        pos.set("ctrlrange", joint_range[name])
        pos.set("forcerange", joint_frc[name])


def add_table_and_object(worldbody):
    table = ET.SubElement(worldbody, "geom")
    table.set("name", "table")
    table.set("type", "box")
    table.set("pos", " ".join(str(c) for c in TABLE_POS))
    table.set("size", " ".join(str(h) for h in TABLE_HALF))
    # condim=6 + a (deliberately exaggerated) rolling-friction coefficient: a frictionless-
    # rolling rigid sphere has no mechanism to ever stop once nudged (condim=3/4 only damp
    # sliding/spin, not rolling) -- confirmed directly: a bare ~9cm sideways nudge during the
    # close ramp set the ball rolling, and it kept coasting across the table for the rest of
    # the episode, well outside grasp range, with no further contact forcing it. Real rolling
    # friction for a hard ball is much smaller than this; this value is picked for the episode
    # to settle in a few seconds, not for physical accuracy.
    table.set("condim", "6")
    table.set("friction", "1.2 0.01 0.02")
    table.set("rgba", "0.55 0.4 0.3 1")

    obj = ET.SubElement(worldbody, "body")
    obj.set("name", "grasp_object")
    obj.set("pos", f"{TABLE_POS[0]} {TABLE_POS[1]} {TABLE_TOP_Z + SPHERE_RADIUS}")
    free = ET.SubElement(obj, "freejoint")
    free.set("name", "grasp_object_joint")
    geom = ET.SubElement(obj, "geom")
    geom.set("name", "grasp_object_geom")
    geom.set("type", "sphere")
    geom.set("size", str(SPHERE_RADIUS))
    geom.set("density", "600")
    geom.set("condim", "6")
    geom.set("friction", "1.2 0.01 0.02")
    geom.set("solref", HAND_SOLREF)
    geom.set("solimp", HAND_SOLIMP)
    geom.set("rgba", "0.9 0.4 0.1 1")


tree = common.compile_urdf(URDF)
root = tree.getroot()
worldbody = root.find("worldbody")

common.add_floor_and_light(worldbody)
common.add_ee_site(root)
common.add_arm_actuators(root)
common.disable_wam_self_collision(root)
add_hand_collision(root)
add_hand_actuators(root)
add_table_and_object(worldbody)

ET.indent(tree, space="  ")
tree.write(OUT)
print("wrote", OUT)
