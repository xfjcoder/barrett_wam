"""Builds assets/wam7_bhand.xml (Stage 2's MJCF) from assets/wam7_bhand.urdf (Stage 1's).

MuJoCo's URDF importer has no mechanism to add actuators, sites, or mocap bodies -- so this
compiles the URDF once (mj.mj_saveLastXML) and hand-edits the resulting MJCF to add them. See
the top-level README's Stage 2 section for why each addition below was needed. The shared
pieces (arm actuators, ee site, self-collision disabling) live in mjcf_common.py, reused by
build_stage4_mjcf.py for the grasping scene.

Usage (run from the repo root):
    python3 assets/build_stage2_mjcf.py
"""
import xml.etree.ElementTree as ET

import mjcf_common as common

URDF = common.HERE / "wam7_bhand.urdf"
OUT = common.HERE / "wam7_bhand.xml"

tree = common.compile_urdf(URDF)
root = tree.getroot()
worldbody = root.find("worldbody")

common.add_floor_and_light(worldbody)
common.add_ee_site(root)
common.add_target_marker(worldbody)
common.add_arm_actuators(root)
common.disable_wam_self_collision(root)

ET.indent(tree, space="  ")
tree.write(OUT)
print("wrote", OUT)
