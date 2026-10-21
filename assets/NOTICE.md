`wam7_bhand.urdf` and the meshes in `meshes/` are derived from
[`jhu-lcsr/barrett_model`](https://github.com/jhu-lcsr/barrett_model) (maintainer: Jonathan
Bohren; original URDF: RE2), licensed GPL. Only `robots/wam7_bhand.urdf.xacro` (expanded to a
static URDF via `xacro`) and the `_fine.stl` meshes it references are vendored here, with three
modifications documented in the top-level README's Stage 1 section:

1. `$(find barrett_model)` xacro package references substituted with literal paths before
   expansion.
2. Every `<collision>` mesh (originally `.dae` convex-decomposition files, which MuJoCo's URDF
   importer cannot parse) replaced with that link's own `_fine.stl` visual mesh.
3. `wam/wrist_yaw_link`'s degenerate inertia tensor (one negative eigenvalue, an upstream data
   error) fixed by zeroing its off-diagonal terms.

`wam7_bhand.xml` (Stage 2) is a further-derived artifact: the above URDF compiled to MJCF via
`mj.mj_saveLastXML`, then hand-edited to add position+motor actuators on the 7 arm joints, an
end-effector site, a target-marker mocap body, `armature` on the 3 distal joints, and disabled
self-collision -- see the top-level README's Stage 2 section for why each was needed.

`wam7_bhand_grasp.xml` (Stage 4) is a sibling artifact built the same way (via
`assets/build_stage4_mjcf.py`, sharing `assets/mjcf_common.py`'s helpers with Stage 2's build
script), adding primitive hand collision geometry, 8 BHand finger actuators, a static table, and
a free-floating sphere to grasp -- see the top-level README's Stage 4 section.
