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
