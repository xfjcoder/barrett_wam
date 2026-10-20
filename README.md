# Barrett WAM Manipulation

A manipulation project for the Barrett WAM (Whole Arm Manipulator): a 7-DOF arm plus the
BarrettHand, simulated in MuJoCo. A separate project (and repo) from `go1_rl_walk`, `slam_sim`,
and `wam` (sibling directories), following the same convention: each new kind of problem gets
its own repo, not a branch of an existing one.

<p align="center">
  <img src="media/stage1_kinematic_sweep.gif" width="480" alt="Barrett WAM 7-DOF arm + BHand, each joint swept through its own range one at a time, then the hand closing and opening">
</p>

*Stage 1's verification sweep: each of the 7 arm joints swept through its own range in turn,
then the three BHand fingers closing and opening together.*

**Note on the name**: this project is named after the actual Barrett WAM arm (the robot this
repo simulates), which is unrelated to the sibling `wam` repo's "World Action Model" (a learned
dynamics model for the Unitree Go2 quadruped) -- a naming collision flagged and resolved before
starting this project. They share no code.

## Scoping decisions

- **Platform: the real Barrett WAM 7-DOF arm + BarrettHand**, not a more commonly-available
  research arm (Franka Panda, UR5e) or a simpler 6-DOF industrial arm -- chosen for the more
  dexterous 3-finger underactuated hand and because no ready-made MuJoCo model existed for it in
  this environment (`mujoco_menagerie`, vendored locally at `~/robo/mujoco_menagerie`, has Franka/
  UR5/Kinova/Shadow Hand/Allegro Hand models but no Barrett WAM), making this a genuine build
  rather than importing an already-solved asset.
- **First task: reach a target**, not grasping or pick-and-place -- the simplest closed-loop
  control problem (drive the end-effector to a commanded pose, no contact/grasp physics to get
  right yet), matching this project's own established staged-build discipline (verify the
  simplest real capability before adding the next one).
- **Simulator: MuJoCo**, reusing the same physics engine every sibling project already uses, via
  `go1_rl_walk/mpc`'s venv (same `mujoco>=3.1.0` dependency, no new venv needed).

## Staged plan

1. **Simulator core: vendor and fix a real Barrett WAM + BHand MuJoCo model -- DONE.**
   No MuJoCo-ready model existed for this robot anywhere in this environment, so this stage is
   genuinely about getting one to load and move correctly, not just importing a solved asset.

   Source: [`jhu-lcsr/barrett_model`](https://github.com/jhu-lcsr/barrett_model) (the standard
   ROS description package for this robot), specifically `robots/wam7_bhand.urdf.xacro` (7-DOF
   arm + BHand). Three real problems hit and fixed along the way, not glossed over:
   - **Xacro `$(find barrett_model)` package references don't resolve outside a real ROS
     package.** Worked around by substituting the literal clone path via `sed` before running
     `xacro` (ROS 2 Jazzy's `xacro` CLI, already installed in this environment) to expand the
     macros into a plain, static URDF.
   - **MuJoCo's URDF importer can't parse `.dae` collision meshes** (`no decoder found for mesh
     file ...`) -- every `<collision>` mesh reference used Collada (`.dae`) convex-decomposition
     files. Fixed by substituting each one with its link's own `_fine.stl` visual mesh (already
     present on disk for every link) as the collision geometry too -- a deliberate simplification
     (losing the original convex-decomposition's collision precision) accepted for this stage,
     since the goal here is "loads and moves correctly," not production-grade contact dynamics;
     revisit if a later grasping stage needs tighter finger-pad collision geometry.
   - **One link's inertia tensor is degenerate** (`wam/wrist_yaw_link`, `ixz` in the upstream
     URDF is suspiciously identical to `iyy` -- almost certainly a copy-paste error in the
     original file, not a modeling choice) -- MuJoCo's compiler correctly rejects it
     (`inertia must have positive eigenvalues`). Found by computing eigenvalues of each link's
     3x3 inertia matrix directly (one negative eigenvalue on this link, all 22 others fine) and
     fixed by zeroing that link's off-diagonal terms, keeping its diagonal (clearly sane) values.
   - Vendored only what's actually used: the final URDF (`assets/wam7_bhand.urdf`) plus the 13
     `_fine.stl` meshes it references (5.7MB total, `assets/meshes/`) -- not the full 133MB
     upstream repo (which also carries `.blend` source files and unused collision-decomposition
     variants).

   **Verified, not just "it didn't crash"**: loads with 15 joints (7 arm + 8 BHand, matching the
   real BHand's underactuated kinematics -- fingers 1/2 each get a spread+flex+distal joint,
   finger 3 has no spread since it's fixed relative to the palm on the real hardware), joint
   limits match published WAM/BHand specs (e.g. elbow 0-180°, wrist pitch ±90°). Visually
   confirmed via the demo GIF above and a direct elbow 0°-vs-180° comparison render -- the
   forearm genuinely folds back alongside the upper arm, not just a qpos number changing with no
   visible effect.

2. **(Next) Reach a target** -- not started. Will need: a fixed-base mount (the URDF's world
   joint currently floats freely with no attachment, fine for this stage's kinematic-only demo
   but not for a real controller), actuators (the raw URDF import has none, `model.nu == 0` --
   MuJoCo doesn't auto-generate them from URDF `<transmission>` tags), and a controller (position/
   velocity PD or inverse-kinematics-based) to drive the end-effector toward a commanded pose.

## Repository layout

```
barrett_wam/
├── assets/
│   ├── wam7_bhand.urdf        # vendored + fixed (see Stage 1 writeup above)
│   └── meshes/                # 13 STL meshes the URDF references (bhand/ subdir for the hand)
├── demo_stage1_kinematic_sweep.py   # Stage 1: verification sweep, see media/ GIF above
├── media/
│   └── stage1_kinematic_sweep.gif
├── requirements.txt
└── README.md
```
