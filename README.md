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

2. **Reach a target -- DONE.** Turns out the "fixed-base mount" item flagged as needed above
   was already a non-issue: the URDF's root joint (`wam/wam_fixed_joint`) is `type="fixed"`,
   not floating, so MuJoCo's URDF importer welds `wam/base_link` straight to the world with no
   free joint at all (confirmed: `nq == nv == 15`, matching exactly the 15 revolute joints,
   no extra 6/7 DOF from a floating base). The actual new work:

   - **Compiled the URDF to a static MJCF** (`assets/wam7_bhand.xml`, via
     `mj.mj_saveLastXML`) so actuators, sites, and a target-marker mocap body could be added
     -- MuJoCo's URDF importer has no mechanism for any of those, and doesn't auto-generate
     actuators from URDF `<transmission>` tags (`model.nu == 0` on a raw URDF import). The
     original `assets/wam7_bhand.urdf` stays as Stage 1's artifact; `wam7_bhand.xml` is now
     the model every script after Stage 1 loads.
   - **Two actuators per arm joint** (7 arm joints only -- the BHand stays passive until a
     later grasping stage): a `<position>` servo (`kp`+`kv`, MuJoCo's own implicit PD) for
     feedback, plus a `<motor>` actuator fed `data.qfrc_bias` each step as pure gravity/
     Coriolis-compensation feedforward (`control.py`), so the servo's `kp` only has to correct
     tracking error, not hold the arm up against gravity. A plain position servo alone left
     several cm of steady-state sag; a hand-rolled explicit `-kd*qvel` damping torque (tried
     before settling on the implicit `kv`) **diverged** -- the wrist/palm links' own
     rigid-body inertia is tiny (~1e-4 kg*m^2, since the rigid URDF import has no reflected
     motor/cable inertia the real cable-driven WAM's distal joints actually have), and that
     breaks the standard explicit-damping stability bound; measured joint velocity exploding
     to 15-20 rad/s confirmed it wasn't just "a bit jittery."
   - **`armature` on the three distal joints** (wrist yaw/pitch, palm yaw: 0.05 each) --
     the same missing-reflected-inertia gap also let Coriolis coupling from the much heavier,
     fast-moving shoulder/elbow links fling these joints to the same 15-20 rad/s during a
     reach, independent of controller tuning. `armature` is MuJoCo's standard fix: it adds a
     fixed rotor-inertia term to a joint's effective dynamics, separate from the link's own
     inertia. Fixed it completely (confirmed: qvel on all 7 joints was indistinguishable from
     0 at convergence afterward, not just qualitatively calmer).
   - **Self-collision disabled for every arm/hand geom** -- Stage 1's collision meshes are
     each link's own visual (`_fine.stl`) mesh rather than the original convex-decomposition
     files, and MuJoCo auto-convexifies those for collision; the inflated hulls of adjacent
     links turned out to permanently overlap at every joint, which locked the whole arm solid
     under contact forces the moment any joint was driven (`qfrc_actuator` pinned at its
     ±60 N*m limit while `qpos` didn't move at all). Harmless for a free-space reaching task;
     revisit alongside Stage 1's same flagged collision-precision tradeoff if a later grasping
     stage needs real self-collision.
   - **Damped-least-squares Jacobian IK** (`ik.py`) for the end-effector site (`wam/ee_site`,
     placed at the BHand's real grasp-point offset, 0.18m along the palm's z-axis from
     `wam/wrist_palm_link`) -- numerical rather than closed-form, since closed-form IK for
     this specific 7-DOF geometry would need its own derivation and damped-least-squares
     handles the redundancy (4 extra DOF beyond the 3 needed for a position-only target)
     without one.

   **Evaluated over 20 random targets** (`eval_stage2_reach_target.py`), not a single run:
   each target is a random *valid joint config* forward-kinematics'd to a Cartesian point
   (guaranteed reachable, unlike sampling arbitrary Cartesian points that might not be), IK
   solves it from the arm's current pose, then the controller drives there under gravity for a
   fixed duration. **17/20 reached within 2cm.** The 3 misses are exactly the 3 trials where
   IK itself failed to converge within 200 iterations (not a tracking failure) -- every trial
   where IK converged, closed-loop control reached the target. IK's ~85% convergence rate from
   a fixed home start is a known limitation of local damped-least-squares with this much
   redundancy, not investigated further at this stage (a future stage could retry from a
   randomized seed config on failure, or switch to a global/sampling-based IK).

   <p align="center">
     <img src="media/stage2_reach_target.gif" width="480" alt="WAM arm reaching four sequential targets (green sphere), the end-effector site (red dot) converging each time">
   </p>

   *Stage 2's demo: the arm reaching 4 sequential random targets in one continuous episode
   (`demo_stage2_reach_target.py`), each converging to within 6mm.*

3. **Track a moving target -- DONE.** Same IK (`ik.py`) and controller (`control.py`) as
   Stage 2, but the target now moves (a horizontal circle, `trajectory.py`) and IK is
   warm-started from the arm's current pose each step with few iterations and a loose
   tolerance -- a continuous resolved-rate tracking loop rather than a one-shot solve.

   **Evaluated over 10 random circle centers** (`eval_stage3_moving_target.py`, each center
   sampled the same guaranteed-reachable way as Stage 2's targets; radius 0.15m, omega
   0.5 rad/s; first 2s of settle-from-home excluded from the error stats): **7/10 tracked**
   (steady-state mean error < 5cm). Two of the three misses (seeds 1 and 9) are the *same*
   centers Stage 2's static eval already flagged as IK non-convergence failures -- a
   consistent cross-validation of that known limitation, not a new bug. The third (seed 5)
   is new and more interesting: that center alone *was* reachable in Stage 2's static eval,
   but circling around it sweeps through nearby configurations IK can't always solve --
   tracking a region can fail even when reaching its center point succeeds.

   <p align="center">
     <img src="media/stage3_moving_target.gif" width="480" alt="WAM arm tracking a moving target (green sphere) with the end-effector site (red dot) following at a small visible lag">
   </p>

   *Stage 3's demo: the arm tracking \~1.25 circles of a moving target
   (`demo_stage3_moving_target.py`), red end-effector site visibly lagging the green target
   by a few cm -- the same lag `eval_stage3_moving_target.py` measures, not hidden by camera
   distance.*

## Repository layout

```
barrett_wam/
├── assets/
│   ├── wam7_bhand.urdf        # Stage 1: vendored + fixed, see writeup above
│   ├── wam7_bhand.xml         # Stage 2: compiled MJCF + actuators/site/marker, see writeup
│   └── meshes/                # 13 STL meshes the URDF references (bhand/ subdir for the hand)
├── ik.py                            # Stage 2: damped-least-squares Jacobian IK
├── control.py                       # Stage 2: PD + gravity-compensation controller
├── trajectory.py                    # Stage 3: moving (circular) target generator
├── eval_stage2_reach_target.py      # Stage 2: 20-trial random-target eval
├── eval_stage3_moving_target.py     # Stage 3: 10-trial moving-target tracking eval
├── demo_stage1_kinematic_sweep.py   # Stage 1: verification sweep, see media/ GIF above
├── demo_stage2_reach_target.py      # Stage 2: 4-target reach demo, see media/ GIF above
├── demo_stage3_moving_target.py     # Stage 3: moving-target tracking demo, see media/ GIF
├── media/
│   ├── stage1_kinematic_sweep.gif
│   ├── stage2_reach_target.gif
│   └── stage3_moving_target.gif
├── requirements.txt
└── README.md
```
