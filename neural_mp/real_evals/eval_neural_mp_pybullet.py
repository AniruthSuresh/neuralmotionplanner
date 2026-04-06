"""
eval_neural_mp_pybullet.py
===========================
Runs the REAL NeuralMP checkpoint (mihdalal/NeuralMP) in a PyBullet
Franka Panda simulation.  No real robot / Manimo / cameras needed.

─── RUN ───
 
  python neural_mp/real_evals/eval_neural_mp_pybullet.py
  python neural_mp/real_evals/eval_neural_mp_pybullet.py --tto
  python neural_mp/real_evals/eval_neural_mp_pybullet.py --scene 1 --tto
  python neural_mp/real_evals/eval_neural_mp_pybullet.py --no-gui --runs 5


"""
import argparse
import builtins
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

# ─────────────────────────────────────────────────────────────────────────────
# Patch out all input() prompts that NeuralMP uses for real-robot safety
# ─────────────────────────────────────────────────────────────────────────────
_real_input = builtins.input


def _auto_input(prompt=""):
    print(f"[auto-confirm] {prompt}")
    return "y"


builtins.input = _auto_input

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
try:
    from neural_mp.real_utils.neural_motion_planner import NeuralMP
except ImportError:
    sys.exit(
        "\n[ERROR] Cannot import NeuralMP.\n"
        "Make sure you are running from inside the neuralmotionplanner/ directory\n"
        "and have installed all dependencies. See the docstring above.\n"
    )
from neural_mp.envs.pybullet_franke_env import PybulletFrankaEnv


# ─────────────────────────────────────────────────────────────────────────────
# Scene builder
# ─────────────────────────────────────────────────────────────────────────────
def build_scene(env: PybulletFrankaEnv, scene_id: int):
    env.clear_obstacles()
    if scene_id == 0:
        env.add_obstacle_box([0.04, 0.15, 0.10], [0.50, 0.10, 0.73], color=(0.7, 0.3, 0.1, 1))
        env.add_obstacle_box([0.04, 0.15, 0.10], [0.50, -0.10, 0.73], color=(0.1, 0.5, 0.7, 1))
    elif scene_id == 1:
        shelf_x = 0.52
        base_z = 0.625
        env.add_obstacle_box(
            [0.02, 0.25, 0.30], [shelf_x + 0.23, 0.0, base_z + 0.30], color=(0.5, 0.5, 0.5, 1)
        )
        for sy in [-0.25, 0.25]:
            env.add_obstacle_box(
                [0.25, 0.02, 0.30], [shelf_x, sy, base_z + 0.30], color=(0.5, 0.5, 0.5, 1)
            )
        for rz in [0.15, 0.30, 0.45]:
            env.add_obstacle_box(
                [0.25, 0.24, 0.01], [shelf_x, 0.0, base_z + rz], color=(0.6, 0.4, 0.2, 1)
            )
    elif scene_id == 2:
        import random

        random.seed(7)
        for _ in range(6):
            x = random.uniform(0.35, 0.65)
            y = random.uniform(-0.25, 0.25)
            r = random.uniform(0.03, 0.06)
            env.add_obstacle_sphere(r, [x, y, 0.625 + r], color=(0.2, 0.8, 0.3, 1))
        env.add_obstacle_box(
            [0.06, 0.06, 0.12], [0.48, 0.0, 0.625 + 0.12], color=(0.8, 0.2, 0.2, 1)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Neural MP real checkpoint in PyBullet simulation")
    parser.add_argument("--model-url", default="mihdalal/NeuralMP")
    parser.add_argument(
        "--tto", action="store_true", help="Test-time optimisation (100 rollouts, pick best)"
    )
    parser.add_argument(
        "--train-mode", action="store_true", help="Policy in train mode (stochastic dropout)"
    )
    parser.add_argument("--no-gui", action="store_true", help="Headless — no PyBullet window")
    parser.add_argument(
        "--scene",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="0=two boxes  1=shelf  2=cluttered table",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of planning+execution trials")
    parser.add_argument("--speed", type=float, default=0.15, help="Execution speed (rad/s)")

    parser.add_argument(
        "--start", nargs=7, type=float, default=[-0.538, 0.628, -0.061, -1.750, 0.126, 2.418, 1.610]
    )
    parser.add_argument(
        "--goal", nargs=7, type=float, default=[1.067, 0.847, -0.591, -1.627, 0.623, 2.295, 2.580]
    )
    args = parser.parse_args()
    start_config = np.array(args.start)
    goal_config = np.array(args.goal)

    print("\n[Setup] Connecting to PyBullet...")
    env = PybulletFrankaEnv(gui=not args.no_gui)

    print(f"[Setup] Building scene {args.scene}...")
    build_scene(env, args.scene)

    print(f"\n[Setup] Loading NeuralMP from '{args.model_url}' ...")
    neural_mp = NeuralMP(
        env=env,
        model_url=args.model_url,
        train_mode=args.train_mode,
        in_hand=False,
        in_hand_params=[0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
        visualize=False,
    )

    # =========================================================================
    # Adjust configurations using IK for exact XYZ placement
    # =========================================================================
    def get_ik_for_target_pos(config, target_xyz):
        env.move_robot_to_joint_state(config, time_to_go=0.0)
        p.stepSimulation()

        robot_id = 0
        for i in range(p.getNumBodies()):
            if p.getNumJoints(i) >= 7:
                robot_id = i
                break

        ee_link = p.getNumJoints(robot_id) - 1
        for i in range(p.getNumJoints(robot_id)):
            name = p.getJointInfo(robot_id, i)[12].decode("utf-8").lower()
            if "hand" in name or "tcp" in name or "grasp" in name:
                ee_link = i
                break

        state = p.getLinkState(robot_id, ee_link)
        quat = state[5]

        new_joints = p.calculateInverseKinematics(
            bodyUniqueId=robot_id,
            endEffectorLinkIndex=ee_link,
            targetPosition=target_xyz,
            targetOrientation=quat,
            restPoses=list(config),
        )
        return np.array(new_joints[:7]), target_xyz

    print("[Adjust] Modifying start/goal configs to flank obstacles above table...")
    table_z = 0.625

    # BASIC WORKING SETUP => success
    # target_z = table_z + 0.20
    # target_start_pos = [0.50, -0.45, target_z]
    # target_goal_pos = [0.30, -0.10, target_z]

    # Flanking front-to-back => success
    target_z = table_z + 0.1
    target_start_pos = [0.30, 0.0, target_z]
    target_goal_pos = [0.7, 0.0, target_z]

    start_config, start_pos = get_ik_for_target_pos(start_config, target_start_pos)
    goal_config, goal_pos = get_ik_for_target_pos(goal_config, target_goal_pos)

    print(f"\n[Targeting] Calculated Joint Configs:")
    print(f"  Expected Start: {np.round(start_config, 4)}")
    print(f"  Expected Goal : {np.round(goal_config, 4)}")

    print("\n[Action] Moving to start config...")
    env.move_robot_to_joint_state(start_config, time_to_go=0.0)

    # NEW: Verify the reach before planning
    actual_start_joints = env.get_joint_angles()
    start_reach_err = np.linalg.norm(actual_start_joints - start_config)

    print(f"[Verify] Reached Start Position.")
    print(f"  Actual Joints: {np.round(actual_start_joints, 4)}")
    print(f"  Reach Error  : {start_reach_err:.6f} rad")

    if start_reach_err > 0.05:
        print("  [WARN] Large reach error! Robot might be colliding with the table.")
    # =========================================================================
    # 4. Capture simulated point cloud
    # =========================================================================
    print("[PCD] Capturing scene point cloud from virtual cameras...")
    points, colors = neural_mp.get_scene_pcd()
    print(f"      → {len(points):,} obstacle points captured.\n")

    # --- NEW: Save using Matplotlib and Numpy ---
    if len(points) > 0:
        print("[PCD] Saving Point Cloud visualization and raw arrays...")

        # 1. Dynamically find the neural_mp/test/ directory based on this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets neural_mp/real_evals/
        neural_mp_dir = os.path.dirname(script_dir)  # Gets neural_mp/
        save_dir = os.path.join(neural_mp_dir, "test")  # Gets neural_mp/test/

        os.makedirs(save_dir, exist_ok=True)

        # 2. Save visual plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Subsample the points if there are too many (matplotlib gets slow)
        step = max(1, len(points) // 10000)

        ax.scatter(
            points[::step, 0],
            points[::step, 1],
            points[::step, 2],
            c=colors[::step],
            s=2,
            alpha=0.8,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Captured Scene Point Cloud")

        # USE os.path.join to save inside the directory safely
        plt.savefig(os.path.join(save_dir, "pointcloud_plot.png"), dpi=150)
        plt.close()

        print(f"      → Saved visualization to '{save_dir}/pointcloud_plot.png'.")

        # 3. Save raw data
        np.save(os.path.join(save_dir, "pointcloud_points.npy"), points)
        np.save(os.path.join(save_dir, "pointcloud_colors.npy"), colors)

        print(f"      → Saved raw arrays to '{save_dir}/' directory.\n")
    # ---------------------------------------------
    # =========================================================================
    # Visuals: Spawn visual balls AFTER PCD capture
    # =========================================================================
    print("[Visuals] Spawning visual markers for Start (Green) and Goal (Red)...")

    def spawn_ball(pos, color):
        vid = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=color)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vid, basePosition=pos)

    spawn_ball(start_pos, [0, 1, 0, 1])
    spawn_ball(goal_pos, [1, 0, 0, 1])
    # =========================================================================

    if len(points) < 10:
        print("[WARN] Very few obstacle points captured.")

    # 5. Planning + execution loop
    results = []
    for run_i in range(args.runs):
        print(f"{'─'*60}")
        print(f"Run {run_i + 1}/{args.runs}")
        print(f"  Start: {np.round(start_config, 3)}")
        print(f"  Goal : {np.round(goal_config,  3)}")

        print("  → Moving to start config...")
        env.move_robot_to_joint_state(start_config, time_to_go=3.0)

        print(f"  → Planning (TTO={args.tto})...")
        if args.tto:
            traj, planning_ok, t = neural_mp.motion_plan_with_tto(
                start_config=start_config,
                goal_config=goal_config,
                points=points,
                colors=colors,
            )
        else:
            traj, planning_ok, t = neural_mp.motion_plan(
                start_config=start_config,
                goal_config=goal_config,
                points=points,
                colors=colors,
            )

        if traj is None or len(traj) == 0:
            print("  ✗  No trajectory returned.")
            results.append(dict(success=False, joint_err=None, planning_ok=False))
            continue

        print(f"  → Trajectory: {len(traj)} waypoints | " f"planning_ok={planning_ok}")

        print("  → Executing...")
        success, joint_err = neural_mp.execute_motion_plan(traj, speed=args.speed)
        status = "✓  SUCCESS" if success else "✗  FAILED "
        print(f"  {status} | joint_err={joint_err:.4f} rad")
        results.append(dict(success=success, joint_err=joint_err, planning_ok=planning_ok))

    # 6. Summary
    print(f"\n{'═'*60}")
    n_ok = sum(r["success"] for r in results)
    print(f"RESULTS: {n_ok}/{args.runs} successful " f"({100*n_ok/args.runs:.0f}%)")
    errs = [r["joint_err"] for r in results if r["joint_err"] is not None]
    if errs:
        print(f"         Mean joint error: {np.mean(errs):.4f} rad")

    if not args.no_gui:
        builtins.input = _real_input
        input("\n[Press Enter to close PyBullet]")
    env.close()


if __name__ == "__main__":
    main()
