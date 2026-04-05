"""
pybullet_franka_env.py  (v2 — complete)
========================================
Drop-in PyBullet replacement for FrankaRealEnvManimo.

Satisfies EVERY method that NeuralMP / neural_motion_planner.py calls:

    env.collision_checker.set_cuboid_params(...)            ← __init__
    env.collision_checker.check_scene_collision_batch(...)  ← motion_plan_with_tto
    env.get_gripper_width()
    env.move_robot_to_joint_state(joint_state, time_to_go)
    env.get_scene_pcd(...)                                  ← get_scene_pcd wrapper
    env.execute_plan(plan, init_joint_angles, speed)        ← execute_motion_plan
    env.canonical_joint_pose                                ← reset

Place this file at:
    neuralmotionplanner/neural_mp/envs/pybullet_franka_env.py

Then import with:
    from neural_mp.envs.pybullet_franka_env import PybulletFrankaEnv
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data
import torch

# ── Franka constants (mirror neural_mp/utils/constants.py) ───────────────────
FRANKA_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
FRANKA_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
FRANKA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
ARM_JOINTS = list(range(7))
CTRL_HZ = 60

# Franka sphere approximation for collision checking: (link_idx, local_xyz, radius)
FRANKA_SPHERES = [
    (0, [0.0, 0.0, 0.05], 0.08),
    (1, [0.0, 0.0, 0.0], 0.07),
    (2, [0.0, -0.08, 0.0], 0.07),
    (2, [0.0, 0.0, -0.12], 0.06),
    (3, [0.0, 0.0, 0.0], 0.07),
    (3, [0.07, 0.0, 0.0], 0.06),
    (4, [0.0, 0.0, 0.0], 0.07),
    (4, [0.0, 0.08, 0.0], 0.06),
    (5, [0.0, 0.0, 0.0], 0.06),
    (6, [0.0, 0.0, 0.0], 0.06),
    (6, [0.06, 0.0, 0.0], 0.05),
    (7, [0.0, 0.0, 0.0], 0.05),
    (7, [0.0, 0.0, 0.10], 0.04),
]


# ─────────────────────────────────────────────────────────────────────────────
# Stub collision checker — satisfies FrankaCollisionChecker interface
# ─────────────────────────────────────────────────────────────────────────────


class SimCollisionChecker:
    """
    Provides the two methods NeuralMP calls on env.collision_checker:
      - set_cuboid_params(sizes, centers, oris)          [NeuralMP.__init__]
      - check_scene_collision_batch(configs, scene_pcd,
                                    thred, sphere_repr_only)  [motion_plan_with_tto]
    """

    def __init__(self, env: "PybulletFrankaEnv"):
        self._env = env

    def set_cuboid_params(self, sizes: List, centers: List, oris: List):
        """Store in-hand object params (used only when in_hand=True)."""
        # no-op for in_hand=False, but must exist
        pass

    def check_scene_collision_batch(
        self,
        configs: torch.Tensor,
        scene_pcd_batch: torch.Tensor,
        thred: float = 0.01,
        sphere_repr_only: bool = True,
    ) -> torch.Tensor:
        """
        Sphere-vs-pointcloud collision count for each config in the batch.
        Called by motion_plan_with_tto to score trajectories.

        configs:          (N, 7)   joint angles
        scene_pcd_batch:  (N, K, 3) obstacle points per config
        Returns:          (N,)     float tensor of collision counts
        """
        N = configs.shape[0]
        counts = torch.zeros(N, dtype=torch.float32)
        configs_np = configs.detach().cpu().numpy()
        scene_np = scene_pcd_batch.detach().cpu().numpy()  # (N, K, 3)

        for i in range(N):
            spheres = self._env._get_sphere_centers_world(configs_np[i])
            pts = scene_np[i]  # (K, 3)
            for center, radius in spheres:
                dists = np.linalg.norm(pts - center, axis=1)
                counts[i] += float(np.sum(dists < radius + thred))

        return counts


# ─────────────────────────────────────────────────────────────────────────────
# Main environment
# ─────────────────────────────────────────────────────────────────────────────


class PybulletFrankaEnv:
    """
    PyBullet Franka Panda — complete drop-in for FrankaRealEnvManimo.

    Every method name and signature matches the real env so NeuralMP
    works unchanged.
    """

    def __init__(self, gui: bool = True):
        self.gui = gui
        self.ctrl_hz = CTRL_HZ

        # NeuralMP.get_scene_pcd() calls env.reset() which uses this
        self.canonical_joint_pose = FRANKA_HOME.copy()

        self._client: int = -1
        self._robot: int = -1
        self._obstacles: List[int] = []
        self._gripper_width: float = 0.08  # fully open

        # ← collision_checker is what NeuralMP.__init__ immediately accesses
        self.collision_checker = SimCollisionChecker(self)

        self._connect()
        self._load_scene()

    # ── setup ─────────────────────────────────────────────────────────────────

    def _connect(self):
        mode = p.GUI if self.gui else p.DIRECT
        self._client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.8,
                cameraYaw=50,
                cameraPitch=-25,
                cameraTargetPosition=[0.4, 0.0, 0.4],
                physicsClientId=self._client,
            )
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self._client)

    def _load_scene(self):
        p.loadURDF("plane.urdf", physicsClientId=self._client)
        p.loadURDF(
            "table/table.urdf",
            basePosition=[0.5, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 1.5708]),
            physicsClientId=self._client,
        )
        panda_urdf = os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf")
        self._robot = p.loadURDF(
            panda_urdf, basePosition=[0, 0, 0.625], useFixedBase=True, physicsClientId=self._client
        )
        self._set_joints_instant(FRANKA_HOME)
        for _ in range(50):
            p.stepSimulation(physicsClientId=self._client)

    # ── FrankaRealEnv public interface ────────────────────────────────────────

    def get_joint_angles(self) -> np.ndarray:
        """7-dof joint angles."""
        states = p.getJointStates(self._robot, ARM_JOINTS, physicsClientId=self._client)
        return np.array([s[0] for s in states])

    def get_gripper_width(self) -> float:
        return self._gripper_width

    def reset(self):
        self.move_robot_to_joint_state(self.canonical_joint_pose, time_to_go=2.0)

    def step(
        self, joint_action: Optional[np.ndarray] = None, gripper_action: Optional[float] = None
    ):
        if joint_action is not None:
            joint_action = np.clip(joint_action, FRANKA_LOWER, FRANKA_UPPER)
            for i, a in enumerate(joint_action):
                p.setJointMotorControl2(
                    self._robot,
                    i,
                    p.POSITION_CONTROL,
                    targetPosition=a,
                    force=250,
                    maxVelocity=2.0,
                    physicsClientId=self._client,
                )
        if gripper_action is not None:
            self._gripper_width = float(gripper_action)
        p.stepSimulation(physicsClientId=self._client)

    def move_robot_to_joint_state(self, joint_state: np.ndarray, time_to_go: float = 4.0):
        """Interpolated motion to target config (mirrors Manimo soft_ctrl)."""
        start = self.get_joint_angles()
        target = np.clip(joint_state, FRANKA_LOWER, FRANKA_UPPER)
        n_steps = max(10, int(time_to_go * self.ctrl_hz))
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            self.step(joint_action=start + alpha * (target - start))
            if self.gui:
                time.sleep(1.0 / self.ctrl_hz)

    def get_scene_pcd(
        self,
        debug_raw_pcd: bool = False,
        debug_combined_pcd: bool = False,
        save_pcd: bool = False,
        save_file_name: str = "combined",
        filter: bool = True,
        denoise: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Called by NeuralMP.get_scene_pcd() after it moves the robot to a
        fixed observation config. Returns (points, colors), (N,3) each.
        """
        points, colors = self.get_multi_cam_pcd()
        # Remove points that belong to the robot
        points, colors = self._exclude_robot_pcd(points, colors)
        return points, colors

    def get_multi_cam_pcd(self) -> Tuple[np.ndarray, np.ndarray]:
        """Three virtual depth cameras → combined 3-D point cloud."""
        cam_cfgs = [
            dict(eye=[1.2, 0.0, 1.1], target=[0.5, 0.0, 0.6], up=[0, 0, 1]),
            dict(eye=[0.5, 1.2, 1.1], target=[0.5, 0.0, 0.6], up=[0, 0, 1]),
            dict(eye=[0.5, -1.2, 1.1], target=[0.5, 0.0, 0.6], up=[0, 0, 1]),
        ]
        all_pts, all_rgb = [], []
        for cfg in cam_cfgs:
            pts, rgb = self._render_depth_pcd(**cfg)
            if len(pts):
                all_pts.append(pts)
                all_rgb.append(rgb)
        if not all_pts:
            return np.zeros((1, 3)), np.zeros((1, 3))
        return np.concatenate(all_pts), np.concatenate(all_rgb)

    def execute_plan(
        self,
        plan: np.ndarray,
        init_joint_angles: np.ndarray,
        speed: float = 0.2,
        proprio_feedback: bool = False,
        render: bool = False,
    ) -> Tuple[bool, float, list]:
        """
        Execute a trajectory.
        Matches FrankaRealEnv.execute_plan() signature exactly.
        Returns (success, joint_error, frames).
        """
        if plan is None or len(plan) == 0:
            print("[Sim] execute_plan: empty plan.")
            return False, float("inf"), []

        print("[Sim] Moving to start config...")
        self.move_robot_to_joint_state(init_joint_angles, time_to_go=3.0)

        goal = np.clip(np.array(plan[-1]), FRANKA_LOWER, FRANKA_UPPER)
        frames = []

        print(f"[Sim] Executing {len(plan)}-waypoint trajectory...")
        for idx, wp in enumerate(plan):
            wp = np.clip(wp, FRANKA_LOWER, FRANKA_UPPER)

            if proprio_feedback or idx == 0:
                cur = self.get_joint_angles()
            else:
                cur = np.clip(plan[idx - 1], FRANKA_LOWER, FRANKA_UPPER)

            max_d = max(float(np.max(np.abs(wp - cur))), 1e-6)
            n_steps = max(5, int(max_d / speed * self.ctrl_hz))
            for k in range(n_steps):
                alpha = (k + 1) / n_steps
                self.step(joint_action=cur + alpha * (wp - cur))
                if self.gui:
                    time.sleep(1.0 / self.ctrl_hz)

        final_q = self.get_joint_angles()
        joint_err = float(np.linalg.norm(final_q - goal))

        goal_ee = self._fk_ee(goal)[:3]
        curr_ee = self._fk_ee(final_q)[:3]
        pos_err = np.linalg.norm(curr_ee - goal_ee) * 100  # cm
        success = pos_err < 2.0

        print(
            f"[Sim] joint_err={joint_err:.4f} rad | "
            f"EE pos_err={pos_err:.2f} cm | success={success}"
        )
        return success, joint_err, frames

    # ── obstacle helpers (used by the run script) ─────────────────────────────

    def add_obstacle_box(
        self, half_extents: List[float], position: List[float], color=(0.6, 0.3, 0.1, 1.0)
    ) -> int:
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=list(color),
            physicsClientId=self._client,
        )
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self._client
        )
        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=position,
            physicsClientId=self._client,
        )
        self._obstacles.append(bid)
        return bid

    def add_obstacle_sphere(
        self, radius: float, position: List[float], color=(0.1, 0.4, 0.8, 1.0)
    ) -> int:
        vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=list(color), physicsClientId=self._client
        )
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self._client)
        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=position,
            physicsClientId=self._client,
        )
        self._obstacles.append(bid)
        return bid

    def clear_obstacles(self):
        for bid in self._obstacles:
            p.removeBody(bid, physicsClientId=self._client)
        self._obstacles.clear()

    def visualize_ply(self, ply_path: str):
        """No-op: real env uses open3d for debug viz. Skipped in sim."""
        print(f"[Sim] visualize_ply({ply_path}) — skipped in simulation")

    def close(self):
        if self._client >= 0:
            p.disconnect(self._client)
            self._client = -1

    # ── private helpers ───────────────────────────────────────────────────────

    def _set_joints_instant(self, angles: np.ndarray):
        """Instant reset (no dynamics) — for FK queries."""
        for i, a in enumerate(angles[:7]):
            p.resetJointState(self._robot, i, a, physicsClientId=self._client)

    def _fk_ee(self, angles: np.ndarray) -> np.ndarray:
        """Forward kinematics → [x,y,z, qx,qy,qz,qw] of link 7."""
        saved = self.get_joint_angles()
        self._set_joints_instant(angles)
        for _ in range(3):
            p.stepSimulation(physicsClientId=self._client)
        state = p.getLinkState(self._robot, 7, physicsClientId=self._client)
        pos = np.array(state[4])
        orn = np.array(state[5])
        self._set_joints_instant(saved)
        for _ in range(3):
            p.stepSimulation(physicsClientId=self._client)
        return np.concatenate([pos, orn])

    def _get_sphere_centers_world(self, angles: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Return list of (world_center, radius) for each Franka sphere
        at the given joint configuration.
        """
        saved = self.get_joint_angles()
        self._set_joints_instant(angles)
        for _ in range(2):
            p.stepSimulation(physicsClientId=self._client)

        result = []
        for link_idx, local_off, radius in FRANKA_SPHERES:
            state = p.getLinkState(self._robot, link_idx, physicsClientId=self._client)
            lpos = np.array(state[4])
            rot = np.array(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
            wcenter = lpos + rot @ np.array(local_off)
            result.append((wcenter, radius))

        self._set_joints_instant(saved)
        for _ in range(2):
            p.stepSimulation(physicsClientId=self._client)
        return result

    def _exclude_robot_pcd(
        self, points: np.ndarray, colors: np.ndarray, thred: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove point-cloud points that lie inside robot sphere bounds."""
        if len(points) == 0:
            return points, colors
        spheres = self._get_sphere_centers_world(self.get_joint_angles())
        mask = np.ones(len(points), dtype=bool)
        for center, radius in spheres:
            dists = np.linalg.norm(points - center, axis=1)
            mask &= dists >= radius + thred
        return points[mask], colors[mask]

    def _render_depth_pcd(
        self,
        eye: List[float],
        target: List[float],
        up: List[float],
        width: int = 320,
        height: int = 240,
        fov: float = 60.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        aspect = width / height
        near, far = 0.1, 3.5

        view_mat = p.computeViewMatrix(eye, target, up, physicsClientId=self._client)
        proj_mat = p.computeProjectionMatrixFOV(
            fov, aspect, near, far, physicsClientId=self._client
        )
        _, _, rgb_raw, depth_raw, _ = p.getCameraImage(
            width,
            height,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._client,
        )

        depth = np.array(depth_raw, dtype=np.float32).reshape(height, width)
        depth_lin = far * near / (far - (far - near) * depth)

        # FIX 1: Correct Focal Length math
        f_y = height / (2.0 * np.tan(np.radians(fov / 2.0)))
        f_x = f_y * aspect

        cx, cy = width / 2.0, height / 2.0
        uu, vv = np.meshgrid(np.arange(width), np.arange(height))

        # FIX 2: Standard Computer Vision Coordinates (Z is forward)
        x_cv = (uu - cx) * depth_lin / f_x
        y_cv = (vv - cy) * depth_lin / f_y
        z_cv = depth_lin

        # FIX 3: Convert to OpenGL Coordinates (PyBullet ViewMatrix expects this)
        # OpenGL looks down Negative Z, and Y is Up.
        cam_pts_gl = np.stack([x_cv, -y_cv, -z_cv], axis=-1).reshape(-1, 3)

        # FIX 4: Safely transform to World Space using matrix inversion
        view_np = np.array(view_mat).reshape(4, 4).T
        inv_view = np.linalg.inv(view_np)

        # Multiply Points by Inverse View Matrix
        cam_pts_homo = np.hstack([cam_pts_gl, np.ones((cam_pts_gl.shape[0], 1))])
        world_pts = (inv_view @ cam_pts_homo.T).T[:, :3]

        # Colors
        rgb_img = np.array(rgb_raw, dtype=np.uint8).reshape(height, width, 4)
        colors = rgb_img[:, :, :3].reshape(-1, 3).astype(np.float32) / 255.0

        # --- UPDATE THIS BOTTOM SECTION ---

        # PyBullet raw depth is 1.0 when the ray hits the "void" (far clipping plane)
        valid_depth = depth.flatten() < 0.99

        # Keep points within table height AND that actually hit a real object
        z = world_pts[:, 2]
        mask = (z >= 0.60) & (z <= 1.60) & valid_depth & np.isfinite(world_pts).all(axis=1)

        return world_pts[mask], colors[mask]
