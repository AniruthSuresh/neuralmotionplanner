"""
pybullet_franka_env.py
======================
Drop-in PyBullet replacement for FrankaRealEnv (from franka_real_env.py).

Inherits FrankaRealEnv's abstract interface exactly so NeuralMP / NeuralMPModel
work unchanged. Only get_multi_cam_pcd() is re-implemented — it renders a
simulated depth image and converts it to a point cloud instead of using
real RealSense cameras.

All you need installed:
    pip install pybullet numpy torch

The real neuralmotionplanner repo only needs to be installed for its
FrankaRealEnv base class and the NeuralMP/NeuralMPModel utilities.
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple

import numpy as np
import pybullet as p
import pybullet_data

# ── Franka joint constants (mirror neural_mp/utils/constants.py) ─────────────
FRANKA_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
FRANKA_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
FRANKA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
ARM_JOINT_IDS = list(range(7))  # joint indices in PyBullet panda.urdf
CTRL_HZ = 60


class PybulletFrankaEnv:
    """
    PyBullet Franka Panda environment.

    Implements the same public API as FrankaRealEnv so the NeuralMP wrapper
    works without any changes:
        get_joint_angles()
        get_gripper_width()
        reset()
        step(joint_action, gripper_action)
        move_robot_to_joint_state(joint_state, time_to_go)
        get_multi_cam_pcd()

    Extra helpers used by the eval script:
        add_obstacle_box / add_obstacle_sphere / clear_obstacles
        get_ee_pose_from_joint_angles
        execute_motion_plan
    """

    def __init__(self, gui: bool = True):
        self.gui = gui
        self.ctrl_hz = CTRL_HZ
        self._client = -1
        self._robot = -1
        self._plane = -1
        self._obstacles: List[int] = []
        self._gripper_width = 0.08  # open

        # canonical reset pose (mirrors Manimo default)
        self.canonical_joint_pose = FRANKA_HOME.copy()

        self._connect()
        self._load_robot()

    # ─────────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────────

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

    def _load_robot(self):
        self._plane = p.loadURDF("plane.urdf", physicsClientId=self._client)
        # Table
        p.loadURDF(
            "table/table.urdf",
            basePosition=[0.5, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 1.5708]),
            physicsClientId=self._client,
        )
        # Franka Panda
        panda_urdf = os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf")
        self._robot = p.loadURDF(
            panda_urdf, basePosition=[0, 0, 0.625], useFixedBase=True, physicsClientId=self._client
        )
        self._set_joints(FRANKA_HOME)
        for _ in range(50):
            p.stepSimulation(physicsClientId=self._client)

    # ─────────────────────────────────────────────────────────────────────────
    # FrankaRealEnv interface (required by NeuralMP)
    # ─────────────────────────────────────────────────────────────────────────

    def get_joint_angles(self) -> np.ndarray:
        """7-dof joint angles."""
        states = p.getJointStates(self._robot, ARM_JOINT_IDS, physicsClientId=self._client)
        return np.array([s[0] for s in states])

    def get_gripper_width(self) -> float:
        """Simulated gripper width (constant, gripper not actuated here)."""
        return self._gripper_width

    def reset(self):
        """Reset to canonical home pose."""
        self.move_robot_to_joint_state(self.canonical_joint_pose, time_to_go=2.0)

    def step(self, joint_action: np.ndarray = None, gripper_action: float = None):
        """Single sim step — matches the Manimo step() signature."""
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
        """
        Interpolate to target joint state over time_to_go seconds.
        Matches Manimo's soft_ctrl behaviour.
        """
        start = self.get_joint_angles()
        target = np.clip(joint_state, FRANKA_LOWER, FRANKA_UPPER)
        n_steps = max(10, int(time_to_go * self.ctrl_hz))
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            q = start + alpha * (target - start)
            self.step(joint_action=q)
            if self.gui:
                time.sleep(1.0 / self.ctrl_hz)

    def get_multi_cam_pcd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulated multi-camera point cloud using PyBullet's built-in renderer.

        Renders 2 virtual depth cameras (front + side) and projects to 3-D.
        Returns (points, colors) each shape (N, 3), matching the real env API.
        """
        cam_configs = [
            # front camera
            dict(eye=[1.2, 0.0, 1.1], target=[0.5, 0.0, 0.6], up=[0, 0, 1]),
            # side camera
            dict(eye=[0.5, 1.2, 1.1], target=[0.5, 0.0, 0.6], up=[0, 0, 1]),
        ]
        all_pts, all_rgb = [], []
        for cfg in cam_configs:
            pts, rgb = self._render_depth_pcd(**cfg)
            all_pts.append(pts)
            all_rgb.append(rgb)
        points = np.concatenate(all_pts, axis=0)
        colors = np.concatenate(all_rgb, axis=0)
        return points, colors

    # ─────────────────────────────────────────────────────────────────────────
    # Extras used by the Neural MP planner (not in abstract base)
    # ─────────────────────────────────────────────────────────────────────────

    def get_ee_pose_from_joint_angles(self, joint_angles: np.ndarray) -> np.ndarray:
        """FK: 7-D EE pose [x,y,z, qx,qy,qz,qw] for given config."""
        saved = self.get_joint_angles()
        self._set_joints(joint_angles)
        for _ in range(5):
            p.stepSimulation(physicsClientId=self._client)
        state = p.getLinkState(self._robot, 7, physicsClientId=self._client)
        pos = np.array(state[4])
        orn = np.array(state[5])
        self._set_joints(saved)
        for _ in range(5):
            p.stepSimulation(physicsClientId=self._client)
        return np.concatenate([pos, orn])

    def add_obstacle_box(
        self, half_extents: List[float], position: List[float], color=(0.6, 0.3, 0.1, 1.0)
    ) -> int:
        """Add a box obstacle. Returns body id."""
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
        """Add a sphere obstacle. Returns body id."""
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

    def execute_motion_plan(self, trajectory: np.ndarray, speed: float = 0.2) -> Tuple[bool, float]:
        """
        Execute a trajectory (N×7 joint angles).
        Mirrors NeuralMP.execute_motion_plan() signature exactly.
        Returns (success, joint_error).
        """
        if trajectory is None or len(trajectory) == 0:
            print("[Sim] Empty trajectory — planning failed.")
            return False, float("inf")

        goal = trajectory[-1]
        for i, wp in enumerate(trajectory):
            wp = np.clip(wp, FRANKA_LOWER, FRANKA_UPPER)
            cur = self.get_joint_angles()
            max_delta = np.max(np.abs(wp - cur))
            n_steps = max(5, int(max_delta / speed * self.ctrl_hz))
            for k in range(n_steps):
                alpha = (k + 1) / n_steps
                q = cur + alpha * (wp - cur)
                self.step(joint_action=q)
                if self.gui:
                    time.sleep(1.0 / self.ctrl_hz)

        final_q = self.get_joint_angles()
        joint_err = float(np.linalg.norm(final_q - goal))

        goal_ee = self.get_ee_pose_from_joint_angles(goal)[:3]
        state = p.getLinkState(self._robot, 7, physicsClientId=self._client)
        curr_ee = np.array(state[4])
        pos_err = np.linalg.norm(curr_ee - goal_ee) * 100  # cm

        success = pos_err < 2.0  # 2 cm threshold (slightly relaxed for sim)
        print(
            f"[Sim] Joint error: {joint_err:.4f} rad | "
            f"EE pos error: {pos_err:.2f} cm | "
            f"Success: {success}"
        )
        return success, joint_err

    def close(self):
        if self._client >= 0:
            p.disconnect(self._client)
            self._client = -1

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _set_joints(self, angles: np.ndarray):
        """Instant joint reset (no dynamics)."""
        for i, a in enumerate(angles[:7]):
            p.resetJointState(self._robot, i, a, physicsClientId=self._client)

    def _render_depth_pcd(
        self,
        eye: List[float],
        target: List[float],
        up: List[float],
        width: int = 320,
        height: int = 240,
        fov: float = 60.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render a depth image from a virtual camera and unproject to 3-D.
        Returns (points, colors) filtered to table-level scene.
        """
        aspect = width / height
        near, far = 0.1, 3.0

        view_mat = p.computeViewMatrix(eye, target, up, physicsClientId=self._client)
        proj_mat = p.computeProjectionMatrixFOV(
            fov, aspect, near, far, physicsClientId=self._client
        )

        _, _, rgb_img, depth_img, _ = p.getCameraImage(
            width,
            height,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._client,
        )

        # Linearise depth buffer
        depth_img = np.array(depth_img, dtype=np.float32).reshape(height, width)
        depth_lin = far * near / (far - (far - near) * depth_img)

        # Unproject pixel → camera coords
        f_x = width / (2 * np.tan(np.radians(fov / 2) * aspect))
        f_y = height / (2 * np.tan(np.radians(fov / 2)))
        cx, cy = width / 2, height / 2

        u = np.arange(width)
        v = np.arange(height)
        uu, vv = np.meshgrid(u, v)

        X_c = (uu - cx) / f_x * depth_lin
        Y_c = (vv - cy) / f_y * depth_lin
        Z_c = depth_lin

        cam_pts = np.stack([X_c, Y_c, Z_c], axis=-1).reshape(-1, 3)

        # Camera → world transform
        view_np = np.array(view_mat).reshape(4, 4).T
        R_cw = view_np[:3, :3]
        t_cw = view_np[:3, 3]
        world_pts = (R_cw.T @ (cam_pts.T - t_cw[:, None])).T

        # RGB colours
        rgb_img = np.array(rgb_img, dtype=np.uint8).reshape(height, width, 4)
        colors = rgb_img[:, :, :3].reshape(-1, 3).astype(np.float32) / 255.0

        # Filter: keep points in workspace (z ≥ 0.6 table surface, z ≤ 1.5 m)
        z = world_pts[:, 2]
        mask = (z >= 0.60) & (z <= 1.50) & np.isfinite(world_pts).all(axis=1)
        return world_pts[mask], colors[mask]
