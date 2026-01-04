# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import argparse
import os
import sys

import mujoco
import mujoco_viewer
import numpy as np
import torch
from pynput import keyboard
import time

class SimToSimCfg:
    """Configuration class for sim2sim parameters.

    Must be kept consistent with the training configuration.
    """

    class sim:
        sim_duration = 100.0
        num_action = 20
        num_obs_per_step = 75
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25

    class robot:
        gait_air_ratio_l: float = 0.38
        gait_air_ratio_r: float = 0.38
        gait_phase_offset_l: float = 0.38
        gait_phase_offset_r: float = 0.88
        gait_cycle: float = 0.85


class MujocoRunner:
    """
    Sim2Sim runner that loads a policy and a MuJoCo model
    to run real-time humanoid control simulation.

    Args:
        cfg (SimToSimCfg): Configuration object for simulation.
        policy_path (str): Path to the TorchScript exported policy.
        model_path (str): Path to the MuJoCo XML model.
    """

    def __init__(self, cfg: SimToSimCfg, policy_path, model_path):
        self.cfg = cfg
        network_path = policy_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt

        self.policy = torch.jit.load(network_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False
        self.init_variables()

    def init_variables(self) -> None:
        """Initialize simulation variables and joint index mappings."""
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action)
        self.dof_vel = np.zeros(self.cfg.sim.num_action)
        self.action = np.zeros(self.cfg.sim.num_action)
        # Default joint positions matching training configuration (Isaac Sim order)
        # Order: left_leg(6) + right_leg(6) + left_arm(4) + right_arm(4) = 20
        # left_leg: hip_roll, hip_pitch, hip_yaw, knee, ankle_pitch, ankle_roll
        # right_leg: hip_roll, hip_pitch, hip_yaw, knee, ankle_pitch, ankle_roll
        # left_arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
        # right_arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
        self.default_dof_pos = np.array(
            [0.0, -0.1, 0.0, 0.2, -0.1, 0.0,  # left leg: hip_roll, hip_pitch, hip_yaw, knee, ankle_pitch, ankle_roll
             0.0, -0.1, 0.0, 0.2, -0.1, 0.0,  # right leg: hip_roll, hip_pitch, hip_yaw, knee, ankle_pitch, ankle_roll
             0.0, 0.0, 0.0, 0.0,              # left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
             0.0, 0.0, 0.0, 0.0]              # right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
        )
        self.episode_length_buf = 0
        self.gait_phase = np.zeros(2)
        self.gait_cycle = self.cfg.robot.gait_cycle
        self.phase_ratio = np.array([self.cfg.robot.gait_air_ratio_l, self.cfg.robot.gait_air_ratio_r])
        self.phase_offset = np.array([self.cfg.robot.gait_phase_offset_l, self.cfg.robot.gait_phase_offset_r])

        # MuJoCo joint order (from XML): 
        # 0:left_hip_pitch, 1:left_hip_roll, 2:left_hip_yaw, 3:left_knee, 4:left_ankle_pitch, 5:left_ankle_roll,
        # 6:right_hip_pitch, 7:right_hip_roll, 8:right_hip_yaw, 9:right_knee, 10:right_ankle_pitch, 11:right_ankle_roll,
        # 12:left_shoulder_pitch, 13:left_shoulder_roll, 14:left_shoulder_yaw, 15:left_elbow,
        # 16:right_shoulder_pitch, 17:right_shoulder_roll, 18:right_shoulder_yaw, 19:right_elbow
        # Isaac Sim joint order:
        # 0:left_hip_roll, 1:left_hip_pitch, 2:left_hip_yaw, 3:left_knee, 4:left_ankle_pitch, 5:left_ankle_roll,
        # 6:right_hip_roll, 7:right_hip_pitch, 8:right_hip_yaw, 9:right_knee, 10:right_ankle_pitch, 11:right_ankle_roll,
        # 12:left_shoulder_pitch, 13:left_shoulder_roll, 14:left_shoulder_yaw, 15:left_elbow,
        # 16:right_shoulder_pitch, 17:right_shoulder_roll, 18:right_shoulder_yaw, 19:right_elbow
        self.mujoco_to_isaac_idx = [
            1,  # MuJoCo[0]=left_hip_pitch -> Isaac[1]=left_hip_pitch
            0,  # MuJoCo[1]=left_hip_roll -> Isaac[0]=left_hip_roll
            2,  # MuJoCo[2]=left_hip_yaw -> Isaac[2]=left_hip_yaw
            3,  # MuJoCo[3]=left_knee -> Isaac[3]=left_knee
            4,  # MuJoCo[4]=left_ankle_pitch -> Isaac[4]=left_ankle_pitch
            5,  # MuJoCo[5]=left_ankle_roll -> Isaac[5]=left_ankle_roll
            7,  # MuJoCo[6]=right_hip_pitch -> Isaac[7]=right_hip_pitch
            6,  # MuJoCo[7]=right_hip_roll -> Isaac[6]=right_hip_roll
            8,  # MuJoCo[8]=right_hip_yaw -> Isaac[8]=right_hip_yaw
            9,  # MuJoCo[9]=right_knee -> Isaac[9]=right_knee
            10,  # MuJoCo[10]=right_ankle_pitch -> Isaac[10]=right_ankle_pitch
            11,  # MuJoCo[11]=right_ankle_roll -> Isaac[11]=right_ankle_roll
            12,  # MuJoCo[12]=left_shoulder_pitch -> Isaac[12]=left_shoulder_pitch
            13,  # MuJoCo[13]=left_shoulder_roll -> Isaac[13]=left_shoulder_roll
            14,  # MuJoCo[14]=left_shoulder_yaw -> Isaac[14]=left_shoulder_yaw
            15,  # MuJoCo[15]=left_elbow -> Isaac[15]=left_elbow
            16,  # MuJoCo[16]=right_shoulder_pitch -> Isaac[16]=right_shoulder_pitch
            17,  # MuJoCo[17]=right_shoulder_roll -> Isaac[17]=right_shoulder_roll
            18,  # MuJoCo[18]=right_shoulder_yaw -> Isaac[18]=right_shoulder_yaw
            19,  # MuJoCo[19]=right_elbow -> Isaac[19]=right_elbow
        ]
        # Inverse mapping: Isaac Sim -> MuJoCo
        self.isaac_to_mujoco_idx = [
            1,  # Isaac[0]=left_hip_roll -> MuJoCo[1]=left_hip_roll
            0,  # Isaac[1]=left_hip_pitch -> MuJoCo[0]=left_hip_pitch
            2,  # Isaac[2]=left_hip_yaw -> MuJoCo[2]=left_hip_yaw
            3,  # Isaac[3]=left_knee -> MuJoCo[3]=left_knee
            4,  # Isaac[4]=left_ankle_pitch -> MuJoCo[4]=left_ankle_pitch
            5,  # Isaac[5]=left_ankle_roll -> MuJoCo[5]=left_ankle_roll
            7,  # Isaac[6]=right_hip_roll -> MuJoCo[7]=right_hip_roll
            6,  # Isaac[7]=right_hip_pitch -> MuJoCo[6]=right_hip_pitch
            8,  # Isaac[8]=right_hip_yaw -> MuJoCo[8]=right_hip_yaw
            9,  # Isaac[9]=right_knee -> MuJoCo[9]=right_knee
            10,  # Isaac[10]=right_ankle_pitch -> MuJoCo[10]=right_ankle_pitch
            11,  # Isaac[11]=right_ankle_roll -> MuJoCo[11]=right_ankle_roll
            12,  # Isaac[12]=left_shoulder_pitch -> MuJoCo[12]=left_shoulder_pitch
            13,  # Isaac[13]=left_shoulder_roll -> MuJoCo[13]=left_shoulder_roll
            14,  # Isaac[14]=left_shoulder_yaw -> MuJoCo[14]=left_shoulder_yaw
            15,  # Isaac[15]=left_elbow -> MuJoCo[15]=left_elbow
            16,  # Isaac[16]=right_shoulder_pitch -> MuJoCo[16]=right_shoulder_pitch
            17,  # Isaac[17]=right_shoulder_roll -> MuJoCo[17]=right_shoulder_roll
            18,  # Isaac[18]=right_shoulder_yaw -> MuJoCo[18]=right_shoulder_yaw
            19,  # Isaac[19]=right_elbow -> MuJoCo[19]=right_elbow
        ]
        # Initial command vel
        self.command_vel = np.array([0.0, 0.0, 0.0])
        # Initialize observation history (will be filled gradually)
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )
        # Observation scales (matching training config)
        self.obs_scales = {
            'ang_vel': 1.0,
            'projected_gravity': 1.0,
            'commands': 1.0,
            'joint_pos': 1.0,
            'joint_vel': 1.0,
            'actions': 1.0,
        }

    def get_obs(self) -> np.ndarray:
        """
        Compute current observation vector from MuJoCo sensors and internal state.

        Returns:
            np.ndarray: Normalized and clipped observation history.
        """
        self.dof_pos = self.data.sensordata[0:20]
        self.dof_vel = self.data.sensordata[20:40]

        # Apply observation scales (matching training)
        # Angular velocity: convert from world frame to body frame
        # Isaac Lab uses root_ang_vel_b which is body frame angular velocity
        orientation_quat = self.data.sensor("orientation").data.astype(np.double)  # [w, x, y, z]
        quat_xyzw = orientation_quat[[1, 2, 3, 0]]  # [x, y, z, w] for quat_apply_inverse
        
        # Convert world frame angular velocity (qvel[3:6]) to body frame
        # This matches Isaac Lab's root_ang_vel_b computation
        ang_vel_world = self.data.qvel[3:6].astype(np.double)
        ang_vel = self.quat_apply_inverse(quat_xyzw, ang_vel_world) * self.obs_scales['ang_vel']
        
        # Projected gravity: rotate world gravity [0, 0, -1] to body frame
        projected_gravity = self.quat_apply_inverse(quat_xyzw, np.array([0, 0, -1])) * self.obs_scales['projected_gravity']
        command = self.command_vel * self.obs_scales['commands']
        # Convert dof_pos from MuJoCo order to Isaac Sim order, then subtract default positions
        dof_pos_isaac = self.dof_pos[self.mujoco_to_isaac_idx]
        joint_pos = (dof_pos_isaac - self.default_dof_pos) * self.obs_scales['joint_pos']
        # Convert dof_vel from MuJoCo order to Isaac Sim order (default_joint_vel is 0, so no subtraction needed)
        joint_vel = self.dof_vel[self.mujoco_to_isaac_idx] * self.obs_scales['joint_vel']
        action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions) * self.obs_scales['actions']

        obs = np.concatenate(
            [
                ang_vel,  # 3
                projected_gravity,  # 3
                command,  # 3
                joint_pos,  # 20
                joint_vel,  # 20
                action,  # 20
                np.sin(2 * np.pi * self.gait_phase),  # 2
                np.cos(2 * np.pi * self.gait_phase),  # 2
                self.phase_ratio,  # 2
            ],
            axis=0,
        ).astype(np.float32)

        # Update observation history
        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs.copy()

        return np.clip(self.obs_history, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)
    
    def debug_print(self, step: int):
        """Print detailed debug information about observations."""
        if step % 50 == 0 or step < 5:
            print(f"\n=== Step {step} ===")
            print(f"Root pos: {self.data.qpos[:3]}")
            print(f"Root quat: {self.data.qpos[3:7]}")
            print(f"Root vel (lin): {self.data.qvel[:3]}")
            print(f"Root vel (ang, world): {self.data.qvel[3:6]}")
            
            # Get current observation components
            orientation_quat = self.data.sensor("orientation").data.astype(np.double)
            quat_xyzw = orientation_quat[[1, 2, 3, 0]]
            ang_vel_world = self.data.qvel[3:6].astype(np.double)
            ang_vel_body = self.quat_apply_inverse(quat_xyzw, ang_vel_world)
            projected_gravity = self.quat_apply_inverse(quat_xyzw, np.array([0, 0, -1]))
            
            dof_pos_isaac = self.dof_pos[self.mujoco_to_isaac_idx]
            joint_pos_obs = (dof_pos_isaac - self.default_dof_pos)
            
            print(f"\n--- Observation Components ---")
            print(f"Angular velocity (body): {ang_vel_body}")
            print(f"Projected gravity: {projected_gravity}")
            print(f"Command: {self.command_vel}")
            print(f"Joint pos obs (first 6): {joint_pos_obs[:6]}")
            print(f"Joint vel (first 6, Isaac): {self.dof_vel[self.mujoco_to_isaac_idx][:6]}")
            print(f"Action (first 6): {self.action[:6]}")
            print(f"Gait phase: {self.gait_phase}, sin: {np.sin(2*np.pi*self.gait_phase)}, cos: {np.cos(2*np.pi*self.gait_phase)}")
            print(f"Phase ratio: {self.phase_ratio}")
            
            # Check observation history
            obs_history = self.get_obs()
            print(f"\n--- Observation History (last 75 dims) ---")
            print(f"Obs history shape: {obs_history.shape}")
            print(f"Last obs (first 10): {obs_history[-75:-65]}")
            print(f"Last obs (ang_vel): {obs_history[-75:-72]}")
            print(f"Last obs (proj_grav): {obs_history[-72:-69]}")
            print(f"Last obs (command): {obs_history[-69:-66]}")
            
            # Check action output
            target_pos = self.position_control()
            print(f"\n--- Control ---")
            print(f"Target pos (MuJoCo, first 6): {target_pos[:6]}")
            print(f"Current pos (MuJoCo, first 6): {self.dof_pos[:6]}")

    def position_control(self) -> np.ndarray:
        """
        Apply position control using scaled action.

        Returns:
            np.ndarray: Target joint positions in MuJoCo order.
        """
        # self.action is in Isaac Sim order, scale it
        actions_scaled = self.action * self.cfg.sim.action_scale
        # Add default positions (in Isaac Sim order)
        target_pos_isaac = actions_scaled + self.default_dof_pos
        # Convert to MuJoCo order
        return target_pos_isaac[self.isaac_to_mujoco_idx]

    def run(self) -> None:
        """
        Run the simulation loop with keyboard-controlled commands.
        """
        self.setup_keyboard_listener()
        self.listener.start()

        # Initialize robot to default pose
        # qpos: [root_pos(3), root_quat(4), joint_pos(20)]
        self.data.qpos[:3] = [0, 0, 1.05]  # root position
        self.data.qpos[3:7] = [1, 0, 0, 0]  # root quaternion (w, x, y, z)
        # Set joint positions: default_dof_pos is in Isaac Sim order, need to map to MuJoCo order
        mujoco_joint_pos = self.default_dof_pos[self.isaac_to_mujoco_idx]
        self.data.qpos[7:27] = mujoco_joint_pos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = mujoco_joint_pos
        mujoco.mj_forward(self.model, self.data)

        # Warm-up: Fill observation history before starting control
        print("[INFO] Warming up observation history...")
        # Fill observation history with stable observations (robot at rest, zero velocities)
        self.action[:] = 0.0
        for i in range(self.cfg.sim.actor_obs_history_length):
            # Force zero velocities and stable pose for warm-up observations
            self.data.qvel[:] = 0.0
            self.data.qpos[:3] = [0, 0, 1.05]
            self.data.qpos[3:7] = [1, 0, 0, 0]
            self.data.qpos[7:27] = mujoco_joint_pos
            mujoco.mj_forward(self.model, self.data)
            
            # Get observation at stable state (zero velocities)
            obs = self.get_obs()
            self.episode_length_buf += 1
            self.calculate_gait_para()
        
        # Now do a few steps to stabilize before starting control
        print("[INFO] Stabilizing robot...")
        for _ in range(10):
            self.data.ctrl = self.position_control()
            mujoco.mj_step(self.model, self.data)
            # Reset velocities if they get too large
            if np.any(np.abs(self.data.qvel[3:6]) > 0.1):
                self.data.qvel[3:6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        
        # Check initial state
        orientation_quat = self.data.sensor("orientation").data.astype(np.double)
        quat_xyzw = orientation_quat[[1,2,3,0]]
        initial_ang_vel_body = self.quat_apply_inverse(quat_xyzw, self.data.qvel[3:6])
        print(f"[INFO] Initial angular velocity (body): {initial_ang_vel_body}")
        print(f"[INFO] Initial joint velocities (first 6): {self.dof_vel[self.mujoco_to_isaac_idx][:6]}")
        
        if np.any(np.abs(initial_ang_vel_body) > 0.1) or np.any(np.abs(self.dof_vel[self.mujoco_to_isaac_idx][:6]) > 0.5):
            print(f"[WARNING] Large initial velocities detected! This may cause instability.")
        
        print("[INFO] Starting control loop...")

        step_count = 0
        while self.data.time < self.cfg.sim.sim_duration:
            # Get observation (this updates obs_history internally)
            obs_history = self.get_obs()
            
            # Get action from policy
            with torch.no_grad():
                action_tensor = self.policy(torch.tensor(obs_history, dtype=torch.float32))
                self.action[:] = action_tensor.detach().numpy()[:20]
            self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)
            
            # Debug output
            if step_count % 100 == 0:
                self.debug_print(step_count)

            # Apply action for decimation steps
            for sim_update in range(self.cfg.sim.decimation):
                step_start_time = time.time()

                self.data.ctrl = self.position_control()
                mujoco.mj_step(self.model, self.data)
                self.viewer.render()

                elapsed = time.time() - step_start_time
                sleep_time = self.cfg.sim.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.episode_length_buf += 1
            self.calculate_gait_para()
            step_count += 1

        self.listener.stop()
        self.viewer.close()

    def quat_apply_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate a vector by the inverse of a quaternion.

        Args:
            q (np.ndarray): Quaternion (x, y, z, w) format.
            v (np.ndarray): Vector to rotate.

        Returns:
            np.ndarray: Rotated vector.
        """
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0

        return a - b + c

    def calculate_gait_para(self) -> None:
        """
        Update gait phase parameters based on simulation time and offset.
        """
        t = self.episode_length_buf * self.dt / self.gait_cycle
        self.gait_phase[0] = (t + self.phase_offset[0]) % 1.0
        self.gait_phase[1] = (t + self.phase_offset[1]) % 1.0

    def adjust_command_vel(self, idx: int, increment: float) -> None:
        """
        Adjust command velocity vector.

        Args:
            idx (int): Index of velocity component (0=x, 1=y, 2=yaw).
            increment (float): Value to increment.
        """
        self.command_vel[idx] += increment
        self.command_vel[idx] = np.clip(self.command_vel[idx], -1.0, 1.0)  # vel clip

    def setup_keyboard_listener(self) -> None:
        """
        Set up keyboard event listener for user control input.
        """

        def on_press(key):
            try:
                if key.char == "8":  # NumPad 8      x += 0.2
                    self.adjust_command_vel(0, 0.2)
                elif key.char == "2":  # NumPad 2      x -= 0.2
                    self.adjust_command_vel(0, -0.2)
                elif key.char == "4":  # NumPad 4      y -= 0.2
                    self.adjust_command_vel(1, -0.2)
                elif key.char == "6":  # NumPad 6      y += 0.2
                    self.adjust_command_vel(1, 0.2)
                elif key.char == "7":  # NumPad 7      yaw += 0.2
                    self.adjust_command_vel(2, -0.2)
                elif key.char == "9":  # NumPad 9      yaw -= 0.2
                    self.adjust_command_vel(2, 0.2)
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press)


if __name__ == "__main__":
    LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    parser = argparse.ArgumentParser(description="Run sim2sim Mujoco controller.")
    parser.add_argument(
        "--task",
        type=str,
        default="walk",
        choices=["walk", "run"],
        help="Task type: 'walk' or 'run' to set gait parameters",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to policy.pt. If not specified, it will be set automatically based on --task",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(LEGGED_LAB_ROOT_DIR, "legged_lab/assets/tsts2/mjcf/tsts2_joint21.xml"),
        help="Path to model.xml",
    )
    parser.add_argument("--duration", type=float, default=100.0, help="Simulation duration in seconds")
    args = parser.parse_args()

    if args.policy is None:
        args.policy = os.path.join(LEGGED_LAB_ROOT_DIR, "Exported_policy", f"{args.task}.pt")

    if not os.path.isfile(args.policy):
        print(f"[ERROR] Policy file not found: {args.policy}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[ERROR] MuJoCo model file not found: {args.model}")
        sys.exit(1)

    print(f"[INFO] Loaded task preset: {args.task.upper()}")
    print(f"[INFO] Loaded policy: {args.policy}")
    print(f"[INFO] Loaded model: {args.model}")

    sim_cfg = SimToSimCfg()
    sim_cfg.sim.sim_duration = args.duration

    # Set gait parameters according to task
    if args.task == "walk":
        sim_cfg.robot.gait_air_ratio_l = 0.38
        sim_cfg.robot.gait_air_ratio_r = 0.38
        sim_cfg.robot.gait_phase_offset_l = 0.38
        sim_cfg.robot.gait_phase_offset_r = 0.88
        sim_cfg.robot.gait_cycle = 0.85
    elif args.task == "run":
        sim_cfg.robot.gait_air_ratio_l = 0.6
        sim_cfg.robot.gait_air_ratio_r = 0.6
        sim_cfg.robot.gait_phase_offset_l = 0.6
        sim_cfg.robot.gait_phase_offset_r = 0.1
        sim_cfg.robot.gait_cycle = 0.5

    runner = MujocoRunner(
        cfg=sim_cfg,
        policy_path=args.policy,
        model_path=args.model,
    )
    runner.run()
