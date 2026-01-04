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

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`G1_MINIMAL_CFG`: G1 humanoid robot with minimal collision bodies
* :obj:`TIENKUNG2LITE_CFG`: TienKung2Lite robot with effort-strength scaled actuator

Reference: https://github.com/unitreerobotics/unitree_ros
"""


import torch
from collections.abc import Sequence
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass
from isaaclab.actuators import (
    ImplicitActuatorCfg,
    DelayedPDActuator,
    DelayedPDActuatorCfg,
)
from isaaclab.utils.types import ArticulationActions


class Tsts2DelayedPDActuator(DelayedPDActuator):


    def __init__(
        self,
        cfg: DelayedPDActuatorCfg,
        joint_names: list[str],
        joint_ids: Sequence[int],
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        armature: torch.Tensor | float = 0.0,
        friction: torch.Tensor | float = 0.0,
        effort_limit: torch.Tensor | float = torch.inf,
        velocity_limit: torch.Tensor | float = torch.inf,
    ):

        super().__init__(
            cfg, joint_names, joint_ids, num_envs, device, 
            stiffness, damping, armature, friction, 
            torch.inf, torch.inf
        )

        self._effort_strength = math_utils.sample_uniform(
            *cfg.effort_strength, (num_envs, self.num_joints), device=device
        )
        print(f"Tsts2DelayedPDActuator initialized: effort_strength range {cfg.effort_strength}")

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:

        control_action = super().compute(control_action, joint_pos, joint_vel)
        

        scaled_effort = control_action.joint_efforts * self._effort_strength

        applied_effort = self._clip_effort(scaled_effort)

        control_action.joint_efforts = applied_effort

        return control_action


@configclass
class Tsts2DelayedPDActuatorCfg(DelayedPDActuatorCfg):

    class_type: type = Tsts2DelayedPDActuator  
    effort_strength: tuple[float, float] = (1.0, 1.0)  


import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from legged_lab.assets import ISAAC_ASSET_DIR

TIENKUNG2LITE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/tsts2/usd/tsts2_joint21_ArmDown.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.1,  
            ".*_knee_joint": 0.2,
            ".*_ankle_pitch_joint": -0.1,
            ".*_ankle_roll_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={

        "legs": Tsts2DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint"],

            effort_limit_sim=380,
            velocity_limit_sim=10.0,
            stiffness={
                ".*_hip_yaw_joint": 300,
                ".*_hip_roll_joint": 400,
                ".*_hip_pitch_joint": 400,
                ".*_knee_joint": 400,
            },
            damping={
                ".*_hip_yaw_joint": 5,
                ".*_hip_roll_joint": 5,
                ".*_hip_pitch_joint": 5,
                ".*_knee_joint": 5,
            },

            effort_strength=(0.8, 1.2),

            min_delay=0,
            max_delay=0,
        ),
        "feet": Tsts2DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],

            effort_limit_sim=80,
            velocity_limit_sim=10.0,
            stiffness={".*_ankle.*": 30},
            damping={".*_ankle.*": 2},

            effort_strength=(0.9, 1.1),
            min_delay=0,
            max_delay=0,
        ),

        "arms": Tsts2DelayedPDActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"],

            effort_limit_sim=100,
            velocity_limit_sim=10.0,
            stiffness={
                ".*_shoulder_pitch_joint": 50,
                ".*_shoulder_roll_joint": 50,
                ".*_shoulder_yaw_joint": 50,
                ".*_elbow_joint": 50,
            },
            damping={
                ".*_shoulder_pitch_joint": 2,
                ".*_shoulder_roll_joint": 2,
                ".*_shoulder_yaw_joint": 2,
                ".*_elbow_joint": 2,
            },

            effort_strength=(0.9, 1.1),
            min_delay=0,
            max_delay=0,
        ),
    },
)
