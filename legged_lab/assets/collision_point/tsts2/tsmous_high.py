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

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import torch
from dataclasses import MISSING
from typing import Union, Dict, Type
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils.types import ArticulationActions
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

class Tsts2ImplicitActuator(ImplicitActuator):


    cfg: "Tsts2ImplicitActuatorCfg"

    def __init__(self, cfg: "Tsts2ImplicitActuatorCfg", *args, **kwargs):

        super().__init__(cfg, *args, **kwargs)
        

        self._motor_strength = math_utils.sample_uniform(
            *cfg.motor_strength, 
            (self._num_envs, self.num_joints), 
            device=self._device
        )
        print("----set motor strength domain random ------",cfg.motor_strength)


    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        
        

        control_action = super().compute(control_action, joint_pos, joint_vel)
        
        self.computed_effort *= self._motor_strength
        
        self.applied_effort = self._clip_effort(self.computed_effort)

        
        return control_action


@configclass
class Tsts2ImplicitActuatorCfg(ImplicitActuatorCfg):

    class_type: Type[Tsts2ImplicitActuator] = Tsts2ImplicitActuator
    

    motor_strength: tuple[float, float] = (1.0, 1.0)

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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.1, #-0.3,
            ".*_knee_joint": 0.2, #0.3,
            ".*_ankle_pitch_joint": -0.1, #-0.18,
            ".*_ankle_roll_joint": 0.0,
            # "torso_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.0, #0.2,
            "left_shoulder_roll_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.0, #-0.6, 
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint", ],
            effort_limit_sim=380,
            velocity_limit_sim=15.0,
            stiffness={
                ".*_hip_yaw_joint": 500,
                ".*_hip_roll_joint": 700,
                ".*_hip_pitch_joint":700,
                ".*_knee_joint": 700,
                # "torso_joint": 200,
            },
            damping={
                ".*_hip_yaw_joint": 5,
                ".*_hip_roll_joint": 10,
                ".*_hip_pitch_joint": 10,
                ".*_knee_joint": 10,
                # "torso_joint": 6,
            },


            ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint",".*_ankle_roll_joint"],
            effort_limit_sim=80,
            velocity_limit_sim=15.0,
            stiffness={".*_ankle.*": 30},
            damping={".*_ankle.*": 2},

            
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"],
            effort_limit_sim=100,
            velocity_limit_sim=12.0,
            stiffness={
                ".*_shoulder_pitch_joint": 60,
                ".*_shoulder_roll_joint": 60,
                ".*_shoulder_yaw_joint": 30,
                ".*_elbow_joint": 10,
            },
            damping={
                ".*_shoulder_pitch_joint": 2,
                ".*_shoulder_roll_joint": 2,
                ".*_shoulder_yaw_joint": 2,
                ".*_elbow_joint": 2,
            },
        ),
        
    },
)
