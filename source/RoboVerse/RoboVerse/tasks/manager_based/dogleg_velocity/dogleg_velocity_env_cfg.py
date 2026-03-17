# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR


from . import mdp

##
# Pre-defined configs
##

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from RoboVerse.assets.dogleg.dogleg_newton.dogleg_newton import DoglegCfg

##
# Scene definition
##


@configclass
class DoglegSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""


    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=ROUGH_TERRAINS_CFG,
    #     max_init_terrain_level=5,
    #     collision_group=1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderwhiteBrickBondHoned/TilesMarbleSpiderwhiteBrickBondHoned.mdl",
    #         project_uvw=True,
    #         texture_scale=(0.25, 0.25),
    #     ),
    #     debug_vis=False,
    # )

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = DoglegCfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        filter_prim_paths_expr=[],
        history_length=3,
        track_air_time=True,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_velocity_leg = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*_thigh_big_joint", ".*_feet_big_joint"],
        scale=10.0,
    )
    joint_velocity_gimbal = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*gimbal_joint"],
        scale=5.0,
    )
    joint_position = mdp.JointPositionActionCfg(    
        asset_name="robot",
        joint_names=[".*shooter_j4_closeloop", ".*_ankle_back_joint"],
        scale=2.0,
    )



@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.8), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.3, 0.3)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 1.6), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.6, 0.6)
        ),
    )




@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            # self.history_length = 3
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

    # privileged observations
    critic: CriticCfg = CriticCfg()
    
@configclass
class EventCfg:
    """Configuration for events."""

    # New: Reset base/root position (add this)
    reset_base_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),  # No joint_names needed for root state
            "pose_range": {
                "x": (-5.0, 5.0),       # Randomize x position (meters)
                "y": (-5.0, 5.0),       # Randomize y position (meters)
                "z": (0.01, 0.01),     # Uncomment and set to fixed height or small range if needed
                # "roll": (-3.1416, 3.1416), # Small randomization (radians); omit for no change
                # "pitch": (-3.1416, 3.1416),
                "yaw": (-3.1416, 3.1416),  # Full random yaw; omit or narrow for less rotation
            },
            "velocity_range": {},  # Empty dict resets linear/angular velocities to zero (recommended for clean resets)
        },
    )
    # reset
    reset_joint_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "position_range": (-0.3, 0.3),
            "velocity_range": (-0.0, 0.0),
        },
    )

    # push_up = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(15, 20),
    #     params={
    #         "velocity_range": {
    #             # "x": (-1.0, -1.0),
    #             # "y": (-1.0, 1.0),
    #             "z": (-0.2, 0.5),
    #             # "roll": (-0.5, 0.5),
    #             # "pitch": (-0.5, 0.5),
    #             # "yaw": (-0.78, 0.78),
    #         }
    #     },
    # )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2,weight=-5.0,)

    # -- penalties
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    body_lin_acc = RewTerm(func=mdp.body_lin_acc_l2,weight=-0.0001,)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_feet.*"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*thigh.*"), "threshold": 1.0},
    # )

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])
        },
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.4,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*base.*"), "threshold": 1.0},
    )


##
# Environment configuration
##


@configclass
class DoglegEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DoglegSceneCfg = DoglegSceneCfg(num_envs=2**12, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation