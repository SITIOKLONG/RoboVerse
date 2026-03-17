import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import DCMotorCfg, ActuatorNetLSTMCfg
import os

DoglegCfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.getcwd() + "/source/RoboVerse/RoboVerse/assets/dogleg/dogleg_newton/dogleg4.usd",
        # usd_path="/home/rl/jacksit/newton/RmLab/source/RmLab/RmLab/assets/dogleg/dogleg4/dogleg_converter/dogleg4.usda",
        scale=(1.0, 1.0, 1.0),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,       # for simulating collision of gimbal and dogleg
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=False,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(
        #     mesh_collision_property=sim_utils.ConvexDecompositionPropertiesCfg(),
        #     collision_enabled=True,
        # )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "thigh": DCMotorCfg(    # velocity contorl
            joint_names_expr=[".*_thigh_big_joint"],
            velocity_limit_sim=1e9,
            effort_limit_sim=1e9,
            stiffness={
                ".*": 0.0,      # set to zero for using velocity control
            },
            effort_limit={
                ".*": 40.0,
            },
            saturation_effort=120.0,
            velocity_limit=1000.0,   # TODO: need check for real velocity
            damping={
                ".*": 100.0,
            },
            armature=0.1,
            friction=0.0,
        ),
        "ankle": DCMotorCfg(    # position control
            joint_names_expr=[".*_ankle_back_joint"],
            velocity_limit_sim=1e9,
            effort_limit_sim=1e9,
            stiffness={
                ".*": 30.0,
            },
            effort_limit={
                ".*": 40.0,
            },
            saturation_effort=120.0,
            velocity_limit=100.0,
            damping={
                ".*": 2.0,
            },
            armature=0.1,
            friction=0.0,
        ),
        "feet": DCMotorCfg(     # velocity control 
            joint_names_expr=[".*_feet_big_joint"],
            velocity_limit_sim=1e9,
            effort_limit_sim=1e9,
            stiffness={
                ".*": 0.0,
            },
            effort_limit={
                ".*": 9.0,
            },
            saturation_effort=27.0,
            velocity_limit=100.0,
            damping={
                ".*": 0.1,
            },
            armature=0.1,
            friction=0.0,
        ),
        "gimbal": DCMotorCfg(       # velocity control
            joint_names_expr=[".*gimbal_joint"],
            velocity_limit_sim=1e9,
            effort_limit_sim=1e9,
            stiffness={
                ".*": 0.0,
            },
            effort_limit={
                ".*": 20.0,
            },
            saturation_effort=60.0,
            velocity_limit=100.0,
            damping={
                ".*": 0.5,
            },
            armature=0.1,
            friction=0.0,
        ),
        "shooter": DCMotorCfg(       # position control
            joint_names_expr=[".*shooter_j4_closeloop"],
            velocity_limit_sim=1e9,
            effort_limit_sim=1e9,
            stiffness={
                ".*": 30.0,
            },
            effort_limit={
                ".*": 9.0,
            },
            saturation_effort=27.0,
            velocity_limit=100.0,
            damping={
                ".*": 0.5,
            },
            armature=0.1   ,
            friction=0.0,
        ),
    },
)
