"""Microduck velocity environment — roller skate variant"""

import math
from copy import deepcopy

# Domain randomization toggles (same as velocity env)
ENABLE_COM_RANDOMIZATION = True
ENABLE_KP_RANDOMIZATION = True
ENABLE_KD_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = False
ENABLE_JOINT_DAMPING_RANDOMIZATION = False
ENABLE_VELOCITY_PUSHES = False
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_NECK_OFFSET_RANDOMIZATION = False

NECK_OFFSET_MAX_ANGLE = 2.5
NECK_OFFSET_INTERVAL_S = (2.0, 5.0)

USE_PROJECTED_GRAVITY = True

COM_RANDOMIZATION_RANGE = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
KP_RANDOMIZATION_RANGE = (0.85, 1.15)
KD_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.98, 1.02)
JOINT_DAMPING_RANDOMIZATION_RANGE = (0.98, 1.02)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
VELOCITY_PUSH_RANGE = (-0.3, 0.3)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 1.0

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROLLERS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp


def make_microduck_velocity_rollers_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck roller skate velocity tracking environment configuration."""

    std_standing = {
        r".*hip_yaw.*": 0.05,
        r".*hip_roll.*": 0.05,
        r".*hip_pitch.*": 0.05,
        r".*knee.*": 0.05,
        r".*ankle.*": 0.05,
        r".*neck.*": 0.05,
        r".*head.*": 0.05,
        r".*passive_.*": 999.0,
    }

    std_walking = {
        r".*hip_yaw.*": 0.3,
        r".*hip_roll.*": 0.3,
        r".*hip_pitch.*": 0.4,
        r".*knee.*": 0.4,
        r".*ankle.*": 0.25,
        r".*neck.*": 0.1,
        r".*head.*": 0.1,
        r".*passive_.*": 999.0,
    }

    std_running = {
        r".*hip_yaw.*": 0.5,
        r".*hip_roll.*": 0.5,
        r".*hip_pitch.*": 0.8,
        r".*knee.*": 0.8,
        r".*ankle.*": 0.5,
        r".*neck.*": 0.1,
        r".*head.*": 0.1,
        r".*passive_.*": 999.0,
    }

    site_names = ["left_foot", "right_foot"]

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(roller_foot1|roller_foot2)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg = make_velocity_env_cfg()

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    # Robot setup
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROLLERS_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    # Action configuration
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    if ENABLE_NECK_OFFSET_RANDOMIZATION:
        joint_pos_action.class_type = microduck_mdp.NeckOffsetJointPositionAction

    # === REWARDS ===
    # Strip everything back to the minimum: pose, upright, com_height, velocity tracking.
    # Let the robot discover the skating gait through exploration.

    # Keep only what we want; delete everything else from the base env
    keep = {"pose", "upright", "track_linear_velocity", "body_ang_vel", "angular_momentum", "action_rate_l2"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    cfg.rewards["pose"].params["std_standing"] = std_standing
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_running
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].params["running_threshold"] = 100.0  # always use walking stds
    cfg.rewards["pose"].weight = 2.0

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 3.0

    cfg.rewards["track_linear_velocity"].weight = 10.0
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.08)

    cfg.rewards["com_height_target"] = RewardTermCfg(
        func=microduck_mdp.com_height_target,
        weight=2.0,
        params={
            "target_height_min": 0.0935,  # 0.08 + 0.0135 (roller height offset)
            "target_height_max": 0.1235,  # 0.11 + 0.0135
        },
    )

    # Regularization — same values as velocity env
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["action_rate_l2"].weight = -0.4  # ramped up by curriculum
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-0.5
    )
    cfg.rewards["neck_joint_pos_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_joint_pos_l2, weight=-2.0
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1e-3
    )

    # === EVENTS ===
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    if ENABLE_NECK_OFFSET_RANDOMIZATION:
        cfg.events["reset_neck_offset"] = EventTermCfg(
            func=microduck_mdp.reset_neck_offset,
            mode="reset",
        )
        cfg.events["randomize_neck_offset_target"] = EventTermCfg(
            func=microduck_mdp.randomize_neck_offset_target,
            mode="interval",
            interval_range_s=NECK_OFFSET_INTERVAL_S,
            params={"max_offset": NECK_OFFSET_MAX_ANGLE},
        )

    del cfg.events["foot_friction"]

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.1335, 0.1435)

    if ENABLE_VELOCITY_PUSHES:
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
    else:
        del cfg.events["push_robot"]

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=mdp.randomize_field,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "field": "body_ipos",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=microduck_mdp.randomize_mass_and_inertia,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "scale_range": MASS_INERTIA_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        cfg.events["randomize_imu_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_imu_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE,
            },
        )

    # === OBSERVATIONS ===
    del cfg.observations["policy"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )

    gravity_term_name = "projected_gravity" if USE_PROJECTED_GRAVITY else "raw_accelerometer"

    cfg.observations["policy"].terms[gravity_term_name] = deepcopy(
        cfg.observations["policy"].terms[gravity_term_name]
    )
    cfg.observations["policy"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["policy"].terms["base_ang_vel"]
    )

    cfg.observations["policy"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["policy"].terms["base_ang_vel"].delay_max_lag = 3
    cfg.observations["policy"].terms["base_ang_vel"].delay_update_period = 64

    cfg.observations["policy"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["policy"].terms[gravity_term_name].delay_max_lag = 3
    cfg.observations["policy"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["policy"].terms["base_ang_vel"].noise = Unoise(n_min=-0.024, n_max=0.024)
    cfg.observations["policy"].terms[gravity_term_name].noise = Unoise(n_min=-0.007, n_max=0.007)
    cfg.observations["policy"].terms["joint_pos"].noise = Unoise(n_min=-0.0006, n_max=0.0006)
    cfg.observations["policy"].terms["joint_vel"].noise = Unoise(n_min=-0.024, n_max=0.024)

    # Exclude passive wheel joints from joint_pos and joint_vel observations
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    cfg.observations["policy"].terms["joint_pos"].params["asset_cfg"] = passive_excluded
    cfg.observations["policy"].terms["joint_vel"].params["asset_cfg"] = passive_excluded
    cfg.observations["critic"].terms["joint_pos"].params["asset_cfg"] = deepcopy(passive_excluded)
    cfg.observations["critic"].terms["joint_vel"].params["asset_cfg"] = deepcopy(passive_excluded)

    # Wheel velocities in critic only (not sim2real safe)
    wheel_cfg = SceneEntityCfg("robot", joint_names=(r"^passive_.*",))
    cfg.observations["critic"].terms["wheel_vel"] = ObservationTermCfg(
        func=mdp.joint_vel_rel,
        scale=1.0,
        params={"asset_cfg": wheel_cfg},
    )

    # === COMMANDS ===
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = (0.3, 0.6)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.viz.z_offset = 0.5
    command.class_type = microduck_mdp.VelocityCommandCommandOnly

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # === CURRICULUM ===
    cfg.curriculum["velocity_command_ranges"] = CurriculumTermCfg(
        func=microduck_mdp.velocity_command_ranges_curriculum,
        params={
            "command_name": "twist",
            "update_lin_vel_y": False,
            "update_ang_vel_z": False,
            "forward_only": True,
            "velocity_stages": [
                {"step": 0,          "lin_vel_range": 0.6,  "ang_vel_range": 0.0},
                {"step": 1000 * 24,  "lin_vel_range": 0.8,  "ang_vel_range": 0.0},
                {"step": 2000 * 24,  "lin_vel_range": 1.2,  "ang_vel_range": 0.0},
                {"step": 3000 * 24,  "lin_vel_range": 1.5,  "ang_vel_range": 0.0},
            ],
        },
    )

    if ENABLE_NECK_OFFSET_RANDOMIZATION:
        cfg.curriculum["neck_offset_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.neck_offset_curriculum,
            params={
                "event_name": "randomize_neck_offset_target",
                "offset_stages": [
                    {"step": 0,          "max_offset": 0.0},
                    {"step": 500 * 24,   "max_offset": 0.3},
                    {"step": 1000 * 24,  "max_offset": 0.7},
                    {"step": 2000 * 24,  "max_offset": 1.3},
                    {"step": 3000 * 24,  "max_offset": NECK_OFFSET_MAX_ANGLE},
                ],
            },
        )

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.4},
                {"step": 250 * 24, "weight": -0.5},
                {"step": 500 * 24, "weight": -0.6},
                {"step": 750 * 24, "weight": -0.7},
                {"step": 1000 * 24, "weight": -0.8},
                {"step": 1250 * 24, "weight": -0.9},
                {"step": 1500 * 24, "weight": -1.0},
            ],
        },
    )

    # Tighter fall termination — prevents inverted-pendulum wheelie exploit
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(45.0)},
    )

    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    return cfg


MicroduckRollersRlCfg = RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.03,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="velocity_rollers",
    run_name="velocity_rollers",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
