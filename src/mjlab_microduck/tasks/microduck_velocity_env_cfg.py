"""Microduck environment"""

import math
from copy import deepcopy

# Domain randomization toggles
ENABLE_COM_RANDOMIZATION = True
ENABLE_KP_RANDOMIZATION = True
ENABLE_KD_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True  # Can enable once walking is stable
ENABLE_JOINT_FRICTION_RANDOMIZATION = False  # Too disruptive - affects joint movement
ENABLE_JOINT_DAMPING_RANDOMIZATION = False  # Too disruptive - affects joint dynamics
ENABLE_VELOCITY_PUSHES = True  # Velocity-based pushes for robustness training
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # Simulates mounting errors
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False  # Randomize initial tilt to force reactive behavior

# Observation configuration
USE_PROJECTED_GRAVITY = True  # If True, use projected gravity instead of raw accelerometer

# Domain randomization ranges (adjust as needed)
# Conservative ranges proven to be stable - can increase gradually if needed
COM_RANDOMIZATION_RANGE = 0.003  # ±3mm
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)  # ±5% applied to BOTH mass and inertia together.
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # ±15%
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10% (can increase to 0.8-1.2)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.98, 1.02)  # ±2% VERY conservative - affects walking
JOINT_DAMPING_RANDOMIZATION_RANGE = (0.98, 1.02)  # ±2% VERY conservative - affects dynamics
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)  # Apply pushes every 3-6 seconds
VELOCITY_PUSH_RANGE = (-0.3, 0.3)  # Velocity change range in m/s
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 1.0  # ±2° IMU mounting error
BASE_ORIENTATION_MAX_PITCH_DEG = 10.0  # ±10° forward/backward tilt at episode start
BASE_ORIENTATION_MAX_ROLL_DEG = 5.0  # ±5° side-to-side tilt at episode start

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
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

from mjlab_microduck.robot.microduck_constants import MICRODUCK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp


def make_microduck_velocity_env_cfg(
    play: bool = False
) -> ManagerBasedRlEnvCfg:
    """Create Microduck velocity tracking environment configuration."""

    std_walking = {
        # Lower body
        r".*hip_yaw.*": 0.3,
        r".*hip_roll.*": 0.2,
        r".*hip_pitch.*": 0.4,
        r".*knee.*": 0.4,
        r".*ankle.*": 0.25, # was 0.15
        # Head
        r".*neck.*": 0.1,
        r".*head.*": 0.1,
    }

    site_names = ["left_foot", "right_foot"]

    # Contact sensor for feet - LEFT, RIGHT order
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(foot_tpu_bottom|foot)$",  # LEFT foot first, RIGHT foot second
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

    # mode="body" (not subtree) so only geoms directly on these bodies are checked,
    # avoiding false positives from foot geoms in the subtree of parent bodies.
    # trunk_base has the uc collision mesh, head_base_plate has its own collision mesh.
    non_foot_ground_cfg = ContactSensorCfg(
        name="non_foot_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(trunk_base|head_base_plate)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )

    # Base configuration
    cfg = make_velocity_env_cfg()
    # Replace instant tilt termination with a sustained one: reset only after 3s at >80°
    # cfg.terminations["fell_over"] = TerminationTermCfg(
        # func=microduck_mdp.bad_orientation_sustained,
        # params={
            # "limit_angle": math.radians(70.0),
            # "duration": 0.0,
        # },
    # )

    # for to_remove in [
    #     # "foot_clearance",
    #     # "foot_swing_height",
    #     # "angular_momentum",
    #     # "body_ang_vel",
    # ]:
    #     del cfg.rewards[to_remove]

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    # Robot setup
    cfg.scene.entities = {"robot": MICRODUCK_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, non_foot_ground_cfg)
    cfg.viewer.body_name = "trunk_base"

    # Action configuration
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # === REWARDS ===
    # Pose reward configuration
    cfg.rewards["pose"].params["std_standing"] = std_walking
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].weight = 2.0  # was 1.0

    # Body-specific reward configurations
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 1.0  # was 1.0

    # Foot-specific configurations
    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    # Body-specific configurations
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)

    cfg.rewards["foot_slip"].weight = -0.1  # was -1.0
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.01

    # Body dynamics rewards
    cfg.rewards["soft_landing"].weight = -1e-05

    # Air time reward
    cfg.rewards["air_time"].weight = 5.0
    cfg.rewards["air_time"].params["command_threshold"] = 0.01
    cfg.rewards["air_time"].params["threshold_min"] = 0.10  # Increased from 0.055 to slow down gait (100ms swing)
    cfg.rewards["air_time"].params["threshold_max"] = 0.25  # Increased from 0.15 to allow slower stepping (250ms max swing)

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    # Velocity tracking rewards
    cfg.rewards["track_linear_velocity"].weight = 3.0 # Checkpoint : 3
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.15) # Checkpoint 0.15
    cfg.rewards["track_angular_velocity"].weight = 3.0 # Checkpoint : 3
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.40) # Checkpoint 0.4

    # Action smoothness
    cfg.rewards["action_rate_l2"].weight = -0.6 # was -0.4

    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.01
    cfg.rewards["foot_clearance"].params["target_height"] = 0.01  # Reduced for small robot (was 0.03)

    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.01  # Reduced for small robot (was 0.03)

    # cfg.rewards["leg_action_rate_l2"] = RewardTermCfg(
    # func=microduck_mdp.leg_action_rate_l2, weight=-0.5
    # )

    # Leg joint velocity penalty (encourage slower, smoother motion)
    # cfg.rewards["leg_joint_vel_l2"] = RewardTermCfg(
    # func=microduck_mdp.leg_joint_vel_l2, weight=-0.02
    # )

    # Neck stability
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-0.1
    )
    # cfg.rewards["neck_joint_vel_l2"] = RewardTermCfg(
    # func=microduck_mdp.neck_joint_vel_l2, weight=-0.1
    # )

    # CoM height target
    cfg.rewards["com_height_target"] = RewardTermCfg(
        func=microduck_mdp.com_height_target,
        weight=1.2,
        params={
            "target_height_min": 0.08,
            "target_height_max": 0.11,
        },
    )

    # === SURVIVAL REWARD (applies to all tasks) ===
    # Critical baseline reward for staying alive
    # cfg.rewards["survival"] = RewardTermCfg(
    #     func=microduck_mdp.is_alive, weight=2.0
    # )

    # Non-foot ground contact penalty (discourages sitting/kneeling)
    cfg.rewards["non_foot_ground_contact"] = RewardTermCfg(
        func=microduck_mdp.non_foot_ground_contact, weight=-10.0
    )

    # === REGULARIZATION REWARDS (applies to all tasks) ===
    # Joint torques penalty
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1e-3
    )

    # Joint accelerations penalty
    # cfg.rewards["joint_accelerations_l2"] = RewardTermCfg(
    # func=microduck_mdp.joint_accelerations_l2, weight=-2.5e-6
    # )

    # Leg action acceleration penalty
    # cfg.rewards["leg_action_acceleration_l2"] = RewardTermCfg(
    # func=microduck_mdp.leg_action_acceleration_l2, weight=-0.45
    # )

    # Neck action acceleration penalty
    # cfg.rewards["neck_action_acceleration_l2"] = RewardTermCfg(
    # func=microduck_mdp.neck_action_acceleration_l2, weight=-5.0
    # )

    # Events
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = foot_frictions_geom_names
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    # Velocity-based pushes for robustness training
    if ENABLE_VELOCITY_PUSHES:
        from mjlab.managers.scene_entity_config import SceneEntityCfg

        # In play mode, use shorter interval for better visibility
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

    # Domain randomization - sampled once per episode at reset
    if ENABLE_COM_RANDOMIZATION:
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        # Randomize CoM position
        cfg.events["randomize_com"] = EventTermCfg(
            func=mdp.randomize_field,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "field": "body_ipos",  # Body inertial position (CoM)
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        # Randomize motor PD gains
        # Uses custom function that handles DelayedActuator
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
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        # Randomize mass and inertia together (physically consistent)
        # Using the same scale for both prevents invalid inertia tensors
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=microduck_mdp.randomize_mass_and_inertia,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "scale_range": MASS_INERTIA_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        # Randomize joint friction losses (wear, temperature effects)
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=mdp.randomize_field,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "field": "dof_frictionloss",
                "ranges": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_JOINT_DAMPING_RANDOMIZATION:
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        # Randomize joint damping (lubrication, temperature effects)
        cfg.events["randomize_joint_damping"] = EventTermCfg(
            func=mdp.randomize_field,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "field": "dof_damping",
                "ranges": JOINT_DAMPING_RANDOMIZATION_RANGE,
            },
        )

    # IMU orientation randomization (simulates mounting errors)
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        cfg.events["randomize_imu_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_imu_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE,
            },
        )

    # Base orientation randomization (forces reactive behavior)
    if ENABLE_BASE_ORIENTATION_RANDOMIZATION:
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        cfg.events["randomize_base_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_base_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_pitch_deg": BASE_ORIENTATION_MAX_PITCH_DEG,
                "max_roll_deg": BASE_ORIENTATION_MAX_ROLL_DEG,
            },
        )

    # Observations
    del cfg.observations["policy"].terms["base_lin_vel"]

    # Add base_lin_vel to critic only (privileged information)
    from mjlab.managers.manager_term_config import ObservationTermCfg
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )

    # Determine gravity/accelerometer term name based on flag
    gravity_term_name = "projected_gravity" if USE_PROJECTED_GRAVITY else "raw_accelerometer"

    # Replace projected_gravity with raw_accelerometer if flag is False
    if not USE_PROJECTED_GRAVITY:
        # Remove projected_gravity and add raw_accelerometer
        del cfg.observations["policy"].terms["projected_gravity"]
        from mjlab.managers.manager_term_config import ObservationTermCfg
        cfg.observations["policy"].terms["raw_accelerometer"] = ObservationTermCfg(
            func=microduck_mdp.raw_accelerometer,
            scale=1.0,
        )

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

    # Observation noise configuration (edit these values as needed)
    cfg.observations["policy"].terms["base_ang_vel"].noise = Unoise(n_min=-0.024, n_max=0.024) # was 0.2
    cfg.observations["policy"].terms[gravity_term_name].noise = Unoise(n_min=-0.007, n_max=0.007)  # was 0.15
    cfg.observations["policy"].terms["joint_pos"].noise = Unoise(n_min=-0.0006, n_max=0.0006)  # was 0.05
    cfg.observations["policy"].terms["joint_vel"].noise = Unoise(n_min=-0.024, n_max=0.024)  # was 2.0

    # Commands
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.02
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = (-0.3, 0.3)
    command.ranges.lin_vel_y = (-0.3, 0.3)
    command.ranges.ang_vel_z = (-1.5, 1.5)
    command.viz.z_offset = 0.5

    # Terrain
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Add action rate curriculum
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                # 250 iterations × 24 steps/iter = 6000 steps
                {"step": 0, "weight": -0.4},
                {"step": 250 * 24, "weight": -0.8},
                {"step": 500 * 24, "weight": -1.0},
                # {"step": 750 * 24, "weight": -1.2},
                # {"step": 1000 * 24, "weight": -1.4},
                # {"step": 1250 * 24, "weight": -1.6},
                # {"step": 1500 * 24, "weight": -1.8},
                # {"step": 1750 * 24, "weight": -1.8},
            ],
        },
    )

    # Add linear velocity tracking curriculum
    # cfg.curriculum["linear_velocity_weight"] = CurriculumTermCfg(
        # func=mdp.reward_weight,
        # params={
            # "reward_name": "track_linear_velocity",
            # "weight_stages": [
                # {"step": 0, "weight": 2.0},
                # {"step": 500 * 24, "weight": 3.0},
                # {"step": 750 * 24, "weight": 4.0},
            # ],
        # },
    # )

    # Add angular velocity tracking curriculum
    # cfg.curriculum["angular_velocity_weight"] = CurriculumTermCfg(
        # func=mdp.reward_weight,
        # params={
            # "reward_name": "track_angular_velocity",
            # "weight_stages": [
                # {"step": 0, "weight": 2.0},
                # {"step": 500 * 24, "weight": 3.0},
                # {"step": 750 * 24, "weight": 4.0},
            # ],
        # },
    # )

    # # Add standing envs curriculum - gradually increase fraction of standing envs
    # cfg.curriculum["standing_envs"] = CurriculumTermCfg(
        # func=microduck_mdp.standing_envs_curriculum,
        # params={
            # "command_name": "twist",
            # "standing_stages": [
                # # Start with very few standing envs, gradually increase
                # # 250 iterations × 24 steps/iter = 6000 steps
                # {"step": 0, "rel_standing_envs": 0.02},
                # {"step": 250 * 24, "rel_standing_envs": 0.05},
                # {"step": 500 * 24, "rel_standing_envs": 0.1},
                # {"step": 750 * 24, "rel_standing_envs": 0.15},
                # {"step": 1000 * 24, "rel_standing_envs": 0.2},
                # {"step": 1250 * 24, "rel_standing_envs": 0.25},
            # ],
        # },
    # )

    # Push curriculum - start with no pushes (learn clean gait), gradually increase (build robustness)
    # Steps are in env steps (iteration * 24)
    # cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
        # func=microduck_mdp.push_curriculum,
        # params={
            # "event_name": "push_robot",
            # "push_stages": [
                # {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},                   # No pushes - learn basic walking
                # {"step": 250 * 24, "velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},     # Small pushes - build initial robustness
                # {"step": 500 * 24, "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},         # Full pushes - final robustness
            # ],
        # },
    # )

    # Velocity command ranges curriculum - gradually increase target velocities
    # Steps are in env steps (iteration * 24)
    cfg.curriculum["velocity_command_ranges"] = CurriculumTermCfg(
        func=microduck_mdp.velocity_command_ranges_curriculum,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {"step": 0, "lin_vel_range": 0.3, "ang_vel_range": 1.5},           # Start at current values
                {"step": 500 * 24, "lin_vel_range": 0.4, "ang_vel_range": 1.75},   # Intermediate step
                {"step": 1000 * 24, "lin_vel_range": 0.5, "ang_vel_range": 2.0},   # Target values
            ],
        },
    )

    # Disable default curriculum
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    return cfg


MicroduckRlCfg = RslRlOnPolicyRunnerCfg(
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
        entropy_coef=0.01,
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
    experiment_name="velocity",  # Directory name
    run_name="velocity",  # Appended to datetime in wandb: <datetime>_velocity
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
