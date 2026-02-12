"""Microduck imitation (motion tracking) environment configuration."""

from copy import deepcopy
from pathlib import Path

from mjlab.managers.manager_term_config import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,  # Not needed - using only fell_over from base config
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.tasks import imitation_mdp, mdp as microduck_mdp
from mjlab_microduck.tasks.imitation_command import ImitationCommandCfg
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)

# Domain randomization toggles
ENABLE_COM_RANDOMIZATION = True
ENABLE_KP_RANDOMIZATION = True
ENABLE_KD_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True  # Can enable once walking is stable
ENABLE_JOINT_FRICTION_RANDOMIZATION = False  # Too disruptive - affects joint movement
ENABLE_JOINT_DAMPING_RANDOMIZATION = False  # Too disruptive - affects joint dynamics
ENABLE_VELOCITY_PUSHES = True  # Velocity-based pushes for robustness training
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # Simulates mounting errors

# Domain randomization ranges (adjust as needed)
# Conservative ranges proven to be stable - can increase gradually if needed
COM_RANDOMIZATION_RANGE = 0.005  # ±3mm
MASS_INERTIA_RANDOMIZATION_RANGE = (0.90, 1.0)  # ±5% applied to BOTH mass and inertia together.
KP_RANDOMIZATION_RANGE = (0.8, 1.2)  # ±15%
KD_RANDOMIZATION_RANGE = (0.8, 1.2)  # ±10% (can increase to 0.8-1.2)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.98, 1.02)  # ±2% VERY conservative - affects walking
JOINT_DAMPING_RANDOMIZATION_RANGE = (0.98, 1.02)  # ±2% VERY conservative - affects dynamics
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)  # Apply pushes every 3-6 seconds
VELOCITY_PUSH_RANGE = (-0.5, 0.5)  # Velocity change range in m/s
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 2.0  # ±2° IMU mounting error


def make_microduck_imitation_env_cfg(play: bool = False, ghost_vis: bool = False):
    """Create Microduck imitation (motion tracking) environment configuration.

    Starts from the base velocity config and replaces commands, observations, and rewards
    with motion tracking versions.

    Args:
        play: If True, disables observation corruption and extends episode length for visualization.
        ghost_vis: If True, enables ghost visualization of reference motion. Default False.

    Returns:
        Environment configuration for imitation motion tracking task.
    """

    # Start with base velocity config (includes all the domain randomization we want)
    cfg = make_microduck_velocity_env_cfg(play=play)

    # Determine motion file path
    motion_file = str(Path(__file__).parent.parent / "data" / "reference_motion.pkl")

    ##
    # Commands - Replace velocity command with imitation command
    ##

    cfg.commands = {
        "imitation": ImitationCommandCfg(
            entity_name="robot",
            motion_file=motion_file,
            resampling_time_range=(1.0e9, 1.0e9),  # Never resample (motion handles resets)
            debug_vis=ghost_vis,  # Enable ghost visualization only if requested
            velocity_cmd_range={
                "x": (-0.1, 0.15),
                "y": (-0.15, 0.15),
                "yaw": (-1.0, 1.0),
            },
            sampling_mode="uniform" if not play else "adaptive",
            rel_standing_envs=0.0,  # Disable for now
        )
    }

    ##
    # Observations - Keep policy observations simple, add motion data to critic
    ##

    # Policy observations (what the robot can actually sense)
    # Build in specific order: command, phase, base_ang_vel, raw_accelerometer, joint_pos, joint_vel, actions
    base_obs = cfg.observations["policy"].terms
    policy_terms = {
        "command": ObservationTermCfg(
            func=imitation_mdp.velocity_command,
            params={"command_name": "imitation"},
        ),
        "phase": ObservationTermCfg(
            func=imitation_mdp.motion_phase,
            params={"command_name": "imitation"},
        ),
        "base_ang_vel": base_obs["base_ang_vel"],
        # Use raw accelerometer instead of projected gravity
        "raw_accelerometer": ObservationTermCfg(
            func=microduck_mdp.raw_accelerometer,
            scale=1.0,
        ),
        "joint_pos": base_obs["joint_pos"],
        "joint_vel": base_obs["joint_vel"],
        "actions": base_obs["actions"],
    }

    # Critic observations (privileged information including reference motion)
    critic_terms = {
        **policy_terms,
        "motion_root_pos_b": ObservationTermCfg(
            func=imitation_mdp.motion_root_pos_b,
            params={"command_name": "imitation"},
        ),
        "motion_root_ori_b": ObservationTermCfg(
            func=imitation_mdp.motion_root_ori_b,
            params={"command_name": "imitation"},
        ),
        "motion_joint_pos": ObservationTermCfg(
            func=imitation_mdp.motion_joint_pos,
            params={"command_name": "imitation"},
        ),
        "motion_joint_vel": ObservationTermCfg(
            func=imitation_mdp.motion_joint_vel,
            params={"command_name": "imitation"},
        ),
        "motion_joint_vel_error": ObservationTermCfg(
            func=imitation_mdp.motion_joint_vel_error,
            params={"command_name": "imitation"},
        ),
    }

    cfg.observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    ##
    # Rewards - Replace with motion tracking rewards
    ##

    cfg.rewards = {
        # Motion tracking rewards
        "imitation_root_pos": RewardTermCfg(
            func=imitation_mdp.imitation_root_position_error_exp,
            weight=1.0,
            params={"command_name": "imitation", "std": 0.22},
        ),
        "imitation_root_ori": RewardTermCfg(
            func=imitation_mdp.imitation_root_orientation_error_exp,
            weight=1.0,
            params={"command_name": "imitation", "std": 0.22},
        ),
        "imitation_lin_vel": RewardTermCfg(
            func=imitation_mdp.imitation_linear_velocity_error_exp,
            weight=1.0,
            params={"command_name": "imitation", "std": 0.35},
        ),
        "imitation_ang_vel": RewardTermCfg(
            func=imitation_mdp.imitation_angular_velocity_error_exp,
            weight=1.0,
            params={"command_name": "imitation", "std": 0.7},
        ),
        # Velocity command tracking - reward for following velocity commands
        "velocity_cmd_tracking": RewardTermCfg(
            func=imitation_mdp.imitation_velocity_cmd_tracking_exp,
            weight=1.0,  # Start small - can increase gradually (try 2.0, 3.0, etc.)
            params={"command_name": "imitation", "std": 0.5},
        ),
        "imitation_joint_pos_legs": RewardTermCfg(
            func=imitation_mdp.imitation_joint_position_error,
            weight=15.0,  # was 10.0
            params={
                "command_name": "imitation",
                "joint_names": (
                    "right_hip_yaw",
                    "right_hip_roll",
                    "right_hip_pitch",
                    "right_knee",
                    "right_ankle",
                    "left_hip_yaw",
                    "left_hip_roll",
                    "left_hip_pitch",
                    "left_knee",
                    "left_ankle",
                ),
            },
        ),
        "imitation_joint_pos_non_legs": RewardTermCfg(
            func=imitation_mdp.imitation_joint_position_error,
            weight=10.0,  # Reduced from 50.0 - allow natural head movements for balance
            params={
                "command_name": "imitation",
                "joint_names": ("neck_pitch", "head_pitch", "head_yaw", "head_roll"),
            },
        ),
        "foot_contact_match": RewardTermCfg(
            func=imitation_mdp.imitation_foot_contact_match,
            weight=4.0,  # Was 8.0
            params={
                "command_name": "imitation",
                "sensor_name": "feet_ground_contact",
                "force_threshold": 2.5,  # Minimum force (N) to count as contact (~35% of 6.86N total weight)
                "debug_print": play,  # Enable debug printing in play mode
            },
        ),
        "foot_clearance": RewardTermCfg(
            func=imitation_mdp.foot_clearance_reward,
            weight=1.0,  # Was 2.0
            params={
                "command_name": "imitation",
                "sensor_name": "feet_ground_contact",
                "target_height": 0.02,  # 2cm clearance
            },
        ),
        "no_double_support": RewardTermCfg(
            func=imitation_mdp.double_support_penalty,
            weight=1.0,  # Was 2.0
            params={
                "command_name": "imitation",
                "sensor_name": "feet_ground_contact",
                "force_threshold": 2.5,
            },
        ),
        # Regularization rewards (keep from base config)
        "action_rate_l2": RewardTermCfg(
            func=velocity_mdp.action_rate_l2,
            weight=-0.1,
        ),
        # Action acceleration penalties (2nd derivative) for smoother actions
        "leg_action_acceleration": RewardTermCfg(
            func=microduck_mdp.leg_action_acceleration_l2,
            weight=-0.01,  # Start small - can increase gradually
        ),
        "neck_action_acceleration": RewardTermCfg(
            func=microduck_mdp.neck_action_acceleration_l2,
            weight=-0.01,  # Start small - can increase gradually
        ),
        "self_collisions": RewardTermCfg(
            func=velocity_mdp.self_collision_cost,
            weight=-10.0,
            params={"sensor_name": "self_collision"},
        ),
        "alive": RewardTermCfg(
            func=velocity_mdp.is_alive,
            weight=1.0,
        ),
        "termination": RewardTermCfg(
            func=velocity_mdp.is_terminated,
            weight=-100.0,
        ),
        "soft_landing": RewardTermCfg(
            func=velocity_mdp.soft_landing,
            weight=-0.01,  # Increased from -1e-5 to discourage aggressive landings
            params={"sensor_name": "feet_ground_contact"},
        ),
        "smooth_contacts": RewardTermCfg(
            func=imitation_mdp.smooth_foot_forces,
            weight=-0.001,  # Penalize rapid force changes (prevents "tap tap")
            params={"sensor_name": "feet_ground_contact"},
        ),
        "body_ang_vel": RewardTermCfg(
            func=velocity_mdp.body_angular_velocity_penalty,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
        ),
    }

    ##
    # Terminations - Add motion tracking terminations
    ##

    # DISABLED: These early terminations prevent learning recovery strategies
    # and hurt sim2real transfer. The robot needs to learn how to recover from
    # deviations rather than immediately dying when it deviates from reference.
    #
    cfg.terminations["root_pos_error"] = TerminationTermCfg(
        func=imitation_mdp.bad_root_pos,
        params={
            "command_name": "imitation",
            "threshold": 0.15,
            "curriculum_threshold_name": "root_pos_threshold",
        },
    )
    cfg.terminations["root_ori_error"] = TerminationTermCfg(
        func=imitation_mdp.bad_root_ori,
        params={
            "command_name": "imitation",
            "threshold": 0.8,
            "curriculum_threshold_name": "root_ori_threshold",
        },
    )
    # Keep the fell_over termination from base config

    ##
    # Observation Noise (disable in play mode)
    ##

    if not play:
        cfg.observations["policy"].terms["base_ang_vel"] = deepcopy(
            cfg.observations["policy"].terms["base_ang_vel"]
        )
        cfg.observations["policy"].terms["raw_accelerometer"] = deepcopy(
            cfg.observations["policy"].terms["raw_accelerometer"]
        )
        cfg.observations["policy"].terms["joint_pos"] = deepcopy(
            cfg.observations["policy"].terms["joint_pos"]
        )
        cfg.observations["policy"].terms["joint_vel"] = deepcopy(
            cfg.observations["policy"].terms["joint_vel"]
        )

        # Add noise and delay to observations - increased to match real IMU latency
        cfg.observations["policy"].terms["base_ang_vel"].delay_min_lag = 0
        cfg.observations["policy"].terms["base_ang_vel"].delay_max_lag = 3  # 40-120ms at 50Hz
        cfg.observations["policy"].terms["base_ang_vel"].delay_update_period = 64
        cfg.observations["policy"].terms["base_ang_vel"].noise = Unoise(n_min=-0.4, n_max=0.4)

        cfg.observations["policy"].terms["raw_accelerometer"].delay_min_lag = 0
        cfg.observations["policy"].terms["raw_accelerometer"].delay_max_lag = 3
        cfg.observations["policy"].terms["raw_accelerometer"].delay_update_period = 64
        cfg.observations["policy"].terms["raw_accelerometer"].noise = Unoise(n_min=-0.15, n_max=0.15)

        cfg.observations["policy"].terms["joint_pos"].noise = Unoise(n_min=-0.1, n_max=0.1)
        cfg.observations["policy"].terms["joint_vel"].noise = Unoise(n_min=-4.0, n_max=4.0)

    ##
    # Domain Randomization Events
    ##

    # Velocity-based pushes for robustness training
    if ENABLE_VELOCITY_PUSHES:
        # In play mode, use shorter interval for better visibility
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S
        velocity_range = (
            (-1.5, 1.5) if play else VELOCITY_PUSH_RANGE
        )  # Larger pushes in play mode for visibility

        cfg.events["push_robot"] = EventTermCfg(
            func=velocity_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": velocity_range,
                    "y": velocity_range,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    # CoM randomization
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=velocity_mdp.randomize_field,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "field": "body_ipos",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    # PD gains randomization
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

    # Mass and inertia randomization
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=microduck_mdp.randomize_mass_and_inertia,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "scale_range": MASS_INERTIA_RANDOMIZATION_RANGE,
            },
        )

    # IMU orientation randomization (simulates mounting errors)
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        cfg.events["randomize_imu_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_imu_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE,
            },
        )

    # Joint friction randomization (disabled by default)
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=velocity_mdp.randomize_field,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "field": "dof_frictionloss",
                "ranges": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # Joint damping randomization (disabled by default)
    if ENABLE_JOINT_DAMPING_RANDOMIZATION:
        cfg.events["randomize_joint_damping"] = EventTermCfg(
            func=velocity_mdp.randomize_field,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "field": "dof_damping",
                "ranges": JOINT_DAMPING_RANDOMIZATION_RANGE,
            },
        )

    ##
    # Curriculum - Action rate curriculum for smoother movements
    ##

    cfg.curriculum = {
        "action_rate_weight": CurriculumTermCfg(
            func=velocity_mdp.reward_weight,
            params={
                "reward_name": "action_rate_l2",
                "weight_stages": [
                    # Gradually increase action smoothness penalty
                    {"step": 0, "weight": -0.6},
                    {"step": 250 * 24, "weight": -0.8},
                    {"step": 500 * 24, "weight": -1.0},  # Smoother actions for sim2real
                ],
            },
        ),
        "velocity_push_magnitude": CurriculumTermCfg(
            func=imitation_mdp.velocity_push_magnitude_curriculum,
            params={
                "event_name": "push_robot",
                "velocity_stages": [
                    # Start pushes from beginning - learn walking + robustness together
                    # This matches velocity task approach which transfers well
                    {"step": 0, "velocity_range": (-0.4, 0.4)},       # Gentle from start
                    {"step": 250 * 24, "velocity_range": (-0.6, 0.6)},  # Medium
                    {"step": 500 * 24, "velocity_range": (-0.8, 0.8)},  # Stronger
                    {"step": 750 * 24, "velocity_range": (-1.0, 1.0)},  # Full strength
                ],
            },
        ),
        # "root_pos_termination": CurriculumTermCfg(
            # func=imitation_mdp.termination_threshold_curriculum,
            # params={
                # "threshold_param_name": "root_pos_threshold",
                # "threshold_stages": [
                    # # Start strict, then relax to allow recovery from pushes
                    # {"step": 0, "threshold": 0.15},  # Initial: 15cm deviation = terminate
                    # {"step": 500 * 24, "threshold": 0.3},  # After 500 iters: 30cm
                    # {"step": 750 * 24, "threshold": 0.5},  # After 750 iters: 50cm
                    # {"step": 1000 * 24, "threshold": 1.0},  # After 1000 iters: 1m (very relaxed)
                # ],
            # },
        # ),
        # "root_ori_termination": CurriculumTermCfg(
            # func=imitation_mdp.termination_threshold_curriculum,
            # params={
                # "threshold_param_name": "root_ori_threshold",
                # "threshold_stages": [
                    # # Start strict, then relax
                    # {"step": 0, "threshold": 0.8},  # Initial: ~46 degrees
                    # {"step": 500 * 24, "threshold": 1.2},  # After 500 iters: ~69 degrees
                    # {"step": 750 * 24, "threshold": 1.6},  # After 750 iters: ~92 degrees
                    # {"step": 1000 * 24, "threshold": 2.5},  # After 1000 iters: ~143 degrees (very relaxed)
                # ],
            # },
        # ),
    }

    # Extend episode length for play mode
    if play:
        cfg.episode_length_s = 1e9

    return cfg


# RL configuration for imitation task
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

MicroduckImitationRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="imitation",  # Directory name
    run_name="imitation",  # Appended to datetime in wandb: <datetime>_imitation
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
