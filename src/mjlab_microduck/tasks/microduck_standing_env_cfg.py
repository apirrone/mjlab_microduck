"""Microduck standing (balance) environment configuration."""

from copy import deepcopy
from pathlib import Path

from mjlab.managers.manager_term_config import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.tasks import imitation_mdp, mdp as microduck_mdp
from mjlab_microduck.tasks.imitation_command import ImitationCommandCfg
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


# Domain randomization toggles - same as imitation
ENABLE_COM_RANDOMIZATION = True
ENABLE_KP_RANDOMIZATION = True
ENABLE_KD_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = False
ENABLE_JOINT_DAMPING_RANDOMIZATION = False
ENABLE_VELOCITY_PUSHES = True  # Velocity-based pushes for robustness training
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True

# Domain randomization ranges (same as imitation)
COM_RANDOMIZATION_RANGE = 0.003  # ±3mm
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)  # ±5%
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # ±15%
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10%
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.98, 1.02)  # ±2%
JOINT_DAMPING_RANDOMIZATION_RANGE = (0.98, 1.02)  # ±2%
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)  # Apply pushes every 3-6 seconds
VELOCITY_PUSH_RANGE = (-0.6, 0.6)  # Velocity change range in m/s
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 1.0  # ±1° IMU mounting error


def make_microduck_standing_env_cfg(play: bool = False):
    """Create Microduck standing (balance) environment configuration.

    This task teaches the robot to stand still while being robust to perturbations.
    Uses the same observations as the imitation task (including raw accelerometer)
    but focuses on balance and pose matching rather than motion tracking.
    """

    ##
    # Base config from velocity environment
    ##

    cfg = make_microduck_velocity_env_cfg(play=play)

    # Determine motion file path (same as imitation task)
    motion_file = str(Path(__file__).parent.parent / "data" / "reference_motion.pkl")

    ##
    # Commands - Use ImitationCommand with zero velocity for compatibility
    ##

    # Use ImitationCommand so we have the same phase signal as imitation task
    # This allows dynamic switching between standing and walking policies
    cfg.commands = {
        "imitation": ImitationCommandCfg(
            entity_name="robot",
            motion_file=motion_file,
            resampling_time_range=(1.0e9, 1.0e9),  # Never resample
            debug_vis=False,  # No ghost for standing
            velocity_cmd_range={
                "x": (0.0, 0.0),  # Standing only - no velocity commands
                "y": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            sampling_mode="uniform" if not play else "adaptive",
        )
    }

    ##
    # Observations - EXACT same as imitation task
    ##

    # Get base observations from velocity config
    base_obs = cfg.observations["policy"].terms

    # IMPORTANT: Keep exact same observation order as imitation task
    # Order: command, phase, base_ang_vel, raw_accelerometer, joint_pos, joint_vel, actions
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
        "raw_accelerometer": ObservationTermCfg(
            func=microduck_mdp.raw_accelerometer,
            scale=1.0,
        ),
        "joint_pos": base_obs["joint_pos"],
        "joint_vel": base_obs["joint_vel"],
        "actions": base_obs["actions"],
    }

    cfg.observations["policy"] = ObservationGroupCfg(
        terms=policy_terms,
        enable_corruption=not play,
        concatenate_terms=True,
    )

    # Critic gets same observations
    critic_terms = deepcopy(policy_terms)
    cfg.observations["critic"] = ObservationGroupCfg(
        terms=critic_terms,
        enable_corruption=False,
        concatenate_terms=True,
    )

    ##
    # Rewards - Focus on standing, balance, and robustness
    ##

    # Standard deviation for different standing postures
    std_standing = {
        # Lower body - tight tolerances for standing
        r".*hip_yaw.*": 0.15,
        r".*hip_roll.*": 0.15,
        r".*hip_pitch.*": 0.2,
        r".*knee.*": 0.2,
        r".*ankle.*": 0.15,
        # Head - allow some flexibility
        r".*neck.*": 0.1,
        r".*head.*": 0.1,
    }

    cfg.rewards = {
        # Main reward: Match default standing pose
        # Reduced weight to allow recovery stepping
        "pose": RewardTermCfg(
            func=velocity_mdp.variable_posture,
            weight=2.5,  # Reduced from 4.0 to allow stepping
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "command_name": "imitation",  # Use imitation command for consistency
                "std_standing": std_standing,
                "std_walking": std_standing,
                "std_running": std_standing,
                "walking_threshold": 0.01,  # Always use standing std
                "running_threshold": 1.5,
            },
        ),
        # Reward taking steps when pushed (recovery behavior)
        "recovery_stepping": RewardTermCfg(
            func=microduck_mdp.recovery_stepping_reward,
            weight=3.0,  # Encourage stepping when velocity is high
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_threshold": 0.3,  # Start rewarding stepping above 0.3 m/s
                "air_time_threshold": 0.05,  # Foot must be in air for at least 50ms
            },
        ),
        # Stay upright
        "upright": RewardTermCfg(
            func=velocity_mdp.flat_orientation,
            weight=1.5,
            params={
                "std": 0.45,  # math.sqrt(0.2) ≈ 0.45
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            },
        ),
        # Minimize body angular velocity (stay still)
        "body_ang_vel": RewardTermCfg(
            func=velocity_mdp.body_angular_velocity_penalty,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
        ),
        # Minimize angular momentum
        "angular_momentum": RewardTermCfg(
            func=velocity_mdp.angular_momentum_penalty,
            weight=-0.05,
            params={"sensor_name": "robot/root_angmom"},
        ),
        # CoM height target - maintain standing height
        "com_height_target": RewardTermCfg(
            func=microduck_mdp.com_height_target,
            weight=2.0,
            params={
                "target_height_min": 0.08,
                "target_height_max": 0.11,
            },
        ),
        # Regularization: Action smoothness
        "action_rate_l2": RewardTermCfg(
            func=velocity_mdp.action_rate_l2,
            weight=-0.5,  # Encourage smooth, minimal movements
        ),
        "leg_action_acceleration": RewardTermCfg(
            func=microduck_mdp.leg_action_acceleration_l2,
            weight=-0.05,
        ),
        "neck_action_acceleration": RewardTermCfg(
            func=microduck_mdp.neck_action_acceleration_l2,
            weight=-0.05,
        ),
        # Regularization: Joint torques
        "joint_torques_l2": RewardTermCfg(
            func=microduck_mdp.joint_torques_l2,
            weight=-1e-3,
        ),
        # Penalties
        "self_collisions": RewardTermCfg(
            func=velocity_mdp.self_collision_cost,
            weight=-10.0,
            params={"sensor_name": "self_collision"},
        ),
        "alive": RewardTermCfg(
            func=velocity_mdp.is_alive,
            weight=3.0,  # Increased to encourage survival via recovery steps
        ),
        "termination": RewardTermCfg(
            func=velocity_mdp.is_terminated,
            weight=-100.0,
        ),
    }

    ##
    # Terminations - Only fall detection
    ##

    # Keep only the basic termination from velocity config
    # Robot should learn to recover from perturbations

    ##
    # Events - Add domain randomization and perturbations
    ##

    # Domain randomization - same as imitation task
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

    if ENABLE_VELOCITY_PUSHES:
        # Velocity-based pushes for robustness training
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

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        cfg.events["randomize_imu_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_imu_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE,
            },
        )

    # Reset action history
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    ##
    # Curriculum - Gradually increase difficulty
    ##

    cfg.curriculum = {
        # Gradually increase velocity push magnitude for progressive robustness training
        "velocity_push_magnitude": CurriculumTermCfg(
            func=imitation_mdp.velocity_push_magnitude_curriculum,
            params={
                "event_name": "push_robot",
                "velocity_stages": [
                    {"step": 0, "velocity_range": (-0.6, 0.6)},  # Start gentle
                    {"step": 250 * 24, "velocity_range": (-0.8, 0.8)},
                    {"step": 500 * 24, "velocity_range": (-1.0, 1.0)},
                    {"step": 750 * 24, "velocity_range": (-1.2, 1.2)},
                    {"step": 1000 * 24, "velocity_range": (-1.4, 1.4)},
                ],
            },
        ),
    }

    return cfg


# RL configuration - same as imitation task
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

MicroduckStandingRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="standing",  # Shorter name for cleaner wandb run names
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,  # Standing task should converge faster
)
