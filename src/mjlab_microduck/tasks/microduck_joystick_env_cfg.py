"""Microduck joystick (motion tracking) environment configuration."""

from pathlib import Path

from mjlab.managers.manager_term_config import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from mjlab_microduck.tasks import joystick_mdp
from mjlab_microduck.tasks.joystick_command import JoystickCommandCfg
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg


def make_microduck_joystick_env_cfg(play: bool = False):
    """Create Microduck joystick (motion tracking) environment configuration.

    Starts from the base velocity config and replaces commands, observations, and rewards
    with motion tracking versions.

    Args:
        play: If True, disables observation corruption and extends episode length for visualization.

    Returns:
        Environment configuration for joystick motion tracking task.
    """

    # Start with base velocity config (includes all the domain randomization we want)
    cfg = make_microduck_velocity_env_cfg(play=play)

    # Determine motion file path
    motion_file = str(Path(__file__).parent.parent / "data" / "reference_motion.pkl")

    ##
    # Commands - Replace velocity command with joystick command
    ##

    cfg.commands = {
        "joystick": JoystickCommandCfg(
            entity_name="robot",
            motion_file=motion_file,
            resampling_time_range=(1.0e9, 1.0e9),  # Never resample (motion handles resets)
            debug_vis=True,  # Enable ghost visualization
            velocity_cmd_range={
                "x": (-0.1, 0.15),
                "y": (-0.15, 0.15),
                "yaw": (-1.0, 1.0),
            },
            sampling_mode="uniform" if not play else "adaptive",
        )
    }

    ##
    # Observations - Keep policy observations simple, add motion data to critic
    ##

    # Policy observations (what the robot can actually sense)
    policy_terms = {
        "command": ObservationTermCfg(
            func=joystick_mdp.velocity_command,
            params={"command_name": "joystick"},
        ),
        "phase": ObservationTermCfg(
            func=joystick_mdp.motion_phase,
            params={"command_name": "joystick"},
        ),
        # Keep the rest from base config
        **{
            k: v
            for k, v in cfg.observations["policy"].terms.items()
            if k not in ["command", "phase"]
        },
    }

    # Critic observations (privileged information including reference motion)
    critic_terms = {
        **policy_terms,
        "motion_root_pos_b": ObservationTermCfg(
            func=joystick_mdp.motion_root_pos_b,
            params={"command_name": "joystick"},
        ),
        "motion_root_ori_b": ObservationTermCfg(
            func=joystick_mdp.motion_root_ori_b,
            params={"command_name": "joystick"},
        ),
        "motion_joint_pos": ObservationTermCfg(
            func=joystick_mdp.motion_joint_pos,
            params={"command_name": "joystick"},
        ),
        "motion_joint_vel": ObservationTermCfg(
            func=joystick_mdp.motion_joint_vel,
            params={"command_name": "joystick"},
        ),
        "motion_joint_vel_error": ObservationTermCfg(
            func=joystick_mdp.motion_joint_vel_error,
            params={"command_name": "joystick"},
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
        "joystick_root_pos": RewardTermCfg(
            func=joystick_mdp.joystick_root_position_error_exp,
            weight=1.0,
            params={"command_name": "joystick", "std": 0.22},
        ),
        "joystick_root_ori": RewardTermCfg(
            func=joystick_mdp.joystick_root_orientation_error_exp,
            weight=1.0,
            params={"command_name": "joystick", "std": 0.22},
        ),
        "joystick_lin_vel": RewardTermCfg(
            func=joystick_mdp.joystick_linear_velocity_error_exp,
            weight=1.0,
            params={"command_name": "joystick", "std": 0.35},
        ),
        "joystick_ang_vel": RewardTermCfg(
            func=joystick_mdp.joystick_angular_velocity_error_exp,
            weight=1.0,
            params={"command_name": "joystick", "std": 0.7},
        ),
        "joystick_joint_pos_legs": RewardTermCfg(
            func=joystick_mdp.joystick_joint_position_error,
            weight=15.0,
            params={
                "command_name": "joystick",
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
        "joystick_joint_pos_non_legs": RewardTermCfg(
            func=joystick_mdp.joystick_joint_position_error,
            weight=100.0,
            params={
                "command_name": "joystick",
                "joint_names": ("neck_pitch", "head_pitch", "head_yaw", "head_roll"),
            },
        ),
        "foot_contact_match": RewardTermCfg(
            func=joystick_mdp.joystick_foot_contact_match,
            weight=1.0,
            params={"command_name": "joystick", "sensor_name": "feet_ground_contact"},
        ),
        # Regularization rewards (keep from base config)
        "action_rate_l2": RewardTermCfg(
            func=velocity_mdp.action_rate_l2,
            weight=-0.1,
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
            weight=-1e-5,
            params={"sensor_name": "feet_ground_contact"},
        ),
    }

    ##
    # Terminations - Add motion tracking terminations
    ##

    cfg.terminations["root_pos_error"] = TerminationTermCfg(
        func=joystick_mdp.bad_root_pos,
        params={"command_name": "joystick", "threshold": 0.15},
    )
    cfg.terminations["root_ori_error"] = TerminationTermCfg(
        func=joystick_mdp.bad_root_ori,
        params={"command_name": "joystick", "threshold": 0.8},
    )
    # Keep the fell_over termination from base config

    ##
    # Curriculum - Disable velocity curriculum (doesn't apply to motion tracking)
    ##

    cfg.curriculum = {}

    # Extend episode length for play mode
    if play:
        cfg.episode_length_s = 1e9

    return cfg
