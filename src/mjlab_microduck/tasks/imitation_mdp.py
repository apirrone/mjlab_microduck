"""
MDP functions for imitation motion tracking task.

Includes observations, rewards, and terminations for tracking reference motions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude, quat_apply_inverse, quat_mul, quat_inv, matrix_from_quat

from mjlab_microduck.tasks.imitation_command import ImitationCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ============================================================================
# Observations
# ============================================================================


def motion_root_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference root position error in robot's body frame.

    Returns the position difference between reference and robot root,
    expressed in the robot's local frame.
    """
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))

    # Position error in world frame
    pos_error_w = command.root_pos - command.robot_root_pos

    # Transform to robot body frame
    pos_error_b = quat_apply_inverse(command.robot_root_quat, pos_error_w)

    return pos_error_b.view(env.num_envs, -1)


def motion_root_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference root orientation error as rotation matrix columns.

    Returns the first two columns of the relative rotation matrix,
    representing the orientation error between reference and robot root.
    """
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))

    # Relative orientation: ref_quat * inv(robot_quat)
    quat_error = quat_mul(command.root_quat, quat_inv(command.robot_root_quat))
    mat = matrix_from_quat(quat_error)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_joint_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference joint positions."""
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    return command.joint_pos


def motion_joint_pos_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Joint position error (reference - robot)."""
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    return command.joint_pos - command.robot_joint_pos


def motion_joint_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference joint velocities."""
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    return command.joint_vel


def motion_joint_vel_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Joint velocity error (reference - robot)."""
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    return command.joint_vel - command.robot_joint_vel


def motion_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Current gait phase [0, 1)."""
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    return command.phase.unsqueeze(-1)


def velocity_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Velocity command (dx, dy, dtheta)."""
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    return torch.stack(
        [command.vel_cmd_x, command.vel_cmd_y, command.vel_cmd_yaw], dim=-1
    )


# ============================================================================
# Rewards
# ============================================================================


def imitation_root_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Reward for tracking reference root position."""
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    error = torch.sum(
        torch.square(command.root_pos - command.robot_root_pos), dim=-1
    )
    return torch.exp(-error / std**2)


def imitation_root_orientation_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Reward for tracking reference root orientation."""
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    error = quat_error_magnitude(command.root_quat, command.robot_root_quat) ** 2
    return torch.exp(-error / std**2)


def imitation_joint_position_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    joint_names: str | tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Negative squared error for tracking reference joint positions.

    Note: Returns negative values (cost). Consider using imitation_joint_position_error_exp
    for a reward in (0, 1] that increases as tracking improves.

    Args:
        env: The RL environment.
        command_name: Name of the imitation command term.
        joint_names: Optional joint name pattern(s) to consider (supports regex).
            If None, all joints are used. This allows weighting different joint groups
            separately (e.g., ".*_leg_.*" for legs, ".*head.*" for head).
    """
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    ref_pos = command.joint_pos
    robot_pos = command.robot_joint_pos
    if joint_names is not None:
        joint_ids, _ = command.robot.find_joints(joint_names, preserve_order=True)
        ref_pos = ref_pos[:, joint_ids]
        robot_pos = robot_pos[:, joint_ids]
    error = torch.sum(torch.square(ref_pos - robot_pos), dim=-1)
    return -error


def imitation_joint_position_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    joint_names: str | tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Exponential reward for tracking reference joint positions.

    Returns a reward in (0, 1] that increases toward 1 as error → 0.

    Args:
        env: The RL environment.
        command_name: Name of the imitation command term.
        std: Standard deviation for exponential scaling.
        joint_names: Optional joint name pattern(s) to consider (supports regex).
            If None, all joints are used.
    """
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    ref_pos = command.joint_pos
    robot_pos = command.robot_joint_pos
    if joint_names is not None:
        joint_ids, _ = command.robot.find_joints(joint_names, preserve_order=True)
        ref_pos = ref_pos[:, joint_ids]
        robot_pos = robot_pos[:, joint_ids]
    error = torch.sum(torch.square(ref_pos - robot_pos), dim=-1)
    return torch.exp(-error / std**2)


def imitation_joint_velocity_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    joint_names: str | tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Negative squared error for tracking reference joint velocities.

    Note: Returns negative values (cost). Consider using imitation_joint_velocity_error_exp
    for a reward in (0, 1] that increases as tracking improves.

    Args:
        env: The RL environment.
        command_name: Name of the imitation command term.
        joint_names: Optional joint name pattern(s) to consider (supports regex).
            If None, all joints are used. This allows weighting different joint groups
            separately.
    """
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    ref_vel = command.joint_vel
    robot_vel = command.robot_joint_vel
    if joint_names is not None:
        joint_ids, _ = command.robot.find_joints(joint_names, preserve_order=True)
        ref_vel = ref_vel[:, joint_ids]
        robot_vel = robot_vel[:, joint_ids]
    error = torch.sum(torch.square(ref_vel - robot_vel), dim=-1)
    return -error


def imitation_joint_velocity_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    joint_names: str | tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Exponential reward for tracking reference joint velocities.

    Returns a reward in (0, 1] that increases toward 1 as error → 0.

    Args:
        env: The RL environment.
        command_name: Name of the imitation command term.
        std: Standard deviation for exponential scaling.
        joint_names: Optional joint name pattern(s) to consider (supports regex).
            If None, all joints are used.
    """
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    ref_vel = command.joint_vel
    robot_vel = command.robot_joint_vel
    if joint_names is not None:
        joint_ids, _ = command.robot.find_joints(joint_names, preserve_order=True)
        ref_vel = ref_vel[:, joint_ids]
        robot_vel = robot_vel[:, joint_ids]
    error = torch.sum(torch.square(ref_vel - robot_vel), dim=-1)
    return torch.exp(-error / std**2)


def imitation_linear_velocity_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Reward for tracking reference world linear velocity."""
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    error = torch.sum(
        torch.square(command.world_linear_vel - command.robot_world_linear_vel), dim=-1
    )
    return torch.exp(-error / std**2)


def imitation_angular_velocity_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Reward for tracking reference world angular velocity."""
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    error = torch.sum(
        torch.square(command.world_angular_vel - command.robot_world_angular_vel), dim=-1
    )
    return torch.exp(-error / std**2)


def foot_clearance_reward(
    env: ManagerBasedRlEnv, command_name: str, sensor_name: str, target_height: float = 0.02
) -> torch.Tensor:
    """Reward for lifting feet during swing phase.

    When reference says foot should be OFF ground, reward if foot is high enough.
    This encourages proper foot lifting rather than dragging.

    Args:
        target_height: Minimum height (meters) for foot clearance during swing.
    """
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    sensor: ContactSensor = env.scene[sensor_name]

    # Reference foot contacts - which feet should be in swing phase
    ref_contact = command.foot_contacts > 0.5
    ref_swing = ~ref_contact  # Feet that should be lifted (num_envs, 2)

    # Contact forces - higher force means foot is on ground
    forces = sensor.data.found.squeeze(-1)  # (num_envs, 2)

    # During swing phase, reward for LOW contact force (foot lifted)
    # Use exponential reward: exp(-force) is high when force is low
    swing_clearance = torch.exp(-forces / 1.0)  # ~1.0 when foot lifted, ~0 when planted

    # Only apply reward during intended swing phase
    reward = (swing_clearance * ref_swing.float()).sum(dim=-1)

    return reward


def double_support_penalty(
    env: ManagerBasedRlEnv, command_name: str, sensor_name: str, force_threshold: float = 2.5
) -> torch.Tensor:
    """Penalty for having both feet on ground when one should be swinging.

    Double support should only happen briefly during transitions.
    This discourages dragging both feet simultaneously.
    """
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    sensor: ContactSensor = env.scene[sensor_name]

    # Check which feet are actually in contact
    forces = sensor.data.found.squeeze(-1)
    actual_contact = forces > force_threshold  # (num_envs, 2)

    # Reference contacts - how many feet should be on ground
    ref_contact = command.foot_contacts > 0.5
    ref_contact_count = ref_contact.float().sum(dim=-1)  # (num_envs,)

    # Actual contacts count
    actual_contact_count = actual_contact.float().sum(dim=-1)  # (num_envs,)

    # Penalize when more feet are down than reference wants
    # e.g., ref wants 1 foot, but robot has 2 feet down
    excess_contacts = torch.clamp(actual_contact_count - ref_contact_count, min=0.0)

    # Return negative penalty
    return -excess_contacts


def smooth_foot_forces(
    env: ManagerBasedRlEnv, sensor_name: str
) -> torch.Tensor:
    """Penalty for rapid changes in foot contact forces.

    Encourages smooth, gradual contact transitions instead of abrupt "tap tap" behavior.
    Penalizes the L2 norm of force changes between timesteps.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    forces = sensor.data.found.squeeze(-1)  # (num_envs, 2)

    # Initialize storage for previous forces if needed
    if not hasattr(env, '_prev_foot_forces'):
        env._prev_foot_forces = forces.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # Compute force rate of change
    force_rate = forces - env._prev_foot_forces  # (num_envs, 2)

    # Store current forces for next step
    env._prev_foot_forces = forces.clone()

    # Penalize large force changes (L2 norm across both feet)
    penalty = torch.sum(torch.square(force_rate), dim=-1)  # (num_envs,)

    return -penalty


def imitation_velocity_cmd_tracking_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Reward for tracking commanded velocities (dx, dy, dtheta).

    Velocity commands are in body frame (forward/left/yaw relative to robot),
    so we compare against body-frame robot velocities.
    """
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))

    # Use body-frame velocities since commands are relative to robot heading
    robot_lin_vel = command.robot.data.root_link_lin_vel_b
    robot_ang_vel = command.robot.data.root_link_ang_vel_b

    lin_x_err = (command.vel_cmd_x - robot_lin_vel[:, 0]) ** 2
    lin_y_err = (command.vel_cmd_y - robot_lin_vel[:, 1]) ** 2
    ang_yaw_err = (command.vel_cmd_yaw - robot_ang_vel[:, 2]) ** 2

    error = lin_x_err + lin_y_err + ang_yaw_err
    return torch.exp(-error / std**2)


def imitation_foot_contact_match(
    env: ManagerBasedRlEnv, command_name: str, sensor_name: str, force_threshold: float = 2.5, debug_print: bool = False
) -> torch.Tensor:
    """Reward for matching reference foot contacts.

    Returns 1.0 when actual foot contacts match reference, 0.0 otherwise.
    Each foot is evaluated separately and the mean is returned.

    Args:
        force_threshold: Minimum force (in Newtons) to consider as foot contact.
            Should be high enough to distinguish between "foot firmly planted"
            and "foot lightly dragging". Default 2.5 N (~35% of robot weight per foot).
        debug_print: If True, print contact info for env 0 every 10 steps.
    """
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None

    # Reference foot contacts from motion data (num_envs, 2)
    ref_contact = command.foot_contacts > 0.5

    # Actual foot contacts from sensor (num_envs, num_slots)
    # Note: Assuming left-right order matches motion data
    # Use force threshold to detect meaningful ground contact (not just dragging)
    forces = sensor.data.found.squeeze(-1)
    actual_contact = forces > force_threshold

    # Debug printing (only env 0, every 10 steps)
    if debug_print and hasattr(env, '_contact_debug_counter'):
        env._contact_debug_counter += 1
        if env._contact_debug_counter >= 10:
            env._contact_debug_counter = 0
            left_force = forces[0, 0].item()
            right_force = forces[0, 1].item()
            left_ref = "ON" if ref_contact[0, 0].item() else "OFF"
            right_ref = "ON" if ref_contact[0, 1].item() else "OFF"
            left_actual = "ON" if actual_contact[0, 0].item() else "OFF"
            right_actual = "ON" if actual_contact[0, 1].item() else "OFF"
            left_match = "✓" if ref_contact[0, 0] == actual_contact[0, 0] else "✗"
            right_match = "✓" if ref_contact[0, 1] == actual_contact[0, 1] else "✗"

            print(f"[Contacts] L: {left_force:5.2f}N ({left_actual}) ref={left_ref} {left_match}  |  "
                  f"R: {right_force:5.2f}N ({right_actual}) ref={right_ref} {right_match}")
    elif debug_print and not hasattr(env, '_contact_debug_counter'):
        env._contact_debug_counter = 0

    # Reward when contacts match
    match = (ref_contact == actual_contact).float()
    return match.sum(dim=-1)


# ============================================================================
# Terminations
# ============================================================================


def bad_root_pos(
    env: ManagerBasedRlEnv, command_name: str, threshold: float, curriculum_threshold_name: str | None = None
) -> torch.Tensor:
    """Terminate if robot root position deviates too much from reference.

    Args:
        curriculum_threshold_name: If provided, read threshold from curriculum instead of parameter
    """
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    error = torch.norm(command.root_pos - command.robot_root_pos, dim=-1)

    # Use curriculum threshold if available
    if curriculum_threshold_name and hasattr(env, '_curriculum_thresholds'):
        threshold = env._curriculum_thresholds.get(curriculum_threshold_name, threshold)

    return error > threshold


def bad_root_ori(
    env: ManagerBasedRlEnv, command_name: str, threshold: float, curriculum_threshold_name: str | None = None
) -> torch.Tensor:
    """Terminate if robot root orientation deviates too much from reference.

    Args:
        curriculum_threshold_name: If provided, read threshold from curriculum instead of parameter
    """
    command = cast(ImitationCommand, env.command_manager.get_term(command_name))
    error = quat_error_magnitude(command.root_quat, command.robot_root_quat)

    # Use curriculum threshold if available
    if curriculum_threshold_name and hasattr(env, '_curriculum_thresholds'):
        threshold = env._curriculum_thresholds.get(curriculum_threshold_name, threshold)

    return error > threshold


# ============================================================================
# Curriculum
# ============================================================================


def termination_threshold_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    threshold_param_name: str,
    threshold_stages: list[dict],
) -> torch.Tensor:
    """Update termination threshold based on training progress.

    Stores the threshold in the environment object for termination functions to read.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        threshold_param_name: Name to store threshold under (e.g., "root_pos_threshold")
        threshold_stages: List of dicts with 'step' and 'threshold' keys
            Example: [
                {"step": 0, "threshold": 0.15},
                {"step": 12000, "threshold": 0.3},
                {"step": 24000, "threshold": 1.0},  # Effectively disabled
            ]

    Returns:
        Current threshold value as a tensor
    """
    del env_ids  # Unused

    # Initialize threshold storage if needed
    if not hasattr(env, '_curriculum_thresholds'):
        env._curriculum_thresholds = {}

    # Find current threshold based on training progress
    current_threshold = threshold_stages[0]["threshold"]  # Default to first stage
    for stage in threshold_stages:
        if env.common_step_counter >= stage["step"]:
            current_threshold = stage["threshold"]

    # Store threshold in environment
    env._curriculum_thresholds[threshold_param_name] = current_threshold

    return torch.tensor([current_threshold])


def external_force_magnitude_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    magnitude_stages: list[dict],
) -> torch.Tensor:
    """Update external force magnitude based on training progress.

    Updates the force_range parameter of an external force event.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        event_name: Name of the event to update (e.g., "push_robot")
        magnitude_stages: List of dicts with 'step' and 'magnitude' keys
            Example: [
                {"step": 0, "magnitude": (0.0, 0.0)},  # Disabled
                {"step": 36000, "magnitude": (0.1, 0.5)},  # Enabled
            ]

    Returns:
        Current magnitude max value as a tensor
    """
    del env_ids  # Unused

    # Try to get the event term configuration
    try:
        event_term_cfg = env.event_manager.get_term_cfg(event_name)
    except ValueError:
        # Event not found, return 0.0
        return torch.tensor([0.0])

    # Find current magnitude based on training progress
    current_magnitude = magnitude_stages[0]["magnitude"]  # Default to first stage
    for stage in magnitude_stages:
        if env.common_step_counter >= stage["step"]:
            current_magnitude = stage["magnitude"]

    # Update the force_range parameter in the event term's params
    if "force_range" in event_term_cfg.params:
        event_term_cfg.params["force_range"] = (
            -current_magnitude[1],
            current_magnitude[1],
        )

    return torch.tensor([current_magnitude[1]])


def velocity_push_magnitude_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    velocity_stages: list[dict],
) -> torch.Tensor:
    """Update velocity push magnitude based on training progress.

    Updates the velocity_range parameter of a velocity push event.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        event_name: Name of the event to update (e.g., "push_robot")
        velocity_stages: List of dicts with 'step' and 'velocity_range' keys
            Example: [
                {"step": 0, "velocity_range": (-0.3, 0.3)},  # Gentle
                {"step": 36000, "velocity_range": (-0.5, 0.5)},  # Strong
            ]

    Returns:
        Current velocity range max value as a tensor
    """
    del env_ids  # Unused

    # Try to get the event term configuration
    try:
        event_term_cfg = env.event_manager.get_term_cfg(event_name)
    except ValueError:
        # Event not found, return 0.0
        return torch.tensor([0.0])

    # Find current velocity range based on training progress
    current_velocity = velocity_stages[0]["velocity_range"]  # Default to first stage
    for stage in velocity_stages:
        if env.common_step_counter >= stage["step"]:
            current_velocity = stage["velocity_range"]

    # Update the velocity_range parameter in the event term's params
    if "velocity_range" in event_term_cfg.params:
        event_term_cfg.params["velocity_range"]["x"] = current_velocity
        event_term_cfg.params["velocity_range"]["y"] = current_velocity

    return torch.tensor([current_velocity[1]])
