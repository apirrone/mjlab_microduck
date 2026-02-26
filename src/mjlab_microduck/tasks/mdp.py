"""MDP functions for microduck tasks"""

import numpy as np
import torch
from typing import TYPE_CHECKING, Optional
import mujoco

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.entity import Entity
from mjlab_microduck.reference_motion import ReferenceMotionLoader
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.envs.mdp.actions import JointPositionActionCfg as _JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Neck/head joint indices (neck_pitch=5, head_pitch=6, head_yaw=7, head_roll=8)
_NECK_JOINT_INDICES = list(range(5, 9))
# Time constant (seconds) for smooth offset interpolation toward target
_NECK_OFFSET_SMOOTHING_TAU = 0.5


class NeckOffsetJointPositionAction(_JointPositionActionCfg.class_type):
    """JointPositionAction that adds a random offset to neck/head joint targets.

    After the policy output is applied as joint position targets, adds
    env._neck_offset to the ctrl values for neck joints (indices 5–8).
    This trains robustness to external head movement and enables independent
    head control at deployment (add any offset on top of policy output).

    The offset smoothly follows env._neck_offset_target, which is updated by
    randomize_neck_offset_target() interval events.
    """

    def apply_actions(self) -> None:
        # Apply standard joint position control from policy output
        super().apply_actions()

        env = self._env

        # Initialize offset tensors on first call
        if not hasattr(env, "_neck_offset"):
            env._neck_offset = torch.zeros(env.num_envs, 4, device=env.device)
            env._neck_offset_target = torch.zeros(env.num_envs, 4, device=env.device)

        # Exponential smoothing: offset tracks target with time constant tau
        alpha = min(1.0, env.step_dt / _NECK_OFFSET_SMOOTHING_TAU)
        env._neck_offset.lerp_(env._neck_offset_target, alpha)

        # Add offset on top of the ctrl values already set by the action manager
        env.sim.data.ctrl[:, _NECK_JOINT_INDICES] += env._neck_offset


def reset_neck_offset(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
):
    """Reset neck joint offsets to zero at episode start."""
    if not hasattr(env, "_neck_offset"):
        env._neck_offset = torch.zeros(env.num_envs, 4, device=env.device)
        env._neck_offset_target = torch.zeros(env.num_envs, 4, device=env.device)

    if len(env_ids) > 0:
        env._neck_offset[env_ids] = 0.0
        env._neck_offset_target[env_ids] = 0.0


def randomize_neck_offset_target(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    max_offset: float = 0.3,
):
    """Sample new random neck offset targets (called at intervals).

    Draws uniform random targets in [-max_offset, max_offset] for each of the
    4 neck/head joints. The offset smoothly interpolates toward the new target.
    """
    if not hasattr(env, "_neck_offset_target"):
        env._neck_offset = torch.zeros(env.num_envs, 4, device=env.device)
        env._neck_offset_target = torch.zeros(env.num_envs, 4, device=env.device)

    if len(env_ids) > 0:
        env._neck_offset_target[env_ids] = (
            torch.rand(len(env_ids), 4, device=env.device) * 2 - 1
        ) * max_offset


class ImitationRewardState:
    """State for tracking imitation reward computation"""

    def __init__(self, ref_motion_loader: ReferenceMotionLoader):
        self.ref_motion_loader = ref_motion_loader
        self.phase = None  # Will be initialized as (num_envs,) tensor
        self.current_motion_idx = None  # (num_envs,) tensor of motion indices

    def initialize(self, num_envs: int, device: str):
        """Initialize phase tracking for each environment"""
        self.phase = torch.zeros(num_envs, device=device)
        self.current_motion_idx = torch.zeros(num_envs, dtype=torch.int32, device=device)

    def reset_phases(self, env_ids: torch.Tensor):
        """Reset phases for specific environments (e.g., after episode termination)

        Randomizes phase instead of always starting at 0.0 to improve push recovery.
        This forces the robot to learn to start walking from any point in the gait cycle.
        """
        if self.phase is not None and len(env_ids) > 0:
            # Randomize phase in [0, 1) for better generalization and push recovery
            self.phase[env_ids] = torch.rand(len(env_ids), device=self.phase.device)


def reset_action_history(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    imitation_state: Optional[ImitationRewardState] = None,
):
    """
    Reset cached action history for environments that are being reset.
    This is critical for action rate and acceleration penalty terms.

    This function should be called in the post_reset callback or at episode termination.

    Args:
        env: The environment
        env_ids: Indices of environments being reset
        asset_cfg: Asset configuration
        imitation_state: Optional imitation state to reset phase tracking
    """
    if len(env_ids) == 0:
        return

    asset: Entity = env.scene[asset_cfg.name]

    # Reset leg action rate cache
    if hasattr(env, '_prev_leg_actions'):
        # Set to current action (or zero if no action yet)
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            leg_joint_indices = list(range(0, 5)) + list(range(9, 14))
            env._prev_leg_actions[env_ids] = env.action_manager.action[env_ids][:, leg_joint_indices]
        else:
            env._prev_leg_actions[env_ids] = 0.0

    # Reset neck action rate cache
    if hasattr(env, '_prev_neck_actions'):
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            neck_joint_indices = list(range(5, 9))
            env._prev_neck_actions[env_ids] = env.action_manager.action[env_ids][:, neck_joint_indices]
        else:
            env._prev_neck_actions[env_ids] = 0.0

    # Reset leg action acceleration cache
    if hasattr(env, '_prev_leg_actions_for_acc'):
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            leg_joint_indices = list(range(0, 5)) + list(range(9, 14))
            current_action = env.action_manager.action[env_ids][:, leg_joint_indices]
            env._prev_leg_actions_for_acc[env_ids] = current_action
            env._prev_prev_leg_actions_for_acc[env_ids] = current_action
        else:
            env._prev_leg_actions_for_acc[env_ids] = 0.0
            env._prev_prev_leg_actions_for_acc[env_ids] = 0.0

    # Reset neck action acceleration cache
    if hasattr(env, '_prev_neck_actions_for_acc'):
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            neck_joint_indices = list(range(5, 9))
            current_action = env.action_manager.action[env_ids][:, neck_joint_indices]
            env._prev_neck_actions_for_acc[env_ids] = current_action
            env._prev_prev_neck_actions_for_acc[env_ids] = current_action
        else:
            env._prev_neck_actions_for_acc[env_ids] = 0.0
            env._prev_prev_neck_actions_for_acc[env_ids] = 0.0

    # Reset joint velocity cache for joint accelerations
    if hasattr(asset.data, '_prev_joint_vel'):
        # Get current joint velocities for reset environments
        joint_vel = asset.data.joint_vel[env_ids, :][:, asset_cfg.joint_ids]
        asset.data._prev_joint_vel[env_ids] = joint_vel

    # Reset contact frequency tracking
    if hasattr(env, '_contact_change_count'):
        env._contact_change_count[env_ids] = 0.0
    if hasattr(env, '_contact_change_timer'):
        env._contact_change_timer[env_ids] = 0.0
    if hasattr(env, '_prev_contacts_for_freq'):
        if "feet_ground_contact" in env.scene.sensors:
            contacts = env.scene.sensors["feet_ground_contact"].data.found[env_ids, :2]
            env._prev_contacts_for_freq[env_ids] = contacts

    # Reset foot force smoothness tracking
    if hasattr(env, '_prev_foot_forces'):
        if "feet_ground_contact" in env.scene.sensors:
            forces = env.scene.sensors["feet_ground_contact"].data.found[env_ids, :2].squeeze(-1)
            env._prev_foot_forces[env_ids] = forces

    # Reset imitation phase tracking
    if imitation_state is not None and imitation_state.phase is not None:
        imitation_state.reset_phases(env_ids)


def imitation_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    imitation_state: Optional[ImitationRewardState] = None,
    command_threshold: float = 0.01,
    weight_torso_pos_xy: float = 1.0,
    weight_torso_orient: float = 1.0,
    weight_lin_vel_xy: float = 1.0,
    weight_lin_vel_z: float = 1.0,
    weight_ang_vel_xy: float = 0.5,
    weight_ang_vel_z: float = 0.5,
    weight_leg_joint_pos: float = 15.0,
    weight_neck_joint_pos: float = 100.0,
    weight_leg_joint_vel: float = 1e-3,
    weight_neck_joint_vel: float = 1.0,
    weight_contact: float = 1.0,
) -> torch.Tensor:
    """
    Imitation reward based on reference motion tracking (BD-X paper structure)

    Args:
        env: The environment
        asset_cfg: Asset configuration
        imitation_state: State object holding reference motion loader and phase tracking
        command_threshold: Minimum command magnitude to apply reward
        weight_torso_pos_xy: Weight for torso position xy tracking
        weight_torso_orient: Weight for torso orientation tracking
        weight_lin_vel_xy: Weight for linear velocity xy tracking
        weight_lin_vel_z: Weight for linear velocity z tracking
        weight_ang_vel_xy: Weight for angular velocity xy tracking
        weight_ang_vel_z: Weight for angular velocity z tracking
        weight_leg_joint_pos: Weight for leg joint position tracking
        weight_neck_joint_pos: Weight for neck joint position tracking
        weight_leg_joint_vel: Weight for leg joint velocity tracking
        weight_neck_joint_vel: Weight for neck joint velocity tracking
        weight_contact: Weight for foot contact matching

    Returns:
        Reward tensor of shape (num_envs,)
    """
    if imitation_state is None or imitation_state.ref_motion_loader is None:
        return torch.zeros(env.num_envs, device=env.device)

    # Initialize phase tracking if needed
    if imitation_state.phase is None:
        imitation_state.initialize(env.num_envs, env.device)

    asset: Entity = env.scene[asset_cfg.name]

    # Get commanded velocity from the environment
    # Assuming velocity command exists with name "twist"
    if "twist" not in env.command_manager._terms:
        return torch.zeros(env.num_envs, device=env.device)

    cmd = env.command_manager.get_command("twist")
    cmd_vel = cmd[:, :3]  # (num_envs, 3) -> [vel_x, vel_y, ang_vel_z]
    cmd_norm = torch.linalg.norm(cmd_vel, dim=1)

    # Only reward when command is above threshold
    active_mask = cmd_norm > command_threshold

    # Find closest reference motion for each environment
    new_motion_indices = imitation_state.ref_motion_loader.find_closest_motion(cmd_vel)

    # Detect motion changes and reset phase when motion changes
    motion_changed = new_motion_indices != imitation_state.current_motion_idx
    imitation_state.phase[motion_changed] = 0.0
    imitation_state.current_motion_idx = new_motion_indices

    # Get periods for all environments
    periods = imitation_state.ref_motion_loader.get_period_batch(new_motion_indices)

    # Update phase for all environments
    dt = env.step_dt
    imitation_state.phase += dt / periods
    imitation_state.phase = torch.fmod(imitation_state.phase, 1.0)  # Keep in [0, 1]

    # Evaluate reference motions at current phases (batched per-environment)
    ref_data = imitation_state.ref_motion_loader.evaluate_motion_batch(
        new_motion_indices, imitation_state.phase, device=env.device
    )

    # Get current state
    # Joint positions and velocities (all 14 joints including head)
    # Joint order: left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
    #              neck_pitch, head_pitch, head_yaw, head_roll,
    #              right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
    joints_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joints_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # Separate leg joints (indices 0-4, 9-13) from neck joints (indices 5-8)
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))  # 10 leg joints
    neck_joint_indices = list(range(5, 9))  # 4 neck joints

    leg_joints_pos = joints_pos[:, leg_joint_indices]
    neck_joints_pos = joints_pos[:, neck_joint_indices]
    leg_joints_vel = joints_vel[:, leg_joint_indices]
    neck_joints_vel = joints_vel[:, neck_joint_indices]

    # Reference joint positions and velocities
    ref_joints_pos = ref_data["joints_pos"]
    ref_joints_vel = ref_data["joints_vel"]
    ref_leg_joints_pos = ref_joints_pos[:, leg_joint_indices]
    ref_neck_joints_pos = ref_joints_pos[:, neck_joint_indices]
    ref_leg_joints_vel = ref_joints_vel[:, leg_joint_indices]
    ref_neck_joints_vel = ref_joints_vel[:, neck_joint_indices]

    # Torso (base) position and orientation
    torso_pos_w = asset.data.root_link_pos_w  # (num_envs, 3) in world frame
    torso_quat_w = asset.data.root_link_quat_w  # (num_envs, 4) quaternion in world frame

    # Base velocities (world frame)
    base_lin_vel = asset.data.root_link_vel_w[:, :3]  # Linear velocity (first 3 components)
    base_ang_vel = asset.data.root_link_vel_w[:, 3:]  # Angular velocity (last 3 components)

    # Foot contacts
    if "feet_ground_contact" in env.scene.sensors:
        contacts = env.scene.sensors["feet_ground_contact"].data.found[:, :2]  # (num_envs, 2)
    else:
        contacts = torch.zeros((env.num_envs, 2), device=env.device)

    # Compute reward components (BD-X paper structure)

    # Torso position XY: exponential reward
    # Note: For periodic motions, torso position in reference is relative to path frame
    # For now, we compute this as zero since ref_data may not include absolute position
    # TODO: Add torso position tracking if reference motions include it
    torso_pos_xy_rew = torch.zeros(env.num_envs, device=env.device) * weight_torso_pos_xy

    # Torso orientation: exponential reward
    # TODO: Add quaternion difference computation if reference includes orientation
    torso_orient_rew = torch.zeros(env.num_envs, device=env.device) * weight_torso_orient

    # Linear velocity XY: exponential reward
    lin_vel_xy_rew = torch.exp(-8.0 * torch.sum(torch.square(base_lin_vel[:, :2] - ref_data["base_linear_vel"][:, :2]), dim=1)) * weight_lin_vel_xy

    # Linear velocity Z: exponential reward
    lin_vel_z_rew = torch.exp(-8.0 * torch.square(base_lin_vel[:, 2] - ref_data["base_linear_vel"][:, 2])) * weight_lin_vel_z

    # Angular velocity XY: exponential reward
    ang_vel_xy_rew = torch.exp(-2.0 * torch.sum(torch.square(base_ang_vel[:, :2] - ref_data["base_angular_vel"][:, :2]), dim=1)) * weight_ang_vel_xy

    # Angular velocity Z: exponential reward
    ang_vel_z_rew = torch.exp(-2.0 * torch.square(base_ang_vel[:, 2] - ref_data["base_angular_vel"][:, 2])) * weight_ang_vel_z

    # Leg joint positions: negative squared error (10 leg joints)
    leg_joint_pos_rew = -torch.sum(torch.square(leg_joints_pos - ref_leg_joints_pos), dim=1) * weight_leg_joint_pos

    # Neck joint positions: negative squared error (4 neck joints)
    neck_joint_pos_rew = -torch.sum(torch.square(neck_joints_pos - ref_neck_joints_pos), dim=1) * weight_neck_joint_pos

    # Leg joint velocities: negative squared error (10 leg joints)
    leg_joint_vel_rew = -torch.sum(torch.square(leg_joints_vel - ref_leg_joints_vel), dim=1) * weight_leg_joint_vel

    # Neck joint velocities: negative squared error (4 neck joints)
    neck_joint_vel_rew = -torch.sum(torch.square(neck_joints_vel - ref_neck_joints_vel), dim=1) * weight_neck_joint_vel

    # Contact reward: Σ_{i∈{L,R}} 1[c_i = ĉ_i] (simple binary match per foot)
    ref_contacts = (ref_data["foot_contacts"] > 0.5).float()
    contacts_float = contacts.float()

    # Compute per-foot contact matching (0 if mismatch, 1 if match)
    contact_matches = (contacts_float == ref_contacts).float()  # (num_envs, 2)

    # Sum matches across both feet (max value is 2.0)
    contact_rew = torch.sum(contact_matches, dim=1) * weight_contact

    # Total reward
    reward = (
        torso_pos_xy_rew
        + torso_orient_rew
        + lin_vel_xy_rew
        + lin_vel_z_rew
        + ang_vel_xy_rew
        + ang_vel_z_rew
        + leg_joint_pos_rew
        + neck_joint_pos_rew
        + leg_joint_vel_rew
        + neck_joint_vel_rew
        + contact_rew
    )

    # Apply mask: zero reward when command magnitude is below threshold
    reward = reward * active_mask.float()

    return reward


def imitation_phase_observation(
    env: ManagerBasedRlEnv,
    imitation_state: Optional[ImitationRewardState] = None,
) -> torch.Tensor:
    """
    Provide phase observation for imitation learning
    Returns [cos(phase * 2π), sin(phase * 2π)] encoding

    Args:
        env: The environment
        imitation_state: State object holding phase tracking

    Returns:
        Phase observation tensor of shape (num_envs, 2)
    """
    if imitation_state is None or imitation_state.phase is None:
        return torch.zeros((env.num_envs, 2), device=env.device)

    # Convert phase [0, 1] to angle [0, 2π]
    angle = imitation_state.phase * 2 * torch.pi

    # Return [cos, sin] encoding
    phase_obs = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)

    return phase_obs


def reference_motion_observation(
    env: ManagerBasedRlEnv,
    imitation_state: Optional[ImitationRewardState] = None,
) -> torch.Tensor:
    """
    Provide full reference motion as privileged observation for the critic

    Args:
        env: The environment
        imitation_state: State object holding reference motion loader and phase tracking

    Returns:
        Reference motion tensor of shape (num_envs, 36) containing:
        - joints_pos (14): Reference joint positions
        - joints_vel (14): Reference joint velocities
        - foot_contacts (2): Reference foot contact states
        - base_linear_vel (3): Reference base linear velocity
        - base_angular_vel (3): Reference base angular velocity
    """
    if imitation_state is None or imitation_state.ref_motion_loader is None:
        return torch.zeros((env.num_envs, 36), device=env.device)

    if imitation_state.phase is None:
        imitation_state.initialize(env.num_envs, env.device)

    # Get commanded velocity to find the reference motion
    if "twist" not in env.command_manager._terms:
        return torch.zeros((env.num_envs, 36), device=env.device)

    cmd = env.command_manager.get_command("twist")
    cmd_vel = cmd[:, :3]

    # Find closest reference motion for each environment
    motion_indices = imitation_state.ref_motion_loader.find_closest_motion(cmd_vel)

    # Evaluate reference motions at current phases
    ref_data = imitation_state.ref_motion_loader.evaluate_motion_batch(
        motion_indices, imitation_state.phase, device=env.device
    )

    # Concatenate all reference motion data
    ref_obs = torch.cat([
        ref_data["joints_pos"],       # 14
        ref_data["joints_vel"],       # 14
        ref_data["foot_contacts"],    # 2
        ref_data["base_linear_vel"],  # 3
        ref_data["base_angular_vel"], # 3
    ], dim=1)

    return ref_obs


def joint_accelerations_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize joint accelerations using L2 squared norm.
    Joint accelerations are computed using finite differences of joint velocities.

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,) - sum of squared joint accelerations
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get current joint velocities
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # Get previous joint velocities (stored in asset data)
    # Note: This assumes the environment stores previous joint velocities
    if not hasattr(asset.data, '_prev_joint_vel'):
        # Initialize on first call
        asset.data._prev_joint_vel = joint_vel.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # Compute joint accelerations using finite differences
    dt = env.step_dt
    joint_acc = (joint_vel - asset.data._prev_joint_vel) / dt

    # Store current velocities for next step
    asset.data._prev_joint_vel = joint_vel.clone()

    # Return L2 squared norm
    return torch.sum(torch.square(joint_acc), dim=1)


def leg_action_rate_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize the rate of change of leg actions (action_t - action_{t-1}).
    Leg joints are indices 0-4 and 9-13 (10 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get leg joint indices
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))

    # Get current and previous actions for leg joints only
    # Actions are stored in env (assuming the action is available)
    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    # Get the joint position action
    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    leg_actions = actions[:, leg_joint_indices]

    if not hasattr(env, '_prev_leg_actions'):
        env._prev_leg_actions = leg_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_rate = leg_actions - env._prev_leg_actions
    env._prev_leg_actions = leg_actions.clone()

    return torch.sum(torch.square(action_rate), dim=1)


def neck_action_rate_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize the rate of change of neck actions (action_t - action_{t-1}).
    Neck joints are indices 5-8 (4 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get neck joint indices
    neck_joint_indices = list(range(5, 9))

    # Get current and previous actions for neck joints only
    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    neck_actions = actions[:, neck_joint_indices]

    if not hasattr(env, '_prev_neck_actions'):
        env._prev_neck_actions = neck_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_rate = neck_actions - env._prev_neck_actions
    env._prev_neck_actions = neck_actions.clone()

    return torch.sum(torch.square(action_rate), dim=1)


def leg_action_acceleration_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize leg action accelerations (action_t - 2*action_{t-1} + action_{t-2}).
    Leg joints are indices 0-4 and 9-13 (10 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get leg joint indices
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))

    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    leg_actions = actions[:, leg_joint_indices]

    if not hasattr(env, '_prev_leg_actions_for_acc'):
        env._prev_leg_actions_for_acc = leg_actions.clone()
        env._prev_prev_leg_actions_for_acc = leg_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_acc = leg_actions - 2 * env._prev_leg_actions_for_acc + env._prev_prev_leg_actions_for_acc

    env._prev_prev_leg_actions_for_acc = env._prev_leg_actions_for_acc.clone()
    env._prev_leg_actions_for_acc = leg_actions.clone()

    return torch.sum(torch.square(action_acc), dim=1)


def neck_action_acceleration_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize neck action accelerations (action_t - 2*action_{t-1} + action_{t-2}).
    Neck joints are indices 5-8 (4 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get neck joint indices
    neck_joint_indices = list(range(5, 9))

    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    neck_actions = actions[:, neck_joint_indices]

    if not hasattr(env, '_prev_neck_actions_for_acc'):
        env._prev_neck_actions_for_acc = neck_actions.clone()
        env._prev_prev_neck_actions_for_acc = neck_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_acc = neck_actions - 2 * env._prev_neck_actions_for_acc + env._prev_prev_neck_actions_for_acc

    env._prev_prev_neck_actions_for_acc = env._prev_neck_actions_for_acc.clone()
    env._prev_neck_actions_for_acc = neck_actions.clone()

    return torch.sum(torch.square(action_acc), dim=1)


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    """
    Reward for staying alive (not terminated)

    Args:
        env: The environment

    Returns:
        Reward tensor of shape (num_envs,) - ones for all envs
    """
    return torch.ones(env.num_envs, device=env.device)


def com_height_target(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    target_height_min: float = 0.1,
    target_height_max: float = 0.15,
) -> torch.Tensor:
    """
    Reward for keeping the center of mass within a target height range.
    Returns positive reward when in range, negative penalty when outside.

    Args:
        env: The environment
        asset_cfg: Asset configuration
        target_height_min: Minimum target height for CoM (meters)
        target_height_max: Maximum target height for CoM (meters)

    Returns:
        Reward tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get center of mass height (z position of root link)
    com_height = asset.data.root_link_pos_w[:, 2]

    # Reward when in range, penalty when outside
    # Use smooth penalty that increases quadratically with distance from range
    below_min = com_height < target_height_min
    above_max = com_height > target_height_max
    in_range = ~(below_min | above_max)

    # Compute penalties for being outside range
    penalty_below = torch.square(com_height - target_height_min) * below_min.float()
    penalty_above = torch.square(com_height - target_height_max) * above_max.float()

    # Reward: +1 when in range, -squared_distance when outside
    reward = in_range.float() - (penalty_below + penalty_above)

    return reward


def neck_joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize neck joint velocities to keep head stable.
    Neck joints are indices 5-8 (4 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get neck joint indices (neck_pitch, head_pitch, head_yaw, head_roll)
    neck_joint_indices = list(range(5, 9))

    # Get joint velocities for neck joints
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    neck_joint_vel = joint_vel[:, neck_joint_indices]

    # Return L2 squared norm of neck joint velocities
    return torch.sum(torch.square(neck_joint_vel), dim=1)


def leg_joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize leg joint velocities to encourage smoother, less dynamic motion.
    Leg joints are indices 0-4 and 9-13 (10 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get leg joint indices (left hip-ankle: 0-4, right hip-ankle: 9-13)
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))

    # Get joint velocities for leg joints
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    leg_joint_vel = joint_vel[:, leg_joint_indices]

    # Return L2 squared norm of leg joint velocities
    return torch.sum(torch.square(leg_joint_vel), dim=1)

def joint_torques_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize actuator forces (torques) to encourage energy-efficient motion.

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,) - sum of squared actuator forces
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get actuator forces (scalar actuation in actuation space)
    actuator_forces = asset.data.actuator_force

    # Return L2 squared norm
    return torch.sum(torch.square(actuator_forces), dim=1)


def contact_frequency_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    max_contact_changes_per_sec: float = 4.0,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """
    Penalize high frequency of contact changes to encourage slower stepping.
    Tracks the number of contact state changes per second and penalizes when above threshold.

    Args:
        env: The environment
        sensor_name: Name of the contact sensor
        max_contact_changes_per_sec: Maximum allowed contact changes per second
        command_threshold: Minimum command magnitude to apply penalty

    Returns:
        Penalty tensor of shape (num_envs,) - negative when exceeding threshold
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    # Check if command is above threshold
    if "twist" in env.command_manager._terms:
        cmd = env.command_manager.get_command("twist")
        cmd_vel = cmd[:, :3]
        cmd_norm = torch.linalg.norm(cmd_vel, dim=1)
        active_mask = cmd_norm > command_threshold
    else:
        active_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    sensor = env.scene.sensors[sensor_name]
    contacts = sensor.data.found[:, :2]  # (num_envs, 2)

    # Initialize tracking if needed
    if not hasattr(env, '_contact_change_count'):
        env._contact_change_count = torch.zeros(env.num_envs, device=env.device)
        env._contact_change_timer = torch.zeros(env.num_envs, device=env.device)
        env._prev_contacts_for_freq = contacts.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # Detect any contact changes (either foot)
    contact_changed = torch.any(contacts != env._prev_contacts_for_freq, dim=1)

    # Increment change counter
    env._contact_change_count += contact_changed.float()

    # Update timer
    env._contact_change_timer += env.step_dt

    # Calculate current frequency (changes per second)
    # Avoid division by zero
    freq = env._contact_change_count / torch.clamp(env._contact_change_timer, min=0.01)

    # Reset counter and timer every 1 second
    reset_mask = env._contact_change_timer >= 1.0
    env._contact_change_count[reset_mask] = 0.0
    env._contact_change_timer[reset_mask] = 0.0

    # Penalize when frequency exceeds maximum
    # Use quadratic penalty for frequencies above threshold
    excess_freq = torch.clamp(freq - max_contact_changes_per_sec, min=0.0)
    penalty = -torch.square(excess_freq)

    # Update previous contacts
    env._prev_contacts_for_freq = contacts.clone()

    # Apply command threshold mask
    penalty = penalty * active_mask.float()

    return penalty


# ==============================================================================
# Domain Randomization Events
# ==============================================================================


def randomize_delayed_actuator_gains(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    operation: str = "scale",
):
    """Randomize PD gains for DelayedActuator (which wraps XmlPositionActuator).

    Args:
        env: The environment
        env_ids: Environment IDs to randomize (None = all envs)
        kp_range: (min, max) for kp randomization
        kd_range: (min, max) for kd randomization
        asset_cfg: Asset configuration
        operation: "scale" or "abs"
    """
    from mjlab.actuator.delayed_actuator import DelayedActuator
    from mjlab.actuator import XmlPositionActuator

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    # Store original gains on first call
    if not hasattr(env, '_original_actuator_gains'):
        env._original_actuator_gains = {}

    # Apply to actuators
    for actuator in asset.actuators:
        # Handle DelayedActuator wrapping XmlPositionActuator
        if isinstance(actuator, DelayedActuator):
            base_actuator = actuator._base_actuator
        else:
            base_actuator = actuator

        # Get control IDs
        ctrl_ids = base_actuator.ctrl_ids

        # Store original values on first call (use tuple of ctrl_ids as key)
        ctrl_key = tuple(ctrl_ids.tolist())
        if ctrl_key not in env._original_actuator_gains:
            # Store a copy of the original values for env 0 (they're the same for all envs initially)
            env._original_actuator_gains[ctrl_key] = {
                'gainprm': env.sim.model.actuator_gainprm[0, ctrl_ids, 0].clone(),
                'biasprm1': env.sim.model.actuator_biasprm[0, ctrl_ids, 1].clone(),
                'biasprm2': env.sim.model.actuator_biasprm[0, ctrl_ids, 2].clone(),
            }

        # Reset to original values first (to prevent accumulation)
        original = env._original_actuator_gains[ctrl_key]
        env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = original['gainprm'].unsqueeze(0).expand(len(env_ids), -1)
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = original['biasprm1'].unsqueeze(0).expand(len(env_ids), -1)
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = original['biasprm2'].unsqueeze(0).expand(len(env_ids), -1)

        # Sample random gains for each env and each control
        kp_samples = torch.rand(len(env_ids), len(ctrl_ids), device=env.device) * (kp_range[1] - kp_range[0]) + kp_range[0]
        kd_samples = torch.rand(len(env_ids), len(ctrl_ids), device=env.device) * (kd_range[1] - kd_range[0]) + kd_range[0]

        # For XmlPositionActuator, modify MuJoCo model parameters directly
        if isinstance(base_actuator, XmlPositionActuator):
            if operation == "scale":
                # Scale the ORIGINAL (now-reset) values
                env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] *= kp_samples
                env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] *= kp_samples
                env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] *= kd_samples
            elif operation == "abs":
                env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = kp_samples
                env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = -kp_samples
                env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = -kd_samples


def randomize_mass_and_inertia(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Randomize body mass and inertia together with the same scaling factor.

    This maintains physical consistency - mass and inertia must scale together
    to avoid creating invalid inertia tensors that cause simulation instability.

    Args:
        env: The environment
        env_ids: Environment IDs to randomize
        scale_range: (min, max) scaling factor applied to both mass and inertia
        asset_cfg: Asset configuration specifying which bodies to randomize
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    # Get body indices
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))[body_ids]
    body_indices = asset.indexing.body_ids[body_ids]

    # Sample ONE random scale per environment (applied to both mass and inertia)
    num_envs = len(env_ids)
    num_bodies = len(body_indices)
    scales = torch.rand(num_envs, num_bodies, device=env.device) * (scale_range[1] - scale_range[0]) + scale_range[0]

    # Store original values on first call
    if not hasattr(env, '_original_mass_inertia'):
        env._original_mass_inertia = {
            'mass': env.sim.model.body_mass[0, body_indices].clone(),
            'inertia': env.sim.model.body_inertia[0, body_indices].clone(),
        }

    # Reset to original first (to prevent accumulation)
    original = env._original_mass_inertia
    env.sim.model.body_mass[env_ids[:, None], body_indices] = original['mass'].unsqueeze(0).expand(num_envs, -1)
    env.sim.model.body_inertia[env_ids[:, None], body_indices] = original['inertia'].unsqueeze(0).expand(num_envs, -1, -1)

    # Apply same scale to both mass and inertia
    env.sim.model.body_mass[env_ids[:, None], body_indices] *= scales
    env.sim.model.body_inertia[env_ids[:, None], body_indices] *= scales.unsqueeze(-1)  # Scale all 3 inertia components


def standing_envs_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    standing_stages: list[dict],
) -> torch.Tensor:
    """Update the relative number of standing environments based on training progress.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        command_name: Name of the velocity command term
        standing_stages: List of dicts with 'step' and 'rel_standing_envs' keys
            Example: [
                {"step": 0, "rel_standing_envs": 0.02},
                {"step": 1000, "rel_standing_envs": 0.1},
                {"step": 2000, "rel_standing_envs": 0.2},
            ]

    Returns:
        Current rel_standing_envs value as a tensor
    """
    del env_ids  # Unused

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    from typing import cast

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"

    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    # Update rel_standing_envs based on current step
    for stage in standing_stages:
        if env.common_step_counter > stage["step"]:
            cfg.rel_standing_envs = stage["rel_standing_envs"]

    return torch.tensor([cfg.rel_standing_envs])


def velocity_tracking_std_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    std_stages: list[dict],
) -> torch.Tensor:
    """Update velocity tracking std parameter based on training progress.

    Starts with loose std (easy rewards) to learn basic walking, then gradually
    tightens to improve velocity tracking accuracy.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        reward_name: Name of the reward term (e.g., "track_linear_velocity")
        std_stages: List of dicts with 'step' and 'std' keys
            Example: [
                {"step": 0, "std": 0.5},      # Start loose - learn to walk
                {"step": 250, "std": 0.3},     # Moderate - refine gait
                {"step": 500, "std": 0.2},     # Strict - accurate tracking
            ]

    Returns:
        Current std value as a tensor
    """
    del env_ids  # Unused

    # Get reward term configuration
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)

    # Update std based on current step
    current_std = std_stages[0]["std"]  # Default to first stage

    for stage in std_stages:
        if env.common_step_counter > stage["step"]:
            current_std = stage["std"]

    # Update the reward term's std parameter
    reward_term_cfg.params["std"] = current_std

    return torch.tensor([current_std])


def push_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    push_stages: list[dict],
) -> torch.Tensor:
    """Update push velocity range based on training progress.

    Starts with no/small pushes to learn clean walking, then gradually increases
    to build robustness without disrupting early learning.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        event_name: Name of the push event term (e.g., "push_robot")
        push_stages: List of dicts with 'step' and 'velocity_range' keys
            Example: [
                {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"step": 250, "velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
                {"step": 500, "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
            ]

    Returns:
        Current max push magnitude as a tensor
    """
    del env_ids  # Unused

    # Access event configuration directly from environment config
    assert event_name in env.cfg.events, f"Event '{event_name}' not found"
    event_cfg = env.cfg.events[event_name]

    # Update velocity_range based on current step
    current_range = push_stages[0]["velocity_range"]  # Default to first stage

    for stage in push_stages:
        if env.common_step_counter > stage["step"]:
            current_range = stage["velocity_range"]

    # Update the event configuration's velocity_range parameter
    event_cfg.params["velocity_range"] = current_range

    # Return max magnitude for logging
    max_push = max(abs(current_range["x"][0]), abs(current_range["x"][1]))
    return torch.tensor([max_push])


def neck_offset_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    offset_stages: list[dict],
) -> torch.Tensor:
    """Update neck offset magnitude based on training progress.

    Gradually increases the max random neck offset so the robot first learns
    to walk with no head disturbance, then progressively harder ones.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        event_name: Name of the neck offset event (e.g., "randomize_neck_offset_target")
        offset_stages: List of dicts with 'step' and 'max_offset' keys
            Example: [
                {"step": 0,         "max_offset": 0.0},
                {"step": 500 * 24,  "max_offset": 0.1},
                {"step": 1000 * 24, "max_offset": 0.2},
                {"step": 1500 * 24, "max_offset": 0.3},
            ]

    Returns:
        Current max_offset value as a tensor (for logging)
    """
    del env_ids  # Unused

    assert event_name in env.cfg.events, f"Event '{event_name}' not found"
    event_cfg = env.cfg.events[event_name]

    current_offset = offset_stages[0]["max_offset"]
    for stage in offset_stages:
        if env.common_step_counter > stage["step"]:
            current_offset = stage["max_offset"]

    event_cfg.params["max_offset"] = current_offset
    return torch.tensor([current_offset])


def velocity_command_ranges_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[dict],
) -> torch.Tensor:
    """Update velocity command ranges based on training progress.

    Gradually increases the commanded velocity ranges to allow the robot to learn
    higher speeds progressively. Starts with smaller ranges for stable learning,
    then expands to more challenging velocities.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        command_name: Name of the velocity command term (e.g., "twist")
        velocity_stages: List of dicts with 'step', 'lin_vel_range', and 'ang_vel_range' keys
            Example: [
                {"step": 0, "lin_vel_range": 0.3, "ang_vel_range": 1.5},
                {"step": 500 * 24, "lin_vel_range": 0.4, "ang_vel_range": 1.75},
                {"step": 1000 * 24, "lin_vel_range": 0.5, "ang_vel_range": 2.0},
            ]

    Returns:
        Current max linear velocity as a tensor
    """
    del env_ids  # Unused

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    from typing import cast

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"

    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    # Update velocity ranges based on current step
    current_lin_vel = velocity_stages[0]["lin_vel_range"]
    current_ang_vel = velocity_stages[0]["ang_vel_range"]

    for stage in velocity_stages:
        if env.common_step_counter > stage["step"]:
            current_lin_vel = stage["lin_vel_range"]
            current_ang_vel = stage["ang_vel_range"]

    # Update command ranges (symmetric around zero)
    cfg.ranges.lin_vel_x = (-current_lin_vel, current_lin_vel)
    cfg.ranges.lin_vel_y = (-current_lin_vel, current_lin_vel)
    cfg.ranges.ang_vel_z = (-current_ang_vel, current_ang_vel)

    return torch.tensor([current_lin_vel])


def projected_gravity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Projected gravity vector in body frame.

    Returns the gravity vector projected into the robot's body frame,
    representing pure orientation without linear acceleration.
    This is simpler than raw accelerometer and only depends on orientation.

    Returns:
        torch.Tensor: Projected gravity in body frame (num_envs, 3)
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def raw_accelerometer(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Raw accelerometer reading (includes gravity + linear acceleration).

    Returns normalized raw accelerometer which mimics what a real IMU measures.
    This is different from pure projected_gravity which only reflects orientation.
    Reads from the MuJoCo accelerometer sensor "imu_accel".

    Returns:
        torch.Tensor: Normalized raw accelerometer reading (num_envs, 3)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Access the model to find the sensor address
    # The accelerometer sensor is the 5th sensor (index 4) in robot.xml
    # Sensors: framequat, gyro, gyro, velocimeter, accelerometer, subtreeangmom
    mj_model = asset.data.model

    # Get sensor address from model arrays (sensor_adr is torch tensor)
    sensor_adr_array = mj_model.sensor_adr  # This is a TorchArray/tensor
    sensor_id = 4  # imu_accel is the 5th sensor (0-indexed)
    sensor_adr = int(sensor_adr_array[sensor_id].item())  # Convert to Python int

    # Read accelerometer data (specific force measured by sensor)
    # Shape: (num_envs, 3)
    accel_raw = asset.data.data.sensordata[:, sensor_adr:sensor_adr+3]

    # MuJoCo accelerometer measures specific force (like real sensor)
    # Negate to match convention: when at rest upright, should point down
    accel_negated = -accel_raw

    # Normalize to unit vector
    accel_norm = torch.norm(accel_negated, dim=-1, keepdim=True)
    accel_normalized = torch.where(
        accel_norm > 0.1,
        accel_negated / accel_norm,
        asset.data.projected_gravity_b  # Fallback to projected gravity
    )

    return accel_normalized

def randomize_imu_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    max_angle_deg: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Randomize IMU sensor mounting orientation by small angles.
    
    Simulates slight mounting errors or calibration offsets in the real robot.
    The IMU orientation is randomized by rotating around random axes by up to max_angle_deg.
    
    Args:
        env: The environment
        env_ids: Environment IDs to randomize
        max_angle_deg: Maximum rotation angle in degrees (default 2.0°)
        asset_cfg: Asset configuration
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)
    
    asset: Entity = env.scene[asset_cfg.name]

    # IMU site is the first site (index 0) in robot.xml
    # Sites: imu (0), left_foot (1), right_foot (2)
    site_id = 0
    
    # Store original orientation on first call
    if not hasattr(env, '_original_imu_quat'):
        env._original_imu_quat = env.sim.model.site_quat[0, site_id].clone()
    
    # Generate random rotations for each environment
    num_envs = len(env_ids)
    max_angle_rad = max_angle_deg * torch.pi / 180.0
    
    # Random rotation angles [-max_angle, +max_angle] for each axis
    angles = (torch.rand(num_envs, 3, device=env.device) * 2 - 1) * max_angle_rad
    
    # Convert Euler angles to quaternions (small angle approximation for efficiency)
    # For small angles: quat ≈ [1, θx/2, θy/2, θz/2]
    half_angles = angles / 2.0
    quats_delta = torch.zeros(num_envs, 4, device=env.device)
    quats_delta[:, 0] = 1.0  # w component
    quats_delta[:, 1:] = half_angles  # x, y, z components
    
    # Normalize the quaternion
    quats_delta = quats_delta / torch.norm(quats_delta, dim=1, keepdim=True)
    
    # Get original quaternion and apply delta rotation
    original_quat = env._original_imu_quat.unsqueeze(0).expand(num_envs, -1)
    
    # Quaternion multiplication: q_new = q_delta * q_original
    # q1 * q2 = [w1*w2 - dot(v1,v2), w1*v2 + w2*v1 + cross(v1,v2)]
    w1, x1, y1, z1 = quats_delta[:, 0], quats_delta[:, 1], quats_delta[:, 2], quats_delta[:, 3]
    w2, x2, y2, z2 = original_quat[:, 0], original_quat[:, 1], original_quat[:, 2], original_quat[:, 3]
    
    new_quat = torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
    ], dim=1)
    
    # Apply to the selected environments
    env.sim.model.site_quat[env_ids, site_id] = new_quat


def standing_phase(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Simple time-based phase for standing task.

    Returns a scalar phase value that cycles from 0 to 1 based on time.
    This allows the policy to have a sense of time progression even when standing.

    Args:
        env: The RL environment
        asset_cfg: Not used, but kept for API consistency

    Returns:
        Phase value [0, 1] as tensor of shape (num_envs, 1)
    """
    # Simple time-based phase that cycles every 2 seconds
    # This gives the policy a time-varying signal
    phase_period = 2.0  # seconds
    time = env.episode_length_buf * env.step_dt
    phase = (time % phase_period) / phase_period

    return phase.unsqueeze(-1)  # Shape: (num_envs, 1)


def air_time_adaptive(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.01,    # below this: no reward (standing)
    running_threshold: float = 0.5,     # above this: use running air-time window
    walk_threshold_min: float = 0.10,
    walk_threshold_max: float = 0.25,
    run_threshold_min: float = 0.05,
    run_threshold_max: float = 0.25,
) -> torch.Tensor:
    """Air-time reward with separate swing-time windows for walking vs running.

    - command < command_threshold  → 0 (standing, no reward)
    - command_threshold–running_threshold → walk window [walk_min, walk_max]
    - command > running_threshold  → run  window [run_min,  run_max]

    This lets the walking gait keep its deliberate 100–250 ms swing while
    running can use a faster 50–250 ms cadence.
    """
    sensor = env.scene.sensors[sensor_name]
    current_air_time = sensor.data.current_air_time  # (num_envs, num_feet)
    assert current_air_time is not None

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])

    is_walking = ((total_speed >= command_threshold) & (total_speed < running_threshold)).float()  # (num_envs,)
    is_running = (total_speed >= running_threshold).float()

    # Per-env thresholds broadcast over feet
    tmin = (is_walking * walk_threshold_min + is_running * run_threshold_min).unsqueeze(1)
    tmax = (is_walking * walk_threshold_max + is_running * run_threshold_max).unsqueeze(1)

    in_range = (current_air_time > tmin) & (current_air_time < tmax)
    reward = torch.sum(in_range.float(), dim=1)  # sum over feet

    # Zero reward when standing
    active = (total_speed >= command_threshold).float()
    return reward * active


def stillness_at_zero_command(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    vel_std: float = 0.1,
) -> torch.Tensor:
    """Reward staying still when command is near zero.

    Returns exp(-body_vel² / vel_std²) when command < threshold, else 0.
    This is monotonically decreasing with body speed — moving faster is always
    less rewarding. There is no threshold the robot can cross to 'escape' it,
    unlike gate-based stepping penalties.
    """
    asset: Entity = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    body_vel = torch.norm(asset.data.root_link_vel_w[:, :2], dim=1)
    stillness = torch.exp(-body_vel ** 2 / vel_std ** 2)

    return is_standing_cmd * stillness


def joint_vel_l2_when_standing(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Penalise leg joint velocities only when command is near zero.

    Targets the standing-shake problem: the policy makes rapid oscillating
    corrections around the home pose when standing. Gated on command so it
    does not affect the walking gait at all.
    """
    asset: Entity = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    leg_indices = list(range(0, 5)) + list(range(9, 14))
    joint_vel = asset.data.joint_vel[:, leg_indices]
    vel_sq = torch.sum(joint_vel ** 2, dim=-1)

    return is_standing_cmd * vel_sq


def foot_step_penalty_when_standing(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    body_vel_threshold: float = 0.2,
    air_time_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalise stepping when at zero command and the body is not being pushed.

    Symmetric counterpart to the air_time reward:
    - air_time gives  +reward for stepping when command > threshold  (walk)
    - this gives      -reward for stepping when command < threshold  (stand)

    The body-velocity gate prevents penalising recovery steps after a push:
    if the robot is already moving fast (pushed), no penalty is applied so it
    can still take steps to catch itself.

    Returns a value in [0, 1] (use a negative weight in the config).
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors["feet_ground_contact"]

    # Was either foot recently lifted? (last completed air phase > threshold)
    air_time = contact_sensor.data.last_air_time[:, :2]  # (num_envs, 2)
    any_foot_stepped = (air_time > air_time_threshold).any(dim=1).float()

    # Are we in standing mode? (command near zero)
    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing = (total_speed < command_threshold).float()

    # Is the body still? (not being pushed)
    body_vel = torch.norm(asset.data.root_link_vel_w[:, :2], dim=1)
    is_still = (body_vel < body_vel_threshold).float()

    return any_foot_stepped * is_standing * is_still


def recovery_stepping_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    velocity_threshold: float = 0.3,
    air_time_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward foot air time only when at zero command AND robot has high velocity (recovering from push).

    This encourages the robot to take steps to recover balance when pushed,
    but does NOT fire during normal walking (command > threshold).

    Args:
        env: The RL environment
        asset_cfg: Asset configuration (unused but kept for API consistency)
        command_name: Name of the velocity command in the command manager
        command_threshold: Speed below which the robot is considered to be in standing mode
        velocity_threshold: Linear velocity threshold to activate stepping reward (m/s)
        air_time_threshold: Minimum air time to count as a step (seconds)

    Returns:
        Reward tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Only fire for standing envs (command near zero)
    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    # Get base linear velocity magnitude
    base_lin_vel = asset.data.root_link_vel_w[:, :3]  # (num_envs, 3)
    vel_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)  # Only XY plane

    # Only reward stepping when velocity is high (being pushed)
    should_step = vel_magnitude > velocity_threshold

    # Get foot air time from contact sensor
    contact_sensor = env.scene.sensors["feet_ground_contact"]
    air_time = contact_sensor.data.last_air_time[:, :2]  # (num_envs, 2) - left and right foot

    # Reward if either foot has been in air recently
    foot_in_air = (air_time > air_time_threshold).any(dim=1)  # (num_envs,)

    # Only give reward when: standing command AND high body velocity AND foot stepped
    reward = is_standing_cmd * should_step.float() * foot_in_air.float()

    return reward


def adaptive_pose_weight(
    env: ManagerBasedRlEnv,
    base_pose_reward: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    velocity_threshold: float = 0.3,
    min_weight: float = 0.3,
) -> torch.Tensor:
    """Reduce pose tracking weight when robot has high velocity (recovering from push).

    This gives the robot freedom to deviate from the standing pose when taking
    recovery steps, while maintaining strict pose tracking when standing still.

    Args:
        env: The RL environment
        base_pose_reward: The original pose reward (before weighting)
        asset_cfg: Asset configuration (unused but kept for API consistency)
        velocity_threshold: Linear velocity threshold to start reducing weight (m/s)
        min_weight: Minimum weight multiplier (0-1) at high velocities

    Returns:
        Weighted reward tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get base linear velocity magnitude
    base_lin_vel = asset.data.root_link_vel_w[:, :3]  # (num_envs, 3)
    vel_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)  # Only XY plane

    # Compute weight: 1.0 when stationary, min_weight at high velocity
    # Use smooth transition via sigmoid-like function
    weight = min_weight + (1.0 - min_weight) * torch.exp(
        -((vel_magnitude - velocity_threshold) / velocity_threshold).clamp(min=0.0) ** 2
    )

    return base_pose_reward * weight


def randomize_base_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    max_pitch_deg: float = 10.0,
    max_roll_deg: float = 5.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Randomize base orientation at episode start to force reactive behavior.

    Adds random pitch and roll to the robot's base orientation at the start of
    each episode. This prevents the policy from memorizing a single initial state
    and forces it to use feedback to adapt to different orientations.

    Args:
        env: The environment
        env_ids: Environment IDs to randomize
        max_pitch_deg: Maximum pitch angle in degrees (forward/backward tilt)
        max_roll_deg: Maximum roll angle in degrees (side-to-side tilt)
        asset_cfg: Asset configuration
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    num_envs = len(env_ids)

    # Generate random pitch and roll angles
    max_pitch_rad = max_pitch_deg * torch.pi / 180.0
    max_roll_rad = max_roll_deg * torch.pi / 180.0

    pitch = (torch.rand(num_envs, device=env.device) * 2 - 1) * max_pitch_rad
    roll = (torch.rand(num_envs, device=env.device) * 2 - 1) * max_roll_rad
    yaw = torch.zeros(num_envs, device=env.device)  # Keep yaw at 0

    # Convert Euler angles (roll, pitch, yaw) to quaternion
    # Using the standard aerospace sequence (ZYX)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    quat_w = cr * cp * cy + sr * sp * sy
    quat_x = sr * cp * cy - cr * sp * sy
    quat_y = cr * sp * cy + sr * cp * sy
    quat_z = cr * cp * sy - sr * sp * cy

    new_quat = torch.stack([quat_w, quat_x, quat_y, quat_z], dim=1)

    # Normalize quaternion
    new_quat = new_quat / torch.norm(new_quat, dim=1, keepdim=True)

    # Get root position index (freejoint starts at qpos index 0)
    # Freejoint: [x, y, z, qw, qx, qy, qz]
    root_quat_idx = 3  # Quaternion starts at index 3

    # Apply the randomized orientation to selected environments
    env.sim.data.qpos[env_ids, root_quat_idx:root_quat_idx+4] = new_quat


class VelocityCommandCommandOnly(UniformVelocityCommand):
    """Like UniformVelocityCommand but only draws the command arrows (no actual velocity arrows)."""

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        batch = visualizer.env_idx
        if batch >= self.num_envs:
            return

        cmds = self.command.cpu().numpy()
        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        base_quat_w = self.robot.data.root_link_quat_w
        base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()

        base_pos_w = base_pos_ws[batch]
        base_mat_w = base_mat_ws[batch]
        cmd = cmds[batch]

        if np.linalg.norm(base_pos_w) < 1e-6:
            return

        def local_to_world(vec: np.ndarray) -> np.ndarray:
            return base_pos_w + base_mat_w @ vec

        scale = self.cfg.viz.scale * 2.0
        z_offset = self.cfg.viz.z_offset

        # Command linear velocity arrow (blue).
        cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
        cmd_lin_to = local_to_world(
            (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
        )
        visualizer.add_arrow(cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015)
