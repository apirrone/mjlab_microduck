"""MDP functions for microduck tasks"""

import torch
from typing import Optional

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.entity import Entity
from mjlab_microduck.reference_motion import ReferenceMotionLoader


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


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
        """Reset phases for specific environments (e.g., after episode termination)"""
        if self.phase is not None and len(env_ids) > 0:
            self.phase[env_ids] = 0.0


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
    Penalize joint torques to encourage energy-efficient motion.

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,) - sum of squared joint torques
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get applied joint torques
    joint_torques = asset.data.applied_torque[:, asset_cfg.joint_ids]

    # Return L2 squared norm
    return torch.sum(torch.square(joint_torques), dim=1)


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
