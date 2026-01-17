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
        self.current_motion_key = None

    def initialize(self, num_envs: int, device: str):
        """Initialize phase tracking for each environment"""
        self.phase = torch.zeros(num_envs, device=device)


def imitation_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    imitation_state: Optional[ImitationRewardState] = None,
    command_threshold: float = 0.01,
    weight_joint_pos: float = 15.0,
    weight_joint_vel: float = 1e-3,
    weight_lin_vel_xy: float = 1.0,
    weight_lin_vel_z: float = 1.0,
    weight_ang_vel_xy: float = 0.5,
    weight_ang_vel_z: float = 0.5,
    weight_contact: float = 1.0,
) -> torch.Tensor:
    """
    Imitation reward based on reference motion tracking

    Args:
        env: The environment
        asset_cfg: Asset configuration
        imitation_state: State object holding reference motion loader and phase tracking
        command_threshold: Minimum command magnitude to apply reward
        weight_*: Weights for different reward components

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

    if not active_mask.any():
        return torch.zeros(env.num_envs, device=env.device)

    # Find closest reference motion (use same motion for all envs for now)
    motion_key = imitation_state.ref_motion_loader.find_closest_motion(cmd_vel)

    # Get period and update phase
    period = imitation_state.ref_motion_loader.get_period(motion_key)
    dt = env.step_dt
    imitation_state.phase += dt / period
    imitation_state.phase = torch.fmod(imitation_state.phase, 1.0)  # Keep in [0, 1]

    # Evaluate reference motion at current phase
    ref_data = imitation_state.ref_motion_loader.evaluate_motion(
        motion_key, imitation_state.phase, device=env.device
    )

    # Get current state
    # Joint positions and velocities (exclude neck/head joints)
    # Joint order (14 total): left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
    #                         neck_pitch, head_pitch, head_yaw, head_roll,
    #                         right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
    joints_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joints_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # Remove neck/head joints (indices 5-8) from both robot and reference
    # Result: 10 joints (left leg 5 + right leg 5)
    joints_pos_filtered = torch.cat([joints_pos[:, :5], joints_pos[:, 9:]], dim=1)
    joints_vel_filtered = torch.cat([joints_vel[:, :5], joints_vel[:, 9:]], dim=1)

    # Filter reference motion the same way
    ref_joints_pos_filtered = torch.cat([ref_data["joints_pos"][:, :5], ref_data["joints_pos"][:, 9:]], dim=1)
    ref_joints_vel_filtered = torch.cat([ref_data["joints_vel"][:, :5], ref_data["joints_vel"][:, 9:]], dim=1)

    # Base velocities (world frame)
    base_lin_vel = asset.data.root_link_vel_w[:, :3]  # Linear velocity (first 3 components)
    base_ang_vel = asset.data.root_link_vel_w[:, 3:]  # Angular velocity (last 3 components)

    # Foot contacts
    if "feet_ground_contact" in env.scene.sensors:
        contacts = env.scene.sensors["feet_ground_contact"].data.found[:, :2]  # (num_envs, 2)
    else:
        contacts = torch.zeros((env.num_envs, 2), device=env.device)

    # Compute reward components
    # Joint position: negative squared error
    joint_pos_rew = -torch.sum(torch.square(joints_pos_filtered - ref_joints_pos_filtered), dim=1) * weight_joint_pos

    # Joint velocity: negative squared error
    joint_vel_rew = -torch.sum(torch.square(joints_vel_filtered - ref_joints_vel_filtered), dim=1) * weight_joint_vel

    # Linear velocity XY: exponential reward
    lin_vel_xy_rew = torch.exp(-8.0 * torch.sum(torch.square(base_lin_vel[:, :2] - ref_data["base_linear_vel"][:, :2]), dim=1)) * weight_lin_vel_xy

    # Linear velocity Z: exponential reward
    lin_vel_z_rew = torch.exp(-8.0 * torch.square(base_lin_vel[:, 2] - ref_data["base_linear_vel"][:, 2])) * weight_lin_vel_z

    # Angular velocity XY: exponential reward
    ang_vel_xy_rew = torch.exp(-2.0 * torch.sum(torch.square(base_ang_vel[:, :2] - ref_data["base_angular_vel"][:, :2]), dim=1)) * weight_ang_vel_xy

    # Angular velocity Z: exponential reward
    ang_vel_z_rew = torch.exp(-2.0 * torch.square(base_ang_vel[:, 2] - ref_data["base_angular_vel"][:, 2])) * weight_ang_vel_z

    # Contact reward: binary match
    ref_contacts = (ref_data["foot_contacts"] > 0.5).float()
    contacts_float = contacts.float()
    contact_rew = torch.sum(contacts_float == ref_contacts, dim=1) * weight_contact

    # Total reward
    reward = (
        joint_pos_rew
        + joint_vel_rew
        + lin_vel_xy_rew
        + lin_vel_z_rew
        + ang_vel_xy_rew
        + ang_vel_z_rew
        + contact_rew
    )

    # Apply mask for active commands only
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


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    """
    Reward for staying alive (not terminated)

    Args:
        env: The environment

    Returns:
        Reward tensor of shape (num_envs,) - ones for all envs
    """
    return torch.ones(env.num_envs, device=env.device)
