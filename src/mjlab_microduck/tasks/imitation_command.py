"""
Imitation command manager for velocity-indexed motion tracking.

Adapted from beyondmimic implementation to work with mjlab_microduck.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import sample_uniform
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab_microduck.motion_loader import PolyMotionLoader

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


class ImitationCommand(CommandTerm):
    """Imitation command that selects reference motion based on velocity commands.

    This command maintains velocity targets (dx, dy, dtheta) for each environment
    and looks up the corresponding reference motion from a velocity-indexed library.
    """

    cfg: ImitationCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: ImitationCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]

        # Load velocity-indexed motion library
        self.motion = PolyMotionLoader(self.cfg.motion_file, device=self.device)

        # Per-environment state
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Velocity commands per environment (dx, dy, dtheta)
        self.vel_cmd_x = torch.zeros(self.num_envs, device=self.device)
        self.vel_cmd_y = torch.zeros(self.num_envs, device=self.device)
        self.vel_cmd_yaw = torch.zeros(self.num_envs, device=self.device)

        # Current motion index per environment (cached for efficiency)
        self._motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._steps_in_period = torch.full(
            (self.num_envs,), self.motion.min_frames, dtype=torch.long, device=self.device
        )

        # Adaptive sampling: track failure counts per motion (velocity command)
        self.motion_failed_count = torch.zeros(
            self.motion.num_motions, dtype=torch.float, device=self.device
        )
        self._current_motion_failed = torch.zeros(
            self.motion.num_motions, dtype=torch.float, device=self.device
        )

        # Metrics
        self.metrics["vel_cmd_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["vel_cmd_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["vel_cmd_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_motion"] = torch.zeros(self.num_envs, device=self.device)

        # Ghost visualization
        self._ghost_model: mujoco.MjModel | None = None
        self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

    @property
    def command(self) -> torch.Tensor:
        """Return velocity command and phase encoding as the command tensor.

        The phase is encoded as sinusoidal features:
        - sin(2πφ), cos(2πφ): fundamental gait cycle
        - sin(4πφ), cos(4πφ): second harmonic (head-bob at 2× freq)
        """
        phase = self.phase
        two_pi_phase = 2 * torch.pi * phase
        four_pi_phase = 4 * torch.pi * phase
        return torch.stack(
            [
                self.vel_cmd_x,
                self.vel_cmd_y,
                self.vel_cmd_yaw,
                torch.sin(two_pi_phase),
                torch.cos(two_pi_phase),
                torch.sin(four_pi_phase),
                torch.cos(four_pi_phase),
            ],
            dim=-1,
        )

    @property
    def phase(self) -> torch.Tensor:
        """Current gait phase [0, 1) for each environment."""
        frame_idx = self._get_frame_idx()
        return self.motion.phase_array[self._motion_idx, frame_idx]

    def _get_frame_idx(self) -> torch.Tensor:
        """Get current frame index within full motion length."""
        return self.time_steps % self.motion.min_frames

    # ---- Reference motion frame data ----

    @property
    def root_pos(self) -> torch.Tensor:
        """Reference root position in world frame (num_envs, 3)."""
        pos = self.motion.get_frame_data(self._motion_idx, self._get_frame_idx(), "root_pos")
        # Add environment origins to transform to world frame
        pos = pos + self._env.scene.env_origins
        return pos

    @property
    def root_quat(self) -> torch.Tensor:
        """Reference root quaternion [w, x, y, z] (num_envs, 4)."""
        # Frame stores xyzw, convert to wxyz for mjlab
        quat_xyzw = self.motion.get_frame_data(
            self._motion_idx, self._get_frame_idx(), "root_quat"
        )
        return quat_xyzw[:, [3, 0, 1, 2]]

    @property
    def joint_pos(self) -> torch.Tensor:
        """Reference joint positions (num_envs, num_joints)."""
        return self.motion.get_frame_data(self._motion_idx, self._get_frame_idx(), "joints_pos")

    @property
    def joint_vel(self) -> torch.Tensor:
        """Reference joint velocities (num_envs, num_joints)."""
        return self.motion.get_frame_data(self._motion_idx, self._get_frame_idx(), "joints_vel")

    @property
    def world_linear_vel(self) -> torch.Tensor:
        """Reference world linear velocity (num_envs, 3)."""
        return self.motion.get_frame_data(self._motion_idx, self._get_frame_idx(), "world_linear_vel")

    @property
    def world_angular_vel(self) -> torch.Tensor:
        """Reference world angular velocity (num_envs, 3)."""
        return self.motion.get_frame_data(self._motion_idx, self._get_frame_idx(), "world_angular_vel")

    @property
    def foot_contacts(self) -> torch.Tensor:
        """Reference foot contacts [left, right] (num_envs, 2)."""
        return self.motion.get_frame_data(self._motion_idx, self._get_frame_idx(), "foot_contacts")

    # ---- Robot state ----

    @property
    def robot_root_pos(self) -> torch.Tensor:
        """Robot root position (num_envs, 3)."""
        return self.robot.data.root_link_pos_w

    @property
    def robot_root_quat(self) -> torch.Tensor:
        """Robot root quaternion [w, x, y, z] (num_envs, 4)."""
        return self.robot.data.root_link_quat_w

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        """Robot joint positions (num_envs, num_joints)."""
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        """Robot joint velocities (num_envs, num_joints)."""
        return self.robot.data.joint_vel

    @property
    def robot_world_linear_vel(self) -> torch.Tensor:
        """Robot world-frame linear velocity (num_envs, 3)."""
        return self.robot.data.root_link_lin_vel_w

    @property
    def robot_world_angular_vel(self) -> torch.Tensor:
        """Robot world-frame angular velocity (num_envs, 3)."""
        return self.robot.data.root_link_ang_vel_w

    def _update_motion_selection(self):
        """Update motion index based on current velocity commands."""
        self._motion_idx = self.motion.get_nearest_motion_idx(
            self.vel_cmd_x, self.vel_cmd_y, self.vel_cmd_yaw
        )
        self._steps_in_period = self.motion.get_motion_steps_in_period(self._motion_idx)

    def _update_metrics(self):
        self.metrics["vel_cmd_x"] = self.vel_cmd_x
        self.metrics["vel_cmd_y"] = self.vel_cmd_y
        self.metrics["vel_cmd_yaw"] = self.vel_cmd_yaw
        self.metrics["error_joint_pos"] = torch.norm(
            self.joint_pos - self.robot_joint_pos, dim=-1
        )
        self.metrics["error_joint_vel"] = torch.norm(
            self.joint_vel - self.robot_joint_vel, dim=-1
        )

    def _adaptive_sampling(self, env_ids: torch.Tensor):
        """Sample motions with probability proportional to failure rate.

        Motions that cause more terminations are sampled more frequently,
        enabling curriculum-like learning on harder velocity commands.
        """
        # Track failures from terminated episodes
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            fail_motions = self._motion_idx[env_ids][episode_failed]
            self._current_motion_failed[:] = torch.bincount(
                fail_motions, minlength=self.motion.num_motions
            ).float()

        # Compute sampling probabilities with uniform baseline
        sampling_probabilities = (
            self.motion_failed_count + self.cfg.adaptive_uniform_ratio / self.motion.num_motions
        )
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        # Sample motion indices according to failure-weighted distribution
        sampled_motion_idx = torch.multinomial(
            sampling_probabilities, len(env_ids), replacement=True
        )
        self._motion_idx[env_ids] = sampled_motion_idx

        # Set velocity commands from sampled motions
        self.vel_cmd_x[env_ids] = self.motion.velocity_points[sampled_motion_idx, 0]
        self.vel_cmd_y[env_ids] = self.motion.velocity_points[sampled_motion_idx, 1]
        self.vel_cmd_yaw[env_ids] = self.motion.velocity_points[sampled_motion_idx, 2]

        # Update sampling metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.motion.num_motions)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_motion"][:] = imax.float() / self.motion.num_motions

    def _uniform_sampling(self, env_ids: torch.Tensor):
        """Sample velocity commands uniformly from configured ranges."""
        vel_x_range = self.cfg.velocity_cmd_range.get("x", (0.0, 0.0))
        vel_y_range = self.cfg.velocity_cmd_range.get("y", (0.0, 0.0))
        vel_yaw_range = self.cfg.velocity_cmd_range.get("yaw", (0.0, 0.0))

        self.vel_cmd_x[env_ids] = sample_uniform(
            vel_x_range[0], vel_x_range[1], (len(env_ids),), device=self.device
        )
        self.vel_cmd_y[env_ids] = sample_uniform(
            vel_y_range[0], vel_y_range[1], (len(env_ids),), device=self.device
        )
        self.vel_cmd_yaw[env_ids] = sample_uniform(
            vel_yaw_range[0], vel_yaw_range[1], (len(env_ids),), device=self.device
        )

        # Update motion selection to match new velocity commands
        self._update_motion_selection()

        # Uniform entropy metrics
        self.metrics["sampling_entropy"][:] = 1.0
        self.metrics["sampling_top1_prob"][:] = 1.0 / self.motion.num_motions
        self.metrics["sampling_top1_motion"][:] = 0.5

    def _resample_command(self, env_ids: torch.Tensor):
        """Resample velocity commands and phase for specified environments."""
        # Sample velocity commands based on sampling mode
        if self.cfg.sampling_mode == "adaptive":
            self._adaptive_sampling(env_ids)
        else:
            # start or uniform mode: sample velocity uniformly from ranges
            self._uniform_sampling(env_ids)

        # Sample phase/time based on sampling mode
        if self.cfg.sampling_mode == "start":
            self.time_steps[env_ids] = 0
        else:  # uniform or adaptive: sample phase uniformly
            # Sample uniformly within [0, steps_in_period) for each env
            max_steps = self._steps_in_period[env_ids].float()
            self.time_steps[env_ids] = (
                sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device) * max_steps
            ).long()

        # Write robot state from reference motion
        root_pos = self.root_pos[env_ids]  # Already in world frame
        root_quat = self.root_quat[env_ids]
        root_lin_vel = self.world_linear_vel[env_ids]
        root_ang_vel = self.world_angular_vel[env_ids]

        root_state = torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        joint_pos = self.joint_pos[env_ids]
        joint_vel = self.joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _update_command(self):
        """Update command each step - advance time and handle period wrapping."""
        self.time_steps += 1

        # Resample when reference motion ends
        env_ids = torch.where(self.time_steps >= self.motion.min_frames)[0]
        if env_ids.numel() > 0:
            self._resample_command(env_ids)

        # Update motion selection in case velocities changed externally
        self._update_motion_selection()

        # Update adaptive sampling failure counts (exponential moving average)
        if self.cfg.sampling_mode == "adaptive":
            self.motion_failed_count = (
                self.cfg.adaptive_alpha * self._current_motion_failed
                + (1 - self.cfg.adaptive_alpha) * self.motion_failed_count
            )
            self._current_motion_failed.zero_()

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Draw ghost robot based on reference motion."""
        # Check if method exists (different visualizer implementations may vary)
        if not hasattr(visualizer, 'get_env_indices'):
            # Fallback: visualize all environments
            env_indices = list(range(self.num_envs))
        else:
            env_indices = visualizer.get_env_indices(self.num_envs)
            if not env_indices:
                return

        if self.cfg.viz.mode == "ghost":
            if self._ghost_model is None:
                self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
                self._ghost_model.geom_rgba[:] = self._ghost_color

            entity: Entity = self._env.scene[self.cfg.entity_name]
            indexing = entity.indexing
            joint_q_adr = indexing.joint_q_adr.cpu().numpy()
            free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()

            for batch in env_indices:
                qpos = np.zeros(self._env.sim.mj_model.nq)

                # Use reference motion root pose (already in world frame)
                ref_pos = self.root_pos[batch].cpu().numpy()
                qpos[free_joint_q_adr[0:3]] = ref_pos

                # root_quat is already in wxyz format for MuJoCo
                qpos[free_joint_q_adr[3:7]] = self.root_quat[batch].cpu().numpy()

                # Use reference joint positions
                qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

                visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")


@dataclass(kw_only=True)
class ImitationCommandCfg(CommandTermCfg):
    """Configuration for imitation velocity command.

    The imitation command samples velocity targets (dx, dy, dtheta) and looks up
    the corresponding reference motion from a velocity-indexed motion library.
    """

    motion_file: str
    """Path to pickle file containing velocity-indexed motion data."""

    entity_name: str
    """Name of the robot entity in the scene."""

    velocity_cmd_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    """Velocity command ranges for x, y, and yaw."""

    sampling_mode: Literal["start", "uniform", "adaptive"] = "uniform"
    """Phase sampling mode:
    - 'start': phase begins at 0, velocity sampled uniformly from ranges
    - 'uniform': phase sampled uniformly in [0, 1), velocity sampled uniformly from ranges
    - 'adaptive': phase sampled uniformly, motion (velocity) sampled proportional to failure rate
    """

    adaptive_uniform_ratio: float = 0.1
    """Baseline uniform probability added to failure-weighted sampling (prevents collapse)."""

    adaptive_alpha: float = 0.001
    """Exponential moving average decay for failure counts."""

    @dataclass
    class VizCfg:
        mode: Literal["ghost", "frames"] = "ghost"
        ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

    viz: VizCfg = field(default_factory=VizCfg)
    """Visualization configuration for ghost rendering."""

    class_type: type[CommandTerm] = ImitationCommand

    def __post_init__(self):
        """Set default velocity ranges if not provided."""
        if not self.velocity_cmd_range:
            self.velocity_cmd_range = {
                "x": (-0.1, 0.15),
                "y": (-0.15, 0.15),
                "yaw": (-1.0, 1.0),
            }
