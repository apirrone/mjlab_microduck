"""Head position command for microduck."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class UniformHeadCommand(CommandTerm):
    """Command term for sampling random head positions.

    Samples 4 random head joint positions:
    - neck_pitch
    - head_pitch
    - head_yaw
    - head_roll
    """

    cfg: UniformHeadCommandCfg

    def __init__(self, cfg: UniformHeadCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.asset_name]

        # Command buffer: [num_envs, 4] for the 4 head joints
        self.head_command = torch.zeros(self.num_envs, 4, device=self.device)

        # Find indices for the head joints
        self.head_joint_names = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
        self.head_joint_indices = []
        for joint_name in self.head_joint_names:
            idx = self.robot.joint_names.index(joint_name)
            self.head_joint_indices.append(idx)
        self.head_joint_indices = torch.tensor(
            self.head_joint_indices, dtype=torch.long, device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        """Returns head command tensor [num_envs, 4]."""
        return self.head_command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Sample new random head positions for given environments."""
        r = torch.empty(len(env_ids), device=self.device)

        # Sample each head joint independently
        self.head_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.neck_pitch)
        self.head_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.head_pitch)
        self.head_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.head_yaw)
        self.head_command[env_ids, 3] = r.uniform_(*self.cfg.ranges.head_roll)

    def _update_command(self) -> None:
        """No dynamic updates needed for head commands."""
        pass

    def _update_metrics(self) -> None:
        """Track head position tracking error."""
        # Get current head joint positions
        current_head_pos = self.robot.data.joint_pos[:, self.head_joint_indices]

        # Compute L2 error
        head_error = torch.norm(self.head_command - current_head_pos, dim=-1)

        if "error_head_pos" not in self.metrics:
            self.metrics["error_head_pos"] = torch.zeros(
                self.num_envs, device=self.device
            )

        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_head_pos"] += head_error / max_command_step


@dataclass(kw_only=True)
class UniformHeadCommandCfg(CommandTermCfg):
    """Configuration for uniform head position command."""

    asset_name: str = "robot"
    class_type: type[CommandTerm] = UniformHeadCommand

    @dataclass
    class Ranges:
        """Ranges for each head joint command (in radians)."""
        neck_pitch: tuple[float, float] = (-0.5, 0.5)
        head_pitch: tuple[float, float] = (-0.5, 0.5)
        head_yaw: tuple[float, float] = (-0.5, 0.5)
        head_roll: tuple[float, float] = (-0.5, 0.5)

    ranges: Ranges
