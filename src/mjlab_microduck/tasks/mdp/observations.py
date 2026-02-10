"""Custom observation functions for microduck."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from mjlab.assets import Entity
from mjlab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def raw_accelerometer(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Raw accelerometer reading (includes gravity + linear acceleration).

    Returns normalized raw accelerometer which mimics what a real IMU measures.
    This is different from pure projected_gravity which only reflects orientation.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get linear acceleration in body frame
    lin_acc_b = asset.data.root_link_lin_acc_b

    # Get projected gravity in body frame
    proj_grav_b = asset.data.projected_gravity_b

    # Raw accelerometer = projected_gravity - linear_acceleration
    # (accelerometer measures specific force)
    raw_accel = proj_grav_b - lin_acc_b

    # Normalize to unit vector
    raw_accel_norm = torch.norm(raw_accel, dim=-1, keepdim=True)
    raw_accel_normalized = torch.where(
        raw_accel_norm > 0.1,
        raw_accel / raw_accel_norm,
        proj_grav_b
    )

    return raw_accel_normalized
