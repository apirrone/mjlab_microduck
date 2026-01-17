"""Reference motion loader for imitation learning"""

import os
import pickle
import numpy as np
import torch
from typing import Dict, Optional


class ReferenceMotionLoader:
    """Loads and evaluates polynomial-fitted reference motions"""

    def __init__(self, pkl_path: str):
        """
        Args:
            pkl_path: Path to the polynomial coefficients pickle file
        """
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Reference motion file not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)

        # Cache for motion keys sorted by velocity magnitude
        self._motion_keys = list(self.data.keys())
        self._parse_velocities()

    def _parse_velocities(self):
        """Parse dx, dy, dtheta from motion keys"""
        self.velocities = {}
        for key in self._motion_keys:
            parts = key.split("_")
            if len(parts) == 3:
                dx, dy, dtheta = float(parts[0]), float(parts[1]), float(parts[2])
                self.velocities[key] = np.array([dx, dy, dtheta])

    def find_closest_motion(self, commanded_vel: torch.Tensor) -> str:
        """
        Find the reference motion closest to the commanded velocity

        Args:
            commanded_vel: (batch, 3) tensor with [vel_x, vel_y, ang_vel_z]

        Returns:
            Key of the closest reference motion
        """
        # Convert to numpy for comparison (use first env in batch)
        cmd_np = commanded_vel[0].cpu().numpy() if torch.is_tensor(commanded_vel) else commanded_vel[0]

        min_dist = float('inf')
        closest_key = self._motion_keys[0]

        for key, vel in self.velocities.items():
            dist = np.linalg.norm(cmd_np - vel)
            if dist < min_dist:
                min_dist = dist
                closest_key = key

        return closest_key

    def evaluate_motion(self, motion_key: str, phase: torch.Tensor, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Evaluate polynomial at given phase(s) in the gait cycle

        Args:
            motion_key: Key identifying the reference motion
            phase: (batch,) tensor with phase values in [0, 1]
            device: Device to place tensors on

        Returns:
            Dictionary with:
                - joints_pos: (batch, 14) joint positions
                - joints_vel: (batch, 14) joint velocities
                - foot_contacts: (batch, 2) foot contact states
                - base_linear_vel: (batch, 3) base linear velocity
                - base_angular_vel: (batch, 3) base angular velocity
        """
        if motion_key not in self.data:
            raise ValueError(f"Motion key {motion_key} not found in reference data")

        motion_data = self.data[motion_key]
        coeffs = motion_data["coefficients"]

        # Ensure phase is in [0, 1]
        phase = torch.clamp(phase, 0.0, 1.0)

        # Evaluate all polynomials
        num_dims = len(coeffs)
        batch_size = phase.shape[0]

        results = torch.zeros((batch_size, num_dims), device=device)

        for dim in range(num_dims):
            poly_coeffs = torch.tensor(coeffs[f"dim_{dim}"], dtype=torch.float32, device=device)
            # Evaluate polynomial: sum(coeff[i] * phase^i)
            # Coefficients are stored from low to high degree [c0, c1, c2, ..., c_n]
            # Use Horner's method for numerical stability
            result = torch.zeros_like(phase)
            for i in range(len(poly_coeffs) - 1, -1, -1):
                result = result * phase + poly_coeffs[i]
            results[:, dim] = result

        # Split results into components
        # Order: joints_pos (14), joints_vel (14), foot_contacts (2), base_linear_vel (3), base_angular_vel (3)
        idx = 0
        joints_pos = results[:, idx:idx+14]
        idx += 14
        joints_vel = results[:, idx:idx+14]
        idx += 14
        foot_contacts = results[:, idx:idx+2]
        idx += 2
        base_linear_vel = results[:, idx:idx+3]
        idx += 3
        base_angular_vel = results[:, idx:idx+3]

        return {
            "joints_pos": joints_pos,
            "joints_vel": joints_vel,
            "foot_contacts": foot_contacts,
            "base_linear_vel": base_linear_vel,
            "base_angular_vel": base_angular_vel,
        }

    def get_period(self, motion_key: str) -> float:
        """Get the period of a reference motion in seconds"""
        return self.data[motion_key]["period"]
