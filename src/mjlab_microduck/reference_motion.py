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

    def find_closest_motion(self, commanded_vel: torch.Tensor) -> torch.Tensor:
        """
        Find the reference motion closest to the commanded velocity for each environment

        Args:
            commanded_vel: (batch, 3) tensor with [vel_x, vel_y, ang_vel_z]

        Returns:
            Integer tensor (batch,) with motion indices for each environment
        """
        # Convert commanded velocities to numpy
        cmd_np = commanded_vel.cpu().numpy() if torch.is_tensor(commanded_vel) else commanded_vel

        batch_size = cmd_np.shape[0]
        motion_indices = np.zeros(batch_size, dtype=np.int32)

        # Pre-compute velocity array for all motions
        motion_keys_list = list(self.velocities.keys())
        vel_array = np.array([self.velocities[key] for key in motion_keys_list])  # (num_motions, 3)

        # For each environment, find closest motion
        for i in range(batch_size):
            # Compute distance to all motions
            dists = np.linalg.norm(vel_array - cmd_np[i], axis=1)
            motion_indices[i] = np.argmin(dists)

        return torch.from_numpy(motion_indices).to(commanded_vel.device)

    def get_motion_key(self, motion_idx: int) -> str:
        """Get motion key from integer index"""
        motion_keys_list = list(self.velocities.keys())
        return motion_keys_list[motion_idx]

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

    def evaluate_motion_batch(self, motion_indices: torch.Tensor, phases: torch.Tensor, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Evaluate multiple different motions at given phases (one motion per environment)

        Args:
            motion_indices: (batch,) tensor with integer motion indices
            phases: (batch,) tensor with phase values in [0, 1]
            device: Device to place tensors on

        Returns:
            Dictionary with batched results for each environment
        """
        batch_size = motion_indices.shape[0]
        motion_keys_list = list(self.velocities.keys())

        # Pre-allocate output tensors
        joints_pos_batch = torch.zeros((batch_size, 14), device=device)
        joints_vel_batch = torch.zeros((batch_size, 14), device=device)
        foot_contacts_batch = torch.zeros((batch_size, 2), device=device)
        base_linear_vel_batch = torch.zeros((batch_size, 3), device=device)
        base_angular_vel_batch = torch.zeros((batch_size, 3), device=device)

        # Group environments by motion to reduce redundant evaluations
        unique_motions = torch.unique(motion_indices)

        for motion_idx in unique_motions:
            mask = motion_indices == motion_idx
            env_indices = torch.where(mask)[0]

            motion_key = motion_keys_list[motion_idx.item()]
            phases_for_motion = phases[env_indices]

            # Evaluate this motion for all environments using it
            result = self.evaluate_motion(motion_key, phases_for_motion, device=device)

            # Scatter results back to the correct environment indices
            joints_pos_batch[env_indices] = result["joints_pos"]
            joints_vel_batch[env_indices] = result["joints_vel"]
            foot_contacts_batch[env_indices] = result["foot_contacts"]
            base_linear_vel_batch[env_indices] = result["base_linear_vel"]
            base_angular_vel_batch[env_indices] = result["base_angular_vel"]

        return {
            "joints_pos": joints_pos_batch,
            "joints_vel": joints_vel_batch,
            "foot_contacts": foot_contacts_batch,
            "base_linear_vel": base_linear_vel_batch,
            "base_angular_vel": base_angular_vel_batch,
        }

    def get_period_batch(self, motion_indices: torch.Tensor) -> torch.Tensor:
        """
        Get periods for a batch of motion indices

        Args:
            motion_indices: (batch,) tensor with integer motion indices

        Returns:
            Periods tensor (batch,) in seconds
        """
        motion_keys_list = list(self.velocities.keys())
        periods = torch.zeros(motion_indices.shape[0], dtype=torch.float32, device=motion_indices.device)

        for i, idx in enumerate(motion_indices):
            motion_key = motion_keys_list[idx.item()]
            periods[i] = self.data[motion_key]["period"]

        return periods
