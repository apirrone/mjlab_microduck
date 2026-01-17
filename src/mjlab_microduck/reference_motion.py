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
        self._precompute_gpu_data()

    def _parse_velocities(self):
        """Parse dx, dy, dtheta from motion keys"""
        self.velocities = {}
        for key in self._motion_keys:
            parts = key.split("_")
            if len(parts) == 3:
                dx, dy, dtheta = float(parts[0]), float(parts[1]), float(parts[2])
                self.velocities[key] = np.array([dx, dy, dtheta])

    def _precompute_gpu_data(self):
        """Precompute data structures for fast GPU operations"""
        # Velocity array for all motions: (num_motions, 3)
        motion_keys_list = list(self.velocities.keys())
        vel_array = np.array([self.velocities[key] for key in motion_keys_list])
        self._vel_array_np = vel_array  # Keep numpy version for CPU fallback
        self._vel_tensor = None  # Will be created on device when needed

        # Period array for all motions: (num_motions,)
        period_array = np.array([self.data[key]["period"] for key in motion_keys_list], dtype=np.float32)
        self._period_array_np = period_array
        self._period_tensor = None  # Will be created on device when needed

    def find_closest_motion(self, commanded_vel: torch.Tensor) -> torch.Tensor:
        """
        Find the reference motion closest to the commanded velocity for each environment
        Fully vectorized GPU implementation for performance.

        Args:
            commanded_vel: (batch, 3) tensor with [vel_x, vel_y, ang_vel_z]

        Returns:
            Integer tensor (batch,) with motion indices for each environment
        """
        device = commanded_vel.device

        # Lazy initialization of GPU tensors
        if self._vel_tensor is None or self._vel_tensor.device != device:
            self._vel_tensor = torch.from_numpy(self._vel_array_np).float().to(device)

        # Vectorized distance computation on GPU
        # commanded_vel: (batch, 3)
        # vel_tensor: (num_motions, 3)
        # Compute pairwise distances: (batch, num_motions)
        # Expand dimensions for broadcasting: (batch, 1, 3) - (1, num_motions, 3)
        diff = commanded_vel.unsqueeze(1) - self._vel_tensor.unsqueeze(0)  # (batch, num_motions, 3)
        distances = torch.linalg.norm(diff, dim=2)  # (batch, num_motions)

        # Find closest motion for each environment
        motion_indices = torch.argmin(distances, dim=1)  # (batch,)

        return motion_indices

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
        Vectorized GPU implementation for performance.

        Args:
            motion_indices: (batch,) tensor with integer motion indices

        Returns:
            Periods tensor (batch,) in seconds
        """
        device = motion_indices.device

        # Lazy initialization of GPU tensors
        if self._period_tensor is None or self._period_tensor.device != device:
            self._period_tensor = torch.from_numpy(self._period_array_np).to(device)

        # Simple indexing operation on GPU - very fast
        periods = self._period_tensor[motion_indices]

        return periods
