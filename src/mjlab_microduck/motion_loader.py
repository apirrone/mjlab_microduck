"""
Frame-based motion loader for velocity-indexed reference motions.

Adapted from beyondmimic implementation to work with frame-by-frame
reference motion data stored in pickle files.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class PolyMotionLoader:
    """Loads velocity-indexed motion data from a pickle file.

    The pickle file contains a dictionary mapping velocity keys (dx_dy_dtheta)
    to motion data with frame-based structure.

    Example structure:
        {
            "0.1_0.0_0.5": {
                "motion_data": ndarray (n_frames, frame_size),
                "phase": ndarray (n_frames,),
                "period": float,
                "fps": float,
                "meta": {
                    "Frame_offset": [{"root_pos": 0, "root_quat": 3, ...}],
                    "Frame_size": [{"root_pos": 3, "root_quat": 4, ...}]
                }
            },
            ...
        }
    """

    def __init__(self, motion_file: str, device: str = "cpu") -> None:
        """
        Args:
            motion_file: Path to the reference motion pickle file
            device: PyTorch device to place tensors on
        """
        motion_path = Path(motion_file)
        if not motion_path.exists():
            raise FileNotFoundError(f"Motion file not found: {motion_file}")

        self.device = torch.device(device)

        # Load pickle data
        with open(motion_file, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict) or len(data) == 0:
            raise ValueError(f"Invalid motion data format in {motion_file}")

        # Extract velocity points and motion data
        velocity_points = []
        motion_data_list = []
        motion_periods_list = []
        motion_fps_list = []
        phase_data_list = []

        # Get frame slices from first entry
        first_key = list(data.keys())[0]
        frame_offsets = data[first_key]["meta"]["Frame_offset"][0]
        frame_sizes = data[first_key]["meta"]["Frame_size"][0]

        self.slices: dict[str, slice] = {}
        for key, offset in frame_offsets.items():
            self.slices[key] = slice(offset, offset + frame_sizes[key])

        # Find minimum frame length across all motions
        min_frames = float("inf")
        valid_entries = []

        for name, entry in data.items():
            # Parse velocity from key
            dx, dy, dtheta = map(float, name.split("_"))
            motion_data = np.array(entry["motion_data"])
            phase_data = np.array(entry["phase"])
            period = entry["period"]
            fps = entry["fps"]

            min_frames = min(min_frames, motion_data.shape[0])
            valid_entries.append((dx, dy, dtheta, motion_data, phase_data, period, fps))

        self.min_frames = int(min_frames)

        # Build arrays, cutting each motion to min_frames for uniform shape
        for dx, dy, dtheta, motion_data, phase_data, period, fps in valid_entries:
            cut_data = motion_data[: self.min_frames]
            cut_phase = phase_data[: self.min_frames]
            velocity_points.append([dx, dy, dtheta])
            motion_data_list.append(cut_data)
            phase_data_list.append(cut_phase)
            motion_periods_list.append(period)
            motion_fps_list.append(fps)

        # Store as tensors
        self.velocity_points = torch.tensor(
            velocity_points, dtype=torch.float32, device=self.device
        )
        self.motion_data_array = torch.tensor(
            np.array(motion_data_list), dtype=torch.float32, device=self.device
        )
        self.phase_array = torch.tensor(
            np.array(phase_data_list), dtype=torch.float32, device=self.device
        )
        self.motion_periods = torch.tensor(
            motion_periods_list, dtype=torch.float32, device=self.device
        )
        self.motion_fps = torch.tensor(
            motion_fps_list, dtype=torch.float32, device=self.device
        )

        # Calculate scale factors for normalized distance computation
        dx_vals = self.velocity_points[:, 0]
        dy_vals = self.velocity_points[:, 1]
        dtheta_vals = self.velocity_points[:, 2]
        self.dx_scale = max(1.0, (dx_vals.max() - dx_vals.min()).item())
        self.dy_scale = max(1.0, (dy_vals.max() - dy_vals.min()).item())
        self.dtheta_scale = max(1.0, (dtheta_vals.max() - dtheta_vals.min()).item())

        self.num_motions = len(velocity_points)

        print(f"Loaded {self.num_motions} motions with {self.min_frames} frames each")

    def get_nearest_motion_idx(
        self, dx: torch.Tensor, dy: torch.Tensor, dtheta: torch.Tensor
    ) -> torch.Tensor:
        """Find nearest motion index for each velocity command.

        Uses normalized Euclidean distance in velocity space.

        Args:
            dx: Linear x velocity commands (num_envs,)
            dy: Linear y velocity commands (num_envs,)
            dtheta: Angular velocity commands (num_envs,)

        Returns:
            Indices into motion_data_array (num_envs,)
        """
        # Shape: (num_envs, 3)
        query = torch.stack([dx, dy, dtheta], dim=-1)

        # Normalized distances: (num_envs, num_motions)
        dx_diff = (self.velocity_points[:, 0] - query[:, 0:1]) / self.dx_scale
        dy_diff = (self.velocity_points[:, 1] - query[:, 1:2]) / self.dy_scale
        dtheta_diff = (self.velocity_points[:, 2] - query[:, 2:3]) / self.dtheta_scale

        distances = dx_diff**2 + dy_diff**2 + dtheta_diff**2
        return torch.argmin(distances, dim=-1)

    def get_motion_steps_in_period(self, motion_idx: torch.Tensor) -> torch.Tensor:
        """Get number of steps in one period for given motion indices.

        Args:
            motion_idx: Motion indices (num_envs,)

        Returns:
            Number of frames per period (num_envs,)
        """
        periods = self.motion_periods[motion_idx]
        fps = self.motion_fps[motion_idx]
        return torch.clamp((periods * fps).long(), max=self.min_frames)

    def get_frame_data(
        self, motion_idx: torch.Tensor, frame_idx: torch.Tensor, field: str
    ) -> torch.Tensor:
        """Get field data for given motion and frame indices.

        Args:
            motion_idx: Motion indices (num_envs,)
            frame_idx: Frame indices (num_envs,)
            field: Field name (e.g., "root_pos", "joints_pos", "foot_contacts")

        Returns:
            Field data (num_envs, field_size)
        """
        if field not in self.slices:
            raise ValueError(
                f"Unknown field '{field}'. Available: {list(self.slices.keys())}"
            )

        s = self.slices[field]
        # motion_data_array: (num_motions, num_frames, frame_size)
        return self.motion_data_array[motion_idx, frame_idx, s]

    def get_field_slice(self, field: str) -> slice:
        """Get the slice object for a specific field.

        Args:
            field: Field name

        Returns:
            Slice object for indexing into motion frames
        """
        if field not in self.slices:
            raise ValueError(
                f"Unknown field '{field}'. Available: {list(self.slices.keys())}"
            )
        return self.slices[field]

    @property
    def available_fields(self) -> list[str]:
        """Get list of available field names."""
        return list(self.slices.keys())

    def __repr__(self) -> str:
        return (
            f"PolyMotionLoader(num_motions={self.num_motions}, "
            f"min_frames={self.min_frames}, device={self.device})"
        )
