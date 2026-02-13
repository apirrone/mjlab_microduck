#!/usr/bin/env python3
"""
Validation script to kinematically replay reference motions.

Uses the same viewer infrastructure as `uv run play` but with a custom
"policy" that directly sets robot states from reference motion data.

Usage:
    uv run validate_reference_motion.py --num-envs 10
"""

import argparse

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer
from mjlab_microduck.motion_loader import PolyMotionLoader
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


class MotionReplayPolicy:
    """Policy that kinematically replays reference motions instead of using actions."""

    def __init__(self, env: ManagerBasedRlEnv, motion_file: str):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs

        # Load motion data
        print(f"Loading motion data from {motion_file}...")
        self.motion_loader = PolyMotionLoader(motion_file, device=self.device)

        # Per-environment state
        self.current_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.selected_motions = torch.randint(
            0, self.motion_loader.num_motions, (self.num_envs,), device=self.device
        )

        # Get robot entity
        self.robot = env.scene["robot"]

        print(f"\nSpawning {self.num_envs} robots with random motions:")
        for i in range(self.num_envs):
            motion_idx = self.selected_motions[i].item()
            vel = self.motion_loader.velocity_points[motion_idx]
            print(f"  Robot {i}: motion {motion_idx} (vel: {vel.tolist()})")

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Called each step - we ignore obs and directly set robot state."""
        # Get reference data for all environments (vectorized)
        root_pos = self.motion_loader.get_frame_data(
            self.selected_motions, self.current_frame, "root_pos"
        )
        root_quat_xyzw = self.motion_loader.get_frame_data(
            self.selected_motions, self.current_frame, "root_quat"
        )
        joint_pos = self.motion_loader.get_frame_data(
            self.selected_motions, self.current_frame, "joints_pos"
        )
        joint_vel = self.motion_loader.get_frame_data(
            self.selected_motions, self.current_frame, "joints_vel"
        )
        root_lin_vel = self.motion_loader.get_frame_data(
            self.selected_motions, self.current_frame, "world_linear_vel"
        )
        root_ang_vel = self.motion_loader.get_frame_data(
            self.selected_motions, self.current_frame, "world_angular_vel"
        )

        # Convert quaternion from xyzw to wxyz (MuJoCo format)
        root_quat_wxyz = torch.stack(
            [
                root_quat_xyzw[:, 3],  # w
                root_quat_xyzw[:, 0],  # x
                root_quat_xyzw[:, 1],  # y
                root_quat_xyzw[:, 2],  # z
            ],
            dim=-1,
        )

        # Add environment origin offset
        root_pos_world = root_pos + self.env.scene.env_origins

        # Prepare root state: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        root_state = torch.cat(
            [root_pos_world, root_quat_wxyz, root_lin_vel, root_ang_vel], dim=-1
        )

        # Write directly to simulation (kinematic playback)
        self.robot.write_root_state_to_sim(root_state, env_ids=None)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=None)

        # Advance motion frames
        self.current_frame += 1

        # Wrap around when reaching end
        wrapped = self.current_frame >= self.motion_loader.min_frames
        if torch.any(wrapped):
            self.current_frame[wrapped] = 0

        # Return dummy actions (not used since we're setting state directly)
        return torch.zeros((self.num_envs, self.robot.num_actuators), device=self.device)


def main():
    parser = argparse.ArgumentParser(description="Validate reference motion data")
    parser.add_argument(
        "--num-envs", type=int, default=10, help="Number of robots to spawn"
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default="src/mjlab_microduck/data/reference_motion.pkl",
        help="Path to reference motion pickle file",
    )
    args = parser.parse_args()

    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create environment configuration
    env_cfg = make_microduck_velocity_env_cfg(play=True)
    env_cfg.scene.num_envs = args.num_envs

    # Disable some features that might interfere with kinematic playback
    env_cfg.episode_length_s = 1e9  # Essentially infinite

    print(f"\nCreating environment with {args.num_envs} robots...")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    # Wrap environment (required by viewer)
    env = RslRlVecEnvWrapper(env)

    # Create motion replay "policy"
    policy = MotionReplayPolicy(env.unwrapped, args.motion_file)

    print("\n" + "=" * 60)
    print("Playing back reference motions kinematically...")
    print("Press ESC to exit")
    print("=" * 60 + "\n")

    # Use the native viewer (same as play command)
    viewer = NativeMujocoViewer(env, policy)
    viewer.run()

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
