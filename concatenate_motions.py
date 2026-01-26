#!/usr/bin/env python3
"""Concatenate multiple BeyondMimic motion files into one for holonomous training.

This allows training on multiple reference motions simultaneously, covering
different velocity commands (forward, backward, left, right, rotation).
"""

import argparse
import numpy as np
from pathlib import Path


def load_motion(npz_path: str) -> dict:
    """Load a single motion .npz file."""
    data = np.load(npz_path)
    return {
        'joint_pos': data['joint_pos'],
        'joint_vel': data['joint_vel'],
        'body_pos_w': data['body_pos_w'],
        'body_quat_w': data['body_quat_w'],
        'body_lin_vel_w': data['body_lin_vel_w'],
        'body_ang_vel_w': data['body_ang_vel_w'],
    }


def concatenate_motions(motion_files: list[str]) -> dict:
    """Concatenate multiple motion files into one.

    Args:
        motion_files: List of paths to .npz motion files

    Returns:
        Dictionary with concatenated arrays
    """
    motions = []
    total_steps = 0

    print(f"Loading {len(motion_files)} motion files...")
    for i, motion_file in enumerate(motion_files):
        motion = load_motion(motion_file)
        motions.append(motion)
        steps = motion['joint_pos'].shape[0]
        total_steps += steps
        print(f"  [{i+1}/{len(motion_files)}] {Path(motion_file).name}: {steps} timesteps")

    print(f"\nTotal timesteps: {total_steps}")

    # Get dimensions from first motion
    num_joints = motions[0]['joint_pos'].shape[1]
    num_bodies = motions[0]['body_pos_w'].shape[1]

    # Allocate concatenated arrays
    joint_pos_cat = np.zeros((total_steps, num_joints), dtype=np.float32)
    joint_vel_cat = np.zeros((total_steps, num_joints), dtype=np.float32)
    body_pos_w_cat = np.zeros((total_steps, num_bodies, 3), dtype=np.float32)
    body_quat_w_cat = np.zeros((total_steps, num_bodies, 4), dtype=np.float32)
    body_lin_vel_w_cat = np.zeros((total_steps, num_bodies, 3), dtype=np.float32)
    body_ang_vel_w_cat = np.zeros((total_steps, num_bodies, 3), dtype=np.float32)

    # Concatenate
    offset = 0
    for motion in motions:
        steps = motion['joint_pos'].shape[0]
        joint_pos_cat[offset:offset+steps] = motion['joint_pos']
        joint_vel_cat[offset:offset+steps] = motion['joint_vel']
        body_pos_w_cat[offset:offset+steps] = motion['body_pos_w']
        body_quat_w_cat[offset:offset+steps] = motion['body_quat_w']
        body_lin_vel_w_cat[offset:offset+steps] = motion['body_lin_vel_w']
        body_ang_vel_w_cat[offset:offset+steps] = motion['body_ang_vel_w']
        offset += steps

    return {
        'joint_pos': joint_pos_cat,
        'joint_vel': joint_vel_cat,
        'body_pos_w': body_pos_w_cat,
        'body_quat_w': body_quat_w_cat,
        'body_lin_vel_w': body_lin_vel_w_cat,
        'body_ang_vel_w': body_ang_vel_w_cat,
    }


def select_holonomous_motions(motion_dir: Path, strategy: str = "edges") -> list[str]:
    """Select motions that cover holonomous walking space.

    Args:
        motion_dir: Directory containing .npz motion files
        strategy: Selection strategy:
            - "edges": Select motions at edges of velocity space (forward, back, left, right, rotate)
            - "grid": Select a grid of motions covering velocity space
            - "all": Use all available motions

    Returns:
        List of motion file paths
    """
    import pickle

    # Parse all motion files to get velocities
    motion_files = list(motion_dir.glob("*.npz"))
    if not motion_files:
        raise ValueError(f"No .npz files found in {motion_dir}")

    # Parse velocities from filenames (format: dx_dy_dtheta.npz)
    motions_with_vel = []
    for f in motion_files:
        parts = f.stem.split("_")
        if len(parts) == 3:
            try:
                dx, dy, dtheta = float(parts[0]), float(parts[1]), float(parts[2])
                motions_with_vel.append((f, dx, dy, dtheta))
            except ValueError:
                print(f"Warning: Could not parse velocity from {f.name}")

    if not motions_with_vel:
        raise ValueError(f"No valid motion files with velocity info found in {motion_dir}")

    print(f"\nFound {len(motions_with_vel)} motions")

    if strategy == "all":
        print(f"Using all {len(motions_with_vel)} motions")
        return [str(f) for f, _, _, _ in motions_with_vel]

    elif strategy == "edges":
        # Select extremes in each dimension
        print("Selecting edge motions (max forward, back, left, right, rotate left/right)...")

        # Find extremes
        max_forward = max(motions_with_vel, key=lambda x: x[1])  # max dx
        max_backward = min(motions_with_vel, key=lambda x: x[1])  # min dx
        max_left = max(motions_with_vel, key=lambda x: x[2])  # max dy
        max_right = min(motions_with_vel, key=lambda x: x[2])  # min dy
        max_rotate_left = max(motions_with_vel, key=lambda x: x[3])  # max dtheta
        max_rotate_right = min(motions_with_vel, key=lambda x: x[3])  # min dtheta

        # Find standing (closest to zero)
        standing = min(motions_with_vel, key=lambda x: abs(x[1]) + abs(x[2]) + abs(x[3]))

        selected = [
            standing,
            max_forward,
            max_backward,
            max_left,
            max_right,
            max_rotate_left,
            max_rotate_right,
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_selected = []
        for motion in selected:
            if motion[0] not in seen:
                seen.add(motion[0])
                unique_selected.append(motion)

        print(f"\nSelected {len(unique_selected)} motions:")
        for f, dx, dy, dtheta in unique_selected:
            print(f"  {f.name}: dx={dx:6.2f}, dy={dy:6.2f}, dtheta={dtheta:6.2f}")

        return [str(f) for f, _, _, _ in unique_selected]

    elif strategy == "grid":
        # Select a grid covering velocity space
        print("Selecting grid of motions...")

        # Find velocity ranges
        dxs = sorted(set(dx for _, dx, _, _ in motions_with_vel))
        dys = sorted(set(dy for _, _, dy, _ in motions_with_vel))
        dthetas = sorted(set(dtheta for _, _, _, dtheta in motions_with_vel))

        # Sample from grid (e.g., take every 3rd value)
        stride = 3
        selected_dxs = dxs[::stride] if len(dxs) > stride else dxs
        selected_dys = dys[::stride] if len(dys) > stride else dys
        selected_dthetas = dthetas[::stride] if len(dthetas) > stride else dthetas

        # Add extremes if not included
        for extreme_list, all_list in [(selected_dxs, dxs), (selected_dys, dys), (selected_dthetas, dthetas)]:
            if all_list[0] not in extreme_list:
                extreme_list.insert(0, all_list[0])
            if all_list[-1] not in extreme_list:
                extreme_list.append(all_list[-1])

        print(f"Grid: {len(selected_dxs)} dx x {len(selected_dys)} dy x {len(selected_dthetas)} dtheta")

        # Find closest matches
        selected = []
        for dx_target in selected_dxs:
            for dy_target in selected_dys:
                for dtheta_target in selected_dthetas:
                    # Find closest motion
                    closest = min(
                        motions_with_vel,
                        key=lambda x: abs(x[1] - dx_target) + abs(x[2] - dy_target) + abs(x[3] - dtheta_target)
                    )
                    if closest not in selected:
                        selected.append(closest)

        print(f"\nSelected {len(selected)} motions from grid")

        return [str(f) for f, _, _, _ in selected]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate multiple BeyondMimic motion files for holonomous training"
    )
    parser.add_argument(
        "motion_dir",
        type=str,
        help="Directory containing .npz motion files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="holonomous_walk.npz",
        help="Output filename"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["edges", "grid", "all"],
        default="edges",
        help="Motion selection strategy (default: edges)"
    )
    parser.add_argument(
        "--motion-files",
        type=str,
        nargs="+",
        default=None,
        help="Specific motion files to concatenate (overrides strategy)"
    )
    args = parser.parse_args()

    motion_dir = Path(args.motion_dir)
    if not motion_dir.exists():
        raise ValueError(f"Motion directory not found: {motion_dir}")

    # Select motions
    if args.motion_files:
        print(f"Using {len(args.motion_files)} specified motion files")
        motion_files = [str(motion_dir / f) if not Path(f).is_absolute() else f
                       for f in args.motion_files]
    else:
        motion_files = select_holonomous_motions(motion_dir, args.strategy)

    # Concatenate
    print(f"\nConcatenating motions...")
    concatenated = concatenate_motions(motion_files)

    # Save
    output_path = Path(args.output)
    print(f"\nSaving to: {output_path}")
    np.savez_compressed(output_path, **concatenated)

    print(f"\n✓ Done!")
    print(f"  Output file: {output_path}")
    print(f"  Total timesteps: {concatenated['joint_pos'].shape[0]}")
    print(f"  Joints: {concatenated['joint_pos'].shape[1]}")
    print(f"  Bodies: {concatenated['body_pos_w'].shape[1]}")
    print(f"\nTo train:")
    print(f"  export MICRODUCK_BEYONDMIMIC_MOTION={output_path}")
    print(f"  uv run train Mjlab-BeyondMimic-MicroDuck --env.scene.num-envs 2048 --registry-name microduck-walk")


if __name__ == "__main__":
    main()
