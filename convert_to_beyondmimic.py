#!/usr/bin/env python3
"""Convert polynomial reference motions to BeyondMimic format.

BeyondMimic expects .npz files with:
- joint_pos: (time_steps, num_joints)
- joint_vel: (time_steps, num_joints)
- body_pos_w: (time_steps, num_bodies, 3)
- body_quat_w: (time_steps, num_bodies, 4)
- body_lin_vel_w: (time_steps, num_bodies, 3)
- body_ang_vel_w: (time_steps, num_bodies, 3)
"""

import argparse
import pickle
import numpy as np
import mujoco
from pathlib import Path

MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene.xml"


def evaluate_polynomial(coeffs, phase):
    """Evaluate polynomial at given phase using Horner's method."""
    result = 0.0
    for i in range(len(coeffs) - 1, -1, -1):
        result = result * phase + coeffs[i]
    return result


def convert_motion_to_beyondmimic(
    motion_key: str,
    motion_data: dict,
    model: mujoco.MjModel,
    fps: int = 50,
    body_names: tuple[str, ...] = ("trunk_base",),
) -> dict:
    """Convert a single motion to BeyondMimic format.

    Args:
        motion_key: Motion identifier (e.g., "0.01_0.0_-0.4")
        motion_data: Dictionary with 'period', 'coefficients', etc.
        model: MuJoCo model for forward kinematics
        fps: Sampling frequency (Hz)
        body_names: Body names to track

    Returns:
        Dictionary with arrays ready for .npz saving
    """
    period = motion_data['period']
    coeffs = motion_data['coefficients']
    num_dims = len(coeffs)

    # Number of timesteps
    num_steps = int(period * fps) + 1  # +1 to include full cycle
    dt = 1.0 / fps

    # Get body IDs
    body_ids = []
    for body_name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found in model")
        body_ids.append(body_id)

    # Number of joints (14 for microduck)
    num_joints = 14

    # Allocate arrays
    joint_pos_arr = np.zeros((num_steps, num_joints), dtype=np.float32)
    joint_vel_arr = np.zeros((num_steps, num_joints), dtype=np.float32)
    body_pos_w_arr = np.zeros((num_steps, len(body_names), 3), dtype=np.float32)
    body_quat_w_arr = np.zeros((num_steps, len(body_names), 4), dtype=np.float32)
    body_lin_vel_w_arr = np.zeros((num_steps, len(body_names), 3), dtype=np.float32)
    body_ang_vel_w_arr = np.zeros((num_steps, len(body_names), 3), dtype=np.float32)

    # Create data for forward kinematics
    data = mujoco.MjData(model)

    # Get freejoint address
    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = model.jnt_qposadr[freejoint_id]
    qvel_adr = model.jnt_dofadr[freejoint_id]

    for step in range(num_steps):
        # Compute phase [0, 1]
        phase = (step * dt) / period
        if phase > 1.0:
            phase = 1.0

        # Evaluate polynomial at this phase
        results = np.zeros(num_dims, dtype=np.float32)
        for dim in range(num_dims):
            poly_coeffs = np.array(coeffs[f"dim_{dim}"], dtype=np.float32)
            results[dim] = evaluate_polynomial(poly_coeffs, phase)

        # Parse results
        # Order: joints_pos (14), joints_vel (14), foot_contacts (2),
        #        base_linear_vel (3), base_angular_vel (3)
        idx = 0
        joints_pos = results[idx:idx+14]
        idx += 14
        joints_vel = results[idx:idx+14]
        idx += 14
        foot_contacts = results[idx:idx+2]
        idx += 2
        base_linear_vel = results[idx:idx+3]
        idx += 3
        base_angular_vel = results[idx:idx+3]

        # Set joint positions and velocities
        joint_pos_arr[step] = joints_pos
        joint_vel_arr[step] = joints_vel

        # Set robot state in MuJoCo
        # Base position (keep at origin, we only care about body transformations)
        data.qpos[qpos_adr:qpos_adr+3] = [0, 0, 0.125]  # x, y, z
        data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]  # identity quaternion

        # Joint positions
        data.qpos[qpos_adr+7:qpos_adr+7+num_joints] = joints_pos

        # Base velocities
        data.qvel[qvel_adr:qvel_adr+3] = base_linear_vel
        data.qvel[qvel_adr+3:qvel_adr+6] = base_angular_vel

        # Joint velocities
        data.qvel[qvel_adr+6:qvel_adr+6+num_joints] = joints_vel

        # Forward kinematics
        mujoco.mj_forward(model, data)

        # Extract body states
        for i, body_id in enumerate(body_ids):
            body_pos_w_arr[step, i] = data.xpos[body_id]
            body_quat_w_arr[step, i] = data.xquat[body_id]  # w, x, y, z format

            # Get body velocities (6D: linear + angular)
            body_lin_vel_w_arr[step, i] = data.cvel[body_id, 3:6]
            body_ang_vel_w_arr[step, i] = data.cvel[body_id, 0:3]

    return {
        'joint_pos': joint_pos_arr,
        'joint_vel': joint_vel_arr,
        'body_pos_w': body_pos_w_arr,
        'body_quat_w': body_quat_w_arr,
        'body_lin_vel_w': body_lin_vel_w_arr,
        'body_ang_vel_w': body_ang_vel_w_arr,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert polynomial reference motions to BeyondMimic format"
    )
    parser.add_argument(
        "input_pkl",
        type=str,
        help="Input polynomial coefficients pickle file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./beyondmimic_motions",
        help="Output directory for .npz files"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Sampling frequency (Hz)"
    )
    parser.add_argument(
        "--body-names",
        type=str,
        nargs="+",
        default=["trunk_base"],
        help="Body names to track (space-separated)"
    )
    parser.add_argument(
        "--motion-keys",
        type=str,
        nargs="+",
        default=None,
        help="Specific motion keys to convert (default: all)"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load polynomial data
    print(f"Loading polynomial data from: {args.input_pkl}")
    with open(args.input_pkl, 'rb') as f:
        poly_data = pickle.load(f)

    print(f"Found {len(poly_data)} motions")

    # Load MuJoCo model
    print(f"Loading MuJoCo model from: {MICRODUCK_XML}")
    model = mujoco.MjModel.from_xml_path(MICRODUCK_XML)

    # Determine which motions to convert
    if args.motion_keys:
        motion_keys = args.motion_keys
        # Validate keys
        for key in motion_keys:
            if key not in poly_data:
                print(f"Warning: Motion key '{key}' not found in data")
                motion_keys.remove(key)
    else:
        motion_keys = list(poly_data.keys())

    print(f"\nConverting {len(motion_keys)} motions...")
    print(f"Output directory: {output_dir}")
    print(f"Sampling rate: {args.fps} Hz")
    print(f"Body names: {args.body_names}")
    print()

    # Convert each motion
    for i, motion_key in enumerate(motion_keys):
        motion_data = poly_data[motion_key]
        print(f"[{i+1}/{len(motion_keys)}] Converting {motion_key}...")
        print(f"  Period: {motion_data['period']:.3f}s")
        print(f"  dx={motion_data['Placo']['dx']:.3f}, "
              f"dy={motion_data['Placo']['dy']:.3f}, "
              f"dtheta={motion_data['Placo']['dtheta']:.3f}")

        try:
            beyondmimic_data = convert_motion_to_beyondmimic(
                motion_key,
                motion_data,
                model,
                fps=args.fps,
                body_names=tuple(args.body_names)
            )

            # Save as .npz
            output_file = output_dir / f"{motion_key}.npz"
            np.savez_compressed(output_file, **beyondmimic_data)

            print(f"  ✓ Saved to: {output_file}")
            print(f"    Timesteps: {beyondmimic_data['joint_pos'].shape[0]}")
            print(f"    Bodies: {beyondmimic_data['body_pos_w'].shape[1]}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\nDone! Converted {len(motion_keys)} motions to {output_dir}")


if __name__ == "__main__":
    main()
