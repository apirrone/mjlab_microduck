#!/usr/bin/env python3
"""
Analyze real vs sim replay recordings to validate dynamics matching.
"""

import pickle
import numpy as np
from pathlib import Path


def load_replay_data(pkl_path):
    """Load replay data from pickle file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Handle different pickle formats
    if isinstance(data, dict) and 'observations' in data and 'timestamps' in data:
        # Format: {'observations': array, 'timestamps': array}
        observations = data['observations']
        timestamps = data['timestamps']
    elif isinstance(data, list):
        # Format: [{'timestamp': ..., 'observation': ...}, ...]
        timestamps = np.array([item['timestamp'] for item in data])
        observations = np.array([item['observation'] for item in data])
    else:
        raise ValueError(f"Unsupported pickle format: {type(data)}")

    return timestamps, observations


def compute_metrics(real_data, sim_data, name):
    """Compute RMS error, correlation, and other metrics."""
    # Handle potential length mismatch
    min_len = min(len(real_data), len(sim_data))
    real_data = real_data[:min_len]
    sim_data = sim_data[:min_len]

    # RMS error
    rms_error = np.sqrt(np.mean((real_data - sim_data) ** 2))

    # Per-dimension RMS if multi-dimensional
    if len(real_data.shape) > 1:
        rms_per_dim = np.sqrt(np.mean((real_data - sim_data) ** 2, axis=0))
    else:
        rms_per_dim = None

    # Correlation
    if real_data.std() > 1e-6 and sim_data.std() > 1e-6:
        correlation = np.corrcoef(real_data.flatten(), sim_data.flatten())[0, 1]
    else:
        correlation = np.nan

    # Mean absolute error
    mae = np.mean(np.abs(real_data - sim_data))

    # Max error
    max_error = np.max(np.abs(real_data - sim_data))

    # Statistics
    real_mean = np.mean(real_data)
    sim_mean = np.mean(sim_data)
    real_std = np.std(real_data)
    sim_std = np.std(sim_data)

    return {
        'name': name,
        'rms_error': rms_error,
        'rms_per_dim': rms_per_dim,
        'correlation': correlation,
        'mae': mae,
        'max_error': max_error,
        'real_mean': real_mean,
        'sim_mean': sim_mean,
        'real_std': real_std,
        'sim_std': sim_std,
    }


def main():
    # Load data
    real_pkl = Path("/home/antoine/MISC/mjlab_microduck/real_1_hang.pkl")
    sim_pkl = Path("/home/antoine/MISC/mjlab_microduck/sim_1_hang.pkl")

    if not real_pkl.exists():
        print(f"Error: {real_pkl} not found")
        return 1
    if not sim_pkl.exists():
        print(f"Error: {sim_pkl} not found")
        return 1

    print("Loading real replay data...")
    real_ts, real_obs = load_replay_data(real_pkl)
    print(f"Loaded {len(real_obs)} real observations (shape: {real_obs.shape})")

    print("Loading sim replay data...")
    sim_ts, sim_obs = load_replay_data(sim_pkl)
    print(f"Loaded {len(sim_obs)} sim observations (shape: {sim_obs.shape})")

    # Ensure same length for comparison
    min_len = min(len(real_obs), len(sim_obs))
    real_obs = real_obs[:min_len]
    sim_obs = sim_obs[:min_len]
    real_ts = real_ts[:min_len]
    sim_ts = sim_ts[:min_len]

    print(f"\nComparing first {min_len} timesteps\n")

    # Observation structure (53D imitation):
    # [0:3]    - Velocity command (3D)
    # [3:5]    - Phase (2D) - [cos(2π*phase), sin(2π*phase)]
    # [5:8]    - Base angular velocity (3D) - gyro
    # [8:11]   - Raw accelerometer (3D)
    # [11:25]  - Joint positions relative (14D)
    # [25:39]  - Joint velocities (14D)
    # [39:53]  - Last action (14D)

    joint_names = [
        'L_hip_yaw', 'L_hip_roll', 'L_hip_pitch', 'L_knee', 'L_ankle',
        'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
        'R_hip_yaw', 'R_hip_roll', 'R_hip_pitch', 'R_knee', 'R_ankle'
    ]

    # Compute metrics for different observation components
    print("="*80)
    print("DYNAMICS MATCHING ANALYSIS")
    print("="*80)

    # 1. Actions (should be identical in replay mode - sanity check)
    print("\n1. ACTIONS (should be identical - sanity check)")
    print("-" * 80)
    actions_metrics = compute_metrics(real_obs[:, 39:53], sim_obs[:, 39:53], "Actions")
    print(f"RMS Error: {actions_metrics['rms_error']:.6f}")
    print(f"Max Error: {actions_metrics['max_error']:.6f}")
    print(f"Correlation: {actions_metrics['correlation']:.6f}")
    if actions_metrics['rms_error'] > 0.001:
        print("⚠️  WARNING: Actions don't match! Replay may not be working correctly.")
    else:
        print("✓ Actions match (replay is working)")

    # 2. Joint positions (critical for motor control validation)
    print("\n2. JOINT POSITIONS (critical for motor control)")
    print("-" * 80)
    joint_pos_metrics = compute_metrics(real_obs[:, 11:25], sim_obs[:, 11:25], "Joint Positions")
    print(f"Overall RMS Error: {joint_pos_metrics['rms_error']:.6f} rad ({np.rad2deg(joint_pos_metrics['rms_error']):.3f}°)")
    print(f"Overall MAE: {joint_pos_metrics['mae']:.6f} rad ({np.rad2deg(joint_pos_metrics['mae']):.3f}°)")
    print(f"Overall Max Error: {joint_pos_metrics['max_error']:.6f} rad ({np.rad2deg(joint_pos_metrics['max_error']):.3f}°)")
    print(f"Correlation: {joint_pos_metrics['correlation']:.6f}")
    print(f"\nReal mean: {joint_pos_metrics['real_mean']:.6f}, std: {joint_pos_metrics['real_std']:.6f}")
    print(f"Sim mean:  {joint_pos_metrics['sim_mean']:.6f}, std: {joint_pos_metrics['sim_std']:.6f}")

    # Per-joint breakdown
    print("\nPer-joint RMS errors:")
    for joint_name, rms in zip(joint_names, joint_pos_metrics['rms_per_dim']):
        print(f"  {joint_name:15s}: {rms:.6f} rad ({np.rad2deg(rms):.3f}°)")

    # 3. Joint velocities (validates dynamics/damping)
    print("\n3. JOINT VELOCITIES (validates dynamics/damping)")
    print("-" * 80)
    joint_vel_metrics = compute_metrics(real_obs[:, 25:39], sim_obs[:, 25:39], "Joint Velocities")
    print(f"Overall RMS Error: {joint_vel_metrics['rms_error']:.6f} rad/s ({np.rad2deg(joint_vel_metrics['rms_error']):.3f}°/s)")
    print(f"Overall MAE: {joint_vel_metrics['mae']:.6f} rad/s ({np.rad2deg(joint_vel_metrics['mae']):.3f}°/s)")
    print(f"Overall Max Error: {joint_vel_metrics['max_error']:.6f} rad/s ({np.rad2deg(joint_vel_metrics['max_error']):.3f}°/s)")
    print(f"Correlation: {joint_vel_metrics['correlation']:.6f}")
    print(f"\nReal mean: {joint_vel_metrics['real_mean']:.6f}, std: {joint_vel_metrics['real_std']:.6f}")
    print(f"Sim mean:  {joint_vel_metrics['sim_mean']:.6f}, std: {joint_vel_metrics['sim_std']:.6f}")

    # Per-joint breakdown
    print("\nPer-joint RMS errors:")
    for joint_name, rms in zip(joint_names, joint_vel_metrics['rms_per_dim']):
        print(f"  {joint_name:15s}: {rms:.6f} rad/s ({np.rad2deg(rms):.3f}°/s)")

    # 4. IMU signals (expected to differ - body dynamics)
    print("\n4. IMU SIGNALS (expected to differ - body dynamics)")
    print("-" * 80)

    # Gyro
    gyro_metrics = compute_metrics(real_obs[:, 5:8], sim_obs[:, 5:8], "Gyro")
    print("Gyroscope (base angular velocity):")
    print(f"  RMS Error: {gyro_metrics['rms_error']:.6f} rad/s")
    print(f"  Correlation: {gyro_metrics['correlation']:.6f}")
    print(f"  Real std: {gyro_metrics['real_std']:.6f}, Sim std: {gyro_metrics['sim_std']:.6f}")

    # Accelerometer
    accel_metrics = compute_metrics(real_obs[:, 8:11], sim_obs[:, 8:11], "Accelerometer")
    print("\nAccelerometer (raw):")
    print(f"  RMS Error: {accel_metrics['rms_error']:.6f} g")
    print(f"  Correlation: {accel_metrics['correlation']:.6f}")
    print(f"  Real mean: {accel_metrics['real_mean']:.6f}, Sim mean: {accel_metrics['sim_mean']:.6f}")
    print(f"  Real std: {accel_metrics['real_std']:.6f}, Sim std: {accel_metrics['sim_std']:.6f}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if joint_pos_metrics['rms_error'] < 0.05:  # < ~3 degrees
        print("✓ Joint positions match well (motor control is accurate)")
    elif joint_pos_metrics['rms_error'] < 0.1:  # < ~6 degrees
        print("⚠️  Joint positions have moderate error (motor control needs tuning)")
    else:
        print("❌ Joint positions have large error (motor control mismatch)")

    if joint_vel_metrics['rms_error'] < 0.5:  # < ~30 deg/s
        print("✓ Joint velocities match reasonably (dynamics/damping are good)")
    elif joint_vel_metrics['rms_error'] < 1.0:  # < ~60 deg/s
        print("⚠️  Joint velocities have moderate error (dynamics may need tuning)")
    else:
        print("❌ Joint velocities have large error (dynamics mismatch)")

    print("\nNote: IMU differences are expected when robot is hanging (no ground contact)")
    print("The key validation is that joint-level control matches well.")

    return 0


if __name__ == "__main__":
    exit(main())
