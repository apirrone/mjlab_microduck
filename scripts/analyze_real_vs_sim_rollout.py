#!/usr/bin/env python3
"""
Analyze real vs sim rollout data to find the factor-of-2 discrepancy.
Compares observations and actions from real robot vs simulation replay.
"""

import pickle
import numpy as np


def load_observations(pkl_path: str):
    """Load observations from pickle file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        observations = np.array(data['observations'])
        timestamps = np.array(data['timestamps'])
    elif isinstance(data, list):
        if isinstance(data[0], dict) and 'timestamp' in data[0] and 'observation' in data[0]:
            timestamps = np.array([item['timestamp'] for item in data])
            observations = np.array([item['observation'] for item in data])
        else:
            observations = np.array(data)
            timestamps = np.arange(len(observations)) * 0.02
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")

    return observations, timestamps


def extract_obs_components(obs, use_imitation=False):
    """Extract observation components based on observation structure."""

    if use_imitation:
        # Imitation: command(3) + phase(1) + ang_vel(3) + proj_grav(3) + joint_pos(14) + joint_vel(14) + actions(14) = 52
        command = obs[:, 0:3]
        phase = obs[:, 3:4]
        ang_vel = obs[:, 4:7]
        proj_grav = obs[:, 7:10]
        joint_pos = obs[:, 10:24]
        joint_vel = obs[:, 24:38]
        actions = obs[:, 38:52]
    else:
        # Velocity: ang_vel(3) + proj_grav(3) + joint_pos(14) + joint_vel(14) + actions(14) + command(3) = 51
        ang_vel = obs[:, 0:3]
        proj_grav = obs[:, 3:6]
        joint_pos = obs[:, 6:20]
        joint_vel = obs[:, 20:34]
        actions = obs[:, 34:48]
        command = obs[:, 48:51]
        phase = None

    return {
        'command': command,
        'phase': phase,
        'ang_vel': ang_vel,
        'proj_grav': proj_grav,
        'joint_pos': joint_pos,
        'joint_vel': joint_vel,
        'actions': actions,
    }


def compare_statistics(real_data, sim_data, name):
    """Compare statistics between real and sim data."""
    real_mean = np.mean(real_data, axis=0)
    sim_mean = np.mean(sim_data, axis=0)

    real_std = np.std(real_data, axis=0)
    sim_std = np.std(sim_data, axis=0)

    real_min = np.min(real_data, axis=0)
    real_max = np.max(real_data, axis=0)
    sim_min = np.min(sim_data, axis=0)
    sim_max = np.max(sim_data, axis=0)

    # Compute ratios (avoid division by zero)
    mean_ratio = np.where(np.abs(sim_mean) > 1e-6, real_mean / sim_mean, 1.0)
    std_ratio = np.where(sim_std > 1e-6, real_std / sim_std, 1.0)
    range_ratio = np.where(
        (sim_max - sim_min) > 1e-6,
        (real_max - real_min) / (sim_max - sim_min),
        1.0
    )

    print(f"\n{name}:")
    print(f"  Shape: {real_data.shape}")
    print(f"  Real: mean={np.mean(np.abs(real_mean)):.4f}, std={np.mean(real_std):.4f}, range=[{np.mean(real_min):.4f}, {np.mean(real_max):.4f}]")
    print(f"  Sim:  mean={np.mean(np.abs(sim_mean)):.4f}, std={np.mean(sim_std):.4f}, range=[{np.mean(sim_min):.4f}, {np.mean(sim_max):.4f}]")
    print(f"  Mean ratio (real/sim): {np.mean(np.abs(mean_ratio)):.4f}")
    print(f"  Std ratio (real/sim):  {np.mean(std_ratio):.4f}")
    print(f"  Range ratio (real/sim): {np.mean(range_ratio):.4f}")

    # Check for factor of 2
    if np.abs(np.mean(std_ratio) - 2.0) < 0.2:
        print(f"  ⚠️  WARNING: Real std is ~2x sim std!")
    if np.abs(np.mean(std_ratio) - 0.5) < 0.1:
        print(f"  ⚠️  WARNING: Real std is ~0.5x sim std!")

    return mean_ratio, std_ratio, range_ratio


def main():
    print("="*80)
    print("REAL VS SIM ROLLOUT ANALYSIS")
    print("="*80)

    # Load data
    print("\nLoading data...")
    real_obs, real_ts = load_observations("best_walk_real_hang.pkl")
    sim_obs, sim_ts = load_observations("best_walk_replay_hang.pkl")

    print(f"Real observations: {real_obs.shape} (duration: {real_ts[-1]:.2f}s)")
    print(f"Sim observations:  {sim_obs.shape} (duration: {sim_ts[-1]:.2f}s)")

    # Determine observation type (velocity=51, imitation=52)
    obs_dim = real_obs.shape[1]
    use_imitation = (obs_dim >= 52)
    print(f"\nObservation type: {'Imitation' if use_imitation else 'Velocity'} (dim={obs_dim})")

    # Extract components
    real_comp = extract_obs_components(real_obs, use_imitation)
    sim_comp = extract_obs_components(sim_obs, use_imitation)

    # Truncate to same length
    min_len = min(len(real_obs), len(sim_obs))
    print(f"\nTruncating to {min_len} samples for comparison")

    print("\n" + "="*80)
    print("COMPONENT-WISE COMPARISON")
    print("="*80)

    # Compare each component
    results = {}
    for key in ['command', 'ang_vel', 'proj_grav', 'joint_pos', 'joint_vel', 'actions']:
        if key == 'phase' and real_comp[key] is None:
            continue

        real_data = real_comp[key][:min_len]
        sim_data = sim_comp[key][:min_len]

        mean_ratio, std_ratio, range_ratio = compare_statistics(real_data, sim_data, key.upper())
        results[key] = {
            'mean_ratio': mean_ratio,
            'std_ratio': std_ratio,
            'range_ratio': range_ratio,
        }

    # Special analysis for actions (the key culprit)
    print("\n" + "="*80)
    print("DETAILED ACTION ANALYSIS")
    print("="*80)

    real_actions = real_comp['actions'][:min_len]
    sim_actions = sim_comp['actions'][:min_len]

    print(f"\nAction statistics (these should be IDENTICAL - same actions replayed):")
    print(f"  Real actions: mean(abs)={np.mean(np.abs(real_actions)):.4f}, std={np.std(real_actions):.4f}")
    print(f"  Sim actions:  mean(abs)={np.mean(np.abs(sim_actions)):.4f}, std={np.std(sim_actions):.4f}")
    print(f"  Difference:   mean(abs)={np.mean(np.abs(real_actions - sim_actions)):.6f}")

    if np.mean(np.abs(real_actions - sim_actions)) > 0.01:
        print("  ⚠️  WARNING: Actions are NOT identical between real and sim!")
        print("  This suggests the replay didn't use the exact same actions.")
    else:
        print("  ✓ Actions are identical (as expected)")

    # Compare resulting joint positions
    print("\n" + "="*80)
    print("JOINT POSITION RESPONSE COMPARISON")
    print("="*80)

    real_joint_pos = real_comp['joint_pos'][:min_len]
    sim_joint_pos = sim_comp['joint_pos'][:min_len]

    print(f"\nJoint positions (relative to default):")
    print(f"  Real: mean(abs)={np.mean(np.abs(real_joint_pos)):.4f}, std={np.std(real_joint_pos):.4f}")
    print(f"  Sim:  mean(abs)={np.mean(np.abs(sim_joint_pos)):.4f}, std={np.std(sim_joint_pos):.4f}")
    print(f"  Ratio (real/sim): {np.mean(np.abs(real_joint_pos)) / np.mean(np.abs(sim_joint_pos)):.4f}")

    # Check per-joint
    print("\nPer-joint position comparison (real / sim ratio):")
    joint_names = [
        'L_hip_yaw', 'L_hip_roll', 'L_hip_pitch', 'L_knee', 'L_ankle',
        'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
        'R_hip_yaw', 'R_hip_roll', 'R_hip_pitch', 'R_knee', 'R_ankle'
    ]

    for i, name in enumerate(joint_names):
        real_std = np.std(real_joint_pos[:, i])
        sim_std = np.std(sim_joint_pos[:, i])
        ratio = real_std / sim_std if sim_std > 1e-6 else 1.0

        flag = ""
        if abs(ratio - 2.0) < 0.2:
            flag = " ⚠️ ~2x"
        elif abs(ratio - 0.5) < 0.1:
            flag = " ⚠️ ~0.5x"

        print(f"  {name:15s}: real_std={real_std:.4f}, sim_std={sim_std:.4f}, ratio={ratio:.3f}{flag}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("""
If you see:
1. Actions are identical → Good, replay worked correctly
2. Joint positions real/sim ratio ≈ 1.0 → Dynamics match (from knee tests)
3. Some other observation has ratio ≈ 2.0 → THAT'S the culprit!

The observation that has a factor of 2 difference is causing the policy
to output actions that are 2x too large/small to compensate.
    """)


if __name__ == "__main__":
    main()
