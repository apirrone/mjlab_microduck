#!/usr/bin/env python3
"""
Plot comparison between real robot observations and simulated observations.
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_observations(pkl_path: str):
    """Load observations from pickle file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        if 'observations' in data and 'timestamps' in data:
            observations = data['observations']
            timestamps = data['timestamps']
        else:
            raise ValueError("Dictionary must contain 'observations' and 'timestamps' keys")
    elif isinstance(data, list):
        if len(data) == 0:
            raise ValueError("Empty data list")

        if isinstance(data[0], dict) and 'timestamp' in data[0] and 'observation' in data[0]:
            timestamps = [item['timestamp'] for item in data]
            observations = [item['observation'] for item in data]
        elif isinstance(data[0], tuple):
            timestamps = [item[0] for item in data]
            observations = [item[1] for item in data]
        else:
            observations = data
            timestamps = [i * 0.02 for i in range(len(observations))]
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")

    return np.array(observations), np.array(timestamps)


def plot_comparison(real_obs, real_ts, sim_obs, sim_ts):
    """
    Plot comparison between real and simulated observations.

    Observation structure (51D or 53D):
    [0:3]    - Base angular velocity
    [3:6]    - Projected gravity
    [6:20]   - Joint positions relative (14D)
    [20:34]  - Joint velocities (14D)
    [34:48]  - Last action (14D)
    [48:51]  - Velocity command (3D)
    [51:53]  - Imitation phase [optional] (2D)
    """

    # Joint names
    joint_names = [
        'L_hip_yaw', 'L_hip_roll', 'L_hip_pitch', 'L_knee', 'L_ankle',
        'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
        'R_hip_yaw', 'R_hip_roll', 'R_hip_pitch', 'R_knee', 'R_ankle'
    ]

    # Determine the minimum observation dimension
    obs_dim = min(real_obs.shape[1], sim_obs.shape[1])

    # Calculate grid size:
    # 3 (base ang vel) + 3 (gravity) + 14 (joint pos) + 14 (joint vel) + 14 (actions) + 3 (vel cmd) = 51
    # Use 4 columns
    ncols = 4
    nrows = 15  # Enough for all components

    # Create figure with subplots (adjusted to make each subplot square)
    # For square subplots: width/ncols = height/nrows, so width = height * (ncols/nrows)
    subplot_size = 5  # inches per subplot
    fig_width = subplot_size * ncols
    fig_height = subplot_size * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    fig.suptitle('Real vs Simulated Observations Comparison', fontsize=20, y=0.998)

    # Add section separators (horizontal lines)
    # Calculate y-positions for section dividers (in figure coordinates)
    section_positions = {
        'Base Angular Velocity': 0.97,
        'Projected Gravity': 0.93,
        'Joint Positions': 0.89,
        'Joint Velocities': 0.62,
        'Actions': 0.35,
    }
    axes = axes.flatten()

    # Create figure-level legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='b', linestyle='-', linewidth=2, label='Real'),
        Line2D([0], [0], color='r', linestyle='--', linewidth=2, label='Sim')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)

    plot_idx = 0

    # Track axes for each group to set common y-limits
    base_ang_vel_axes = []
    gravity_axes = []
    joint_pos_axes = []
    joint_vel_axes = []
    action_axes = []

    # 1. Base angular velocity (3 subplots)
    for i, label in enumerate(['ω_x', 'ω_y', 'ω_z']):
        ax = axes[plot_idx]
        ax.plot(real_ts, real_obs[:, i], 'b-', alpha=0.7, linewidth=1)
        ax.plot(sim_ts, sim_obs[:, i], 'r--', alpha=0.7, linewidth=1)
        ax.set_title(f'Base {label}', fontsize=9)
        ax.set_ylabel('rad/s', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        base_ang_vel_axes.append(ax)
        plot_idx += 1

    # Empty slot
    axes[plot_idx].axis('off')
    plot_idx += 1

    # 2. Projected gravity (3 subplots)
    for i, label in enumerate(['g_x', 'g_y', 'g_z']):
        ax = axes[plot_idx]
        ax.plot(real_ts, real_obs[:, 3+i], 'b-', label='Real', alpha=0.7, linewidth=1)
        ax.plot(sim_ts, sim_obs[:, 3+i], 'r--', label='Sim', alpha=0.7, linewidth=1)
        ax.set_title(f'Gravity {label}', fontsize=9)
        ax.set_ylabel('g', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        gravity_axes.append(ax)
        plot_idx += 1

    # Empty slot
    axes[plot_idx].axis('off')
    plot_idx += 1

    # 3. Joint positions (14 subplots)
    for i in range(14):
        ax = axes[plot_idx]
        if 6 + i < obs_dim:
            ax.plot(real_ts, real_obs[:, 6+i], 'b-', alpha=0.7, linewidth=1)
            ax.plot(sim_ts, sim_obs[:, 6+i], 'r--', alpha=0.7, linewidth=1)
        ax.set_title(f'{joint_names[i]} pos', fontsize=8)
        ax.set_ylabel('rad', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
        joint_pos_axes.append(ax)
        plot_idx += 1

    # Skip 2 empty slots
    for _ in range(2):
        axes[plot_idx].axis('off')
        plot_idx += 1

    # 4. Joint velocities (14 subplots)
    for i in range(14):
        ax = axes[plot_idx]
        if 20 + i < obs_dim:
            ax.plot(real_ts, real_obs[:, 20+i], 'b-', alpha=0.7, linewidth=1)
            ax.plot(sim_ts, sim_obs[:, 20+i], 'r--', alpha=0.7, linewidth=1)
        ax.set_title(f'{joint_names[i]} vel', fontsize=8)
        ax.set_ylabel('rad/s', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
        joint_vel_axes.append(ax)
        plot_idx += 1

    # Skip 2 empty slots
    for _ in range(2):
        axes[plot_idx].axis('off')
        plot_idx += 1

    # 5. Last action (14 subplots)
    for i in range(14):
        ax = axes[plot_idx]
        if 34 + i < obs_dim:
            ax.plot(real_ts, real_obs[:, 34+i], 'b-', alpha=0.7, linewidth=1)
            ax.plot(sim_ts, sim_obs[:, 34+i], 'r--', alpha=0.7, linewidth=1)
        ax.set_title(f'{joint_names[i]} action', fontsize=8)
        ax.set_ylabel('action', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('Time (s)', fontsize=7)
        action_axes.append(ax)
        plot_idx += 1

    # Turn off any remaining axes
    while plot_idx < len(axes):
        axes[plot_idx].axis('off')
        plot_idx += 1

    # Set common y-limits for each group
    def set_common_ylim(ax_list):
        if not ax_list:
            return
        all_ylims = [ax.get_ylim() for ax in ax_list]
        global_min = min(ylim[0] for ylim in all_ylims)
        global_max = max(ylim[1] for ylim in all_ylims)
        for ax in ax_list:
            ax.set_ylim(global_min, global_max)

    set_common_ylim(base_ang_vel_axes)
    set_common_ylim(gravity_axes)
    set_common_ylim(joint_pos_axes)
    set_common_ylim(joint_vel_axes)
    set_common_ylim(action_axes)

    # Add section labels and separators
    for section_name, y_pos in section_positions.items():
        # Add section title on the left
        fig.text(0.01, y_pos, section_name, fontsize=16, fontweight='bold',
                va='center', ha='left', transform=fig.transFigure,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        # Add horizontal line separator
        line = plt.Line2D([0.08, 0.99], [y_pos, y_pos], transform=fig.transFigure,
                         color='gray', linewidth=2, linestyle='-', alpha=0.5)
        fig.add_artist(line)

    plt.tight_layout()
    plt.show()


def compute_metrics(real_obs, real_ts, sim_obs, sim_ts):
    """Compute comparison metrics between real and simulated observations."""

    # Interpolate sim observations to match real timestamps
    from scipy.interpolate import interp1d

    min_dim = min(real_obs.shape[1], sim_obs.shape[1])

    # Only compare where timestamps overlap
    min_time = max(real_ts[0], sim_ts[0])
    max_time = min(real_ts[-1], sim_ts[-1])

    real_mask = (real_ts >= min_time) & (real_ts <= max_time)
    real_ts_clipped = real_ts[real_mask]
    real_obs_clipped = real_obs[real_mask, :min_dim]

    # Interpolate sim to match real timestamps
    sim_obs_interp = np.zeros((len(real_ts_clipped), min_dim))
    for i in range(min_dim):
        f = interp1d(sim_ts, sim_obs[:, i], kind='linear', bounds_error=False, fill_value='extrapolate')
        sim_obs_interp[:, i] = f(real_ts_clipped)

    # Compute RMSE for different observation components
    components = {
        'Base Angular Velocity': (0, 3),
        'Projected Gravity': (3, 6),
        'Joint Positions': (6, 20),
        'Joint Velocities': (20, 34),
        'Last Action': (34, 48),
    }

    print("\n" + "="*60)
    print("Comparison Metrics (RMSE)")
    print("="*60)

    for name, (start, end) in components.items():
        if end <= min_dim:
            diff = real_obs_clipped[:, start:end] - sim_obs_interp[:, start:end]
            rmse = np.sqrt(np.mean(diff**2))
            rmse_per_dim = np.sqrt(np.mean(diff**2, axis=0))
            print(f"\n{name}:")
            print(f"  Overall RMSE: {rmse:.6f}")
            print(f"  Per-dimension RMSE: min={rmse_per_dim.min():.6f}, "
                  f"max={rmse_per_dim.max():.6f}, mean={rmse_per_dim.mean():.6f}")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Compare real and simulated observations"
    )
    parser.add_argument("real_pkl", type=str,
                       help="Path to .pkl file with real robot observations")
    parser.add_argument("sim_pkl", type=str,
                       help="Path to .pkl file with simulated observations")
    parser.add_argument("--no-metrics", action="store_true",
                       help="Skip computing comparison metrics")

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.real_pkl).exists():
        print(f"Error: {args.real_pkl} not found")
        return 1

    if not Path(args.sim_pkl).exists():
        print(f"Error: {args.sim_pkl} not found")
        return 1

    # Load observations
    print(f"Loading real observations from {args.real_pkl}...")
    real_obs, real_ts = load_observations(args.real_pkl)
    print(f"Loaded {len(real_obs)} real observations (shape: {real_obs.shape})")

    print(f"Loading simulated observations from {args.sim_pkl}...")
    sim_obs, sim_ts = load_observations(args.sim_pkl)
    print(f"Loaded {len(sim_obs)} simulated observations (shape: {sim_obs.shape})")

    # Compute metrics
    if not args.no_metrics:
        try:
            compute_metrics(real_obs, real_ts, sim_obs, sim_ts)
        except ImportError:
            print("\nWarning: scipy not available, skipping metrics computation")

    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_comparison(real_obs, real_ts, sim_obs, sim_ts)

    return 0


if __name__ == "__main__":
    exit(main())
