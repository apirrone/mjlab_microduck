#!/usr/bin/env python3
"""
Plot comparison between real and simulated observations using Plotly.
"""

import argparse
import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    Plot comparison between real and simulated observations using Plotly.
    """

    # Joint names
    joint_names = [
        'L_hip_yaw', 'L_hip_roll', 'L_hip_pitch', 'L_knee', 'L_ankle',
        'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
        'R_hip_yaw', 'R_hip_roll', 'R_hip_pitch', 'R_knee', 'R_ankle'
    ]

    obs_dim = min(real_obs.shape[1], sim_obs.shape[1])

    # Create subplot titles with sections
    subplot_titles = []

    # Base angular velocity (3)
    subplot_titles.extend(['<b>BASE ANG VEL</b><br>ω_x', 'ω_y', 'ω_z', ''])

    # Projected gravity (3)
    subplot_titles.extend(['<b>PROJECTED GRAVITY</b><br>g_x', 'g_y', 'g_z', ''])

    # Joint positions (14 + 2 empty)
    subplot_titles.append(f'<b>JOINT POSITIONS</b><br>{joint_names[0]}')
    subplot_titles.extend(joint_names[1:14])
    subplot_titles.extend(['', ''])

    # Joint velocities (14 + 2 empty)
    subplot_titles.append(f'<b>JOINT VELOCITIES</b><br>{joint_names[0]}')
    subplot_titles.extend(joint_names[1:14])
    subplot_titles.extend(['', ''])

    # Actions (14 + 2 empty)
    subplot_titles.append(f'<b>ACTIONS</b><br>{joint_names[0]}')
    subplot_titles.extend(joint_names[1:14])
    subplot_titles.extend(['', ''])

    # Create subplots
    fig = make_subplots(
        rows=15, cols=4,
        subplot_titles=subplot_titles,
        vertical_spacing=0.02,
        horizontal_spacing=0.05,
        row_heights=[1]*15,
    )

    plot_idx = 0

    def add_traces(row, col, real_data, sim_data, y_range=None):
        """Helper to add real and sim traces to a subplot."""
        fig.add_trace(
            go.Scatter(x=real_ts, y=real_data, name='Real',
                      line=dict(color='blue', width=1.5),
                      showlegend=(plot_idx == 0)),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=sim_ts, y=sim_data, name='Sim',
                      line=dict(color='red', width=1.5, dash='dash'),
                      showlegend=(plot_idx == 0)),
            row=row, col=col
        )
        if y_range:
            fig.update_yaxes(range=y_range, row=row, col=col)

    # Track y-ranges for common scaling
    base_ang_vel_data = []
    gravity_data = []
    joint_pos_data = []
    joint_vel_data = []
    action_data = []

    # 1. Base angular velocity (3 subplots)
    for i in range(3):
        row, col = divmod(plot_idx, 4)
        row += 1
        col += 1
        base_ang_vel_data.extend([real_obs[:, i], sim_obs[:, i]])
        add_traces(row, col, real_obs[:, i], sim_obs[:, i])
        fig.update_yaxes(title_text='rad/s', row=row, col=col)
        plot_idx += 1

    # Empty slot
    plot_idx += 1

    # 2. Projected gravity (3 subplots)
    for i in range(3):
        row, col = divmod(plot_idx, 4)
        row += 1
        col += 1
        gravity_data.extend([real_obs[:, 3+i], sim_obs[:, 3+i]])
        add_traces(row, col, real_obs[:, 3+i], sim_obs[:, 3+i])
        fig.update_yaxes(title_text='g', row=row, col=col)
        plot_idx += 1

    # Empty slot
    plot_idx += 1

    # 3. Joint positions (14 subplots)
    for i in range(14):
        row, col = divmod(plot_idx, 4)
        row += 1
        col += 1
        if 6 + i < obs_dim:
            joint_pos_data.extend([real_obs[:, 6+i], sim_obs[:, 6+i]])
            add_traces(row, col, real_obs[:, 6+i], sim_obs[:, 6+i])
        fig.update_yaxes(title_text='rad', row=row, col=col)
        plot_idx += 1

    # Skip 2 empty slots
    plot_idx += 2

    # 4. Joint velocities (14 subplots)
    for i in range(14):
        row, col = divmod(plot_idx, 4)
        row += 1
        col += 1
        if 20 + i < obs_dim:
            joint_vel_data.extend([real_obs[:, 20+i], sim_obs[:, 20+i]])
            add_traces(row, col, real_obs[:, 20+i], sim_obs[:, 20+i])
        fig.update_yaxes(title_text='rad/s', row=row, col=col)
        plot_idx += 1

    # Skip 2 empty slots
    plot_idx += 2

    # 5. Actions (14 subplots)
    for i in range(14):
        row, col = divmod(plot_idx, 4)
        row += 1
        col += 1
        if 34 + i < obs_dim:
            action_data.extend([real_obs[:, 34+i], sim_obs[:, 34+i]])
            add_traces(row, col, real_obs[:, 34+i], sim_obs[:, 34+i])
        fig.update_yaxes(title_text='action', row=row, col=col)
        fig.update_xaxes(title_text='Time (s)', row=row, col=col)
        plot_idx += 1

    # Set common y-ranges for each group
    def compute_range(data_list):
        if not data_list:
            return None
        all_data = np.concatenate([d.flatten() for d in data_list])
        y_min, y_max = np.min(all_data), np.max(all_data)
        margin = (y_max - y_min) * 0.1
        return [y_min - margin, y_max + margin]

    base_ang_vel_range = compute_range(base_ang_vel_data)
    gravity_range = compute_range(gravity_data)
    joint_pos_range = compute_range(joint_pos_data)
    joint_vel_range = compute_range(joint_vel_data)
    action_range = compute_range(action_data)

    # Apply common ranges
    plot_idx = 0
    for i in range(3):  # Base ang vel
        row, col = divmod(plot_idx, 4)
        fig.update_yaxes(range=base_ang_vel_range, row=row+1, col=col+1)
        plot_idx += 1
    plot_idx += 1

    for i in range(3):  # Gravity
        row, col = divmod(plot_idx, 4)
        fig.update_yaxes(range=gravity_range, row=row+1, col=col+1)
        plot_idx += 1
    plot_idx += 1

    for i in range(14):  # Joint pos
        row, col = divmod(plot_idx, 4)
        fig.update_yaxes(range=joint_pos_range, row=row+1, col=col+1)
        plot_idx += 1
    plot_idx += 2

    for i in range(14):  # Joint vel
        row, col = divmod(plot_idx, 4)
        fig.update_yaxes(range=joint_vel_range, row=row+1, col=col+1)
        plot_idx += 1
    plot_idx += 2

    for i in range(14):  # Actions
        row, col = divmod(plot_idx, 4)
        fig.update_yaxes(range=action_range, row=row+1, col=col+1)
        plot_idx += 1

    # Update layout
    fig.update_layout(
        title_text='Real vs Simulated Observations Comparison',
        title_font_size=24,
        height=5000,  # Tall figure for all subplots
        width=1600,
        showlegend=True,
        legend=dict(x=0.85, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified'
    )

    fig.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare real and simulated observations (Plotly version)"
    )
    parser.add_argument("real_pkl", type=str,
                       help="Path to .pkl file with real robot observations")
    parser.add_argument("sim_pkl", type=str,
                       help="Path to .pkl file with simulated observations")

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

    # Plot comparison
    print("\nGenerating interactive comparison plots...")
    plot_comparison(real_obs, real_ts, sim_obs, sim_ts)

    return 0


if __name__ == "__main__":
    exit(main())
