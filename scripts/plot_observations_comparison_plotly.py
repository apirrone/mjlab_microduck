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


def plot_comparison(real_obs, real_ts, sim_obs=None, sim_ts=None, imitation=False):
    """
    Plot comparison between real and simulated observations using Plotly.
    If sim_obs is None, only plots real data.

    Args:
        imitation: If True, observations include command (3) + phase (1) at the start
    """

    # Joint names
    joint_names = [
        'L_hip_yaw', 'L_hip_roll', 'L_hip_pitch', 'L_knee', 'L_ankle',
        'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
        'R_hip_yaw', 'R_hip_roll', 'R_hip_pitch', 'R_knee', 'R_ankle'
    ]

    obs_dim = real_obs.shape[1] if sim_obs is None else min(real_obs.shape[1], sim_obs.shape[1])

    # Observation indices depend on whether this is imitation or velocity
    # Imitation (52D): command (3) + phase (1) + ang_vel (3) + proj_grav (3) + joint_pos (14) + joint_vel (14) + actions (14)
    # Velocity (51D): ang_vel (3) + proj_grav (3) + joint_pos (14) + joint_vel (14) + actions (14) + command (3)

    if imitation:
        base_ang_vel_start = 4
        gravity_start = 7
        joint_pos_start = 10
        joint_vel_start = 24
        action_start = 38
    else:
        base_ang_vel_start = 0
        gravity_start = 3
        joint_pos_start = 6
        joint_vel_start = 20
        action_start = 34

    # Create subplot titles with sections
    subplot_titles = []

    # Command and phase (only for imitation)
    if imitation:
        subplot_titles.extend(['<b>COMMAND</b><br>vel_x', 'vel_y', 'ang_z', ''])
        subplot_titles.extend(['<b>PHASE</b><br>phase', '', '', ''])

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

    # Create subplots (add 2 rows for imitation: command + phase)
    num_rows = 16 if imitation else 14
    fig = make_subplots(
        rows=num_rows, cols=4,
        subplot_titles=subplot_titles,
        vertical_spacing=0.02,
        horizontal_spacing=0.05,
        row_heights=[1]*num_rows,
    )

    plot_idx = 0

    # Track data for common scaling
    command_data = []
    phase_data = []

    def add_traces(row, col, real_data, sim_data=None, y_range=None):
        """Helper to add real and sim traces to a subplot."""
        fig.add_trace(
            go.Scatter(x=real_ts, y=real_data, name='Real',
                      line=dict(color='blue', width=1.5),
                      showlegend=(plot_idx == 0)),
            row=row, col=col
        )
        if sim_data is not None:
            fig.add_trace(
                go.Scatter(x=sim_ts, y=sim_data, name='Sim',
                          line=dict(color='red', width=1.5, dash='dash'),
                          showlegend=(plot_idx == 0)),
                row=row, col=col
            )
        if y_range:
            fig.update_yaxes(range=y_range, row=row, col=col)

    base_ang_vel_data = []
    gravity_data = []
    joint_pos_data = []
    joint_vel_data = []
    action_data = []

    # 0. Command (only for imitation, 3 subplots)
    if imitation:
        for i in range(3):
            row, col = divmod(plot_idx, 4)
            row += 1
            col += 1
            command_data.append(real_obs[:, i])
            if sim_obs is not None:
                command_data.append(sim_obs[:, i])
            add_traces(row, col, real_obs[:, i], None if sim_obs is None else sim_obs[:, i])
            fig.update_yaxes(title_text='m/s or rad/s', row=row, col=col)
            plot_idx += 1
        # Empty slot
        plot_idx += 1

        # Phase (1 subplot)
        row, col = divmod(plot_idx, 4)
        row += 1
        col += 1
        phase_data.append(real_obs[:, 3])
        if sim_obs is not None:
            phase_data.append(sim_obs[:, 3])
        add_traces(row, col, real_obs[:, 3], None if sim_obs is None else sim_obs[:, 3])
        fig.update_yaxes(title_text='phase', row=row, col=col)
        plot_idx += 1
        # Empty slots
        plot_idx += 3

    # 1. Base angular velocity (3 subplots)
    for i in range(3):
        row, col = divmod(plot_idx, 4)
        row += 1
        col += 1
        base_ang_vel_data.append(real_obs[:, base_ang_vel_start+i])
        if sim_obs is not None:
            base_ang_vel_data.append(sim_obs[:, base_ang_vel_start+i])
        add_traces(row, col, real_obs[:, base_ang_vel_start+i], None if sim_obs is None else sim_obs[:, base_ang_vel_start+i])
        fig.update_yaxes(title_text='rad/s', row=row, col=col)
        plot_idx += 1

    # Empty slot
    plot_idx += 1

    # 2. Projected gravity (3 subplots)
    for i in range(3):
        row, col = divmod(plot_idx, 4)
        row += 1
        col += 1
        gravity_data.append(real_obs[:, gravity_start+i])
        if sim_obs is not None:
            gravity_data.append(sim_obs[:, gravity_start+i])
        add_traces(row, col, real_obs[:, gravity_start+i], None if sim_obs is None else sim_obs[:, gravity_start+i])
        fig.update_yaxes(title_text='g', row=row, col=col)
        plot_idx += 1

    # Empty slot
    plot_idx += 1

    # 3. Joint positions (14 subplots)
    for i in range(14):
        row, col = divmod(plot_idx, 4)
        row += 1
        col += 1
        if joint_pos_start + i < obs_dim:
            joint_pos_data.append(real_obs[:, joint_pos_start+i])
            if sim_obs is not None:
                joint_pos_data.append(sim_obs[:, joint_pos_start+i])
            add_traces(row, col, real_obs[:, joint_pos_start+i], None if sim_obs is None else sim_obs[:, joint_pos_start+i])
        fig.update_yaxes(title_text='rad', row=row, col=col)
        plot_idx += 1

    # Skip 2 empty slots
    plot_idx += 2

    # 4. Joint velocities (14 subplots)
    for i in range(14):
        row, col = divmod(plot_idx, 4)
        row += 1
        col += 1
        if joint_vel_start + i < obs_dim:
            joint_vel_data.append(real_obs[:, joint_vel_start+i])
            if sim_obs is not None:
                joint_vel_data.append(sim_obs[:, joint_vel_start+i])
            add_traces(row, col, real_obs[:, joint_vel_start+i], None if sim_obs is None else sim_obs[:, joint_vel_start+i])
        fig.update_yaxes(title_text='rad/s', row=row, col=col)
        plot_idx += 1

    # Skip 2 empty slots
    plot_idx += 2

    # 5. Actions (14 subplots)
    for i in range(14):
        row, col = divmod(plot_idx, 4)
        row += 1
        col += 1
        if action_start + i < obs_dim:
            action_data.append(real_obs[:, action_start+i])
            if sim_obs is not None:
                action_data.append(sim_obs[:, action_start+i])
            add_traces(row, col, real_obs[:, action_start+i], None if sim_obs is None else sim_obs[:, action_start+i])
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

    command_range = compute_range(command_data) if command_data else None
    phase_range = compute_range(phase_data) if phase_data else None

    # Apply common ranges
    plot_idx = 0

    # Command ranges (only for imitation)
    if imitation:
        for i in range(3):
            row, col = divmod(plot_idx, 4)
            fig.update_yaxes(range=command_range, row=row+1, col=col+1)
            plot_idx += 1
        plot_idx += 1

        row, col = divmod(plot_idx, 4)
        fig.update_yaxes(range=phase_range, row=row+1, col=col+1)
        plot_idx += 1
        plot_idx += 3

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
    task_type = ' (Imitation)' if imitation else ' (Velocity)'
    title = 'Real vs Simulated Observations Comparison' if sim_obs is not None else 'Real Robot Observations'
    title += task_type

    height = 5200 if imitation else 4600  # Taller for imitation (extra rows)

    fig.update_layout(
        title_text=title,
        title_font_size=24,
        height=height,
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
    parser.add_argument("sim_pkl", type=str, nargs='?', default=None,
                       help="Path to .pkl file with simulated observations (optional)")
    parser.add_argument("--imitation", action='store_true',
                       help="Use imitation observation structure (includes command + phase)")

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.real_pkl).exists():
        print(f"Error: {args.real_pkl} not found")
        return 1

    # Load observations
    print(f"Loading real observations from {args.real_pkl}...")
    real_obs, real_ts = load_observations(args.real_pkl)
    print(f"Loaded {len(real_obs)} real observations (shape: {real_obs.shape})")

    if args.sim_pkl:
        if not Path(args.sim_pkl).exists():
            print(f"Error: {args.sim_pkl} not found")
            return 1
        print(f"Loading simulated observations from {args.sim_pkl}...")
        sim_obs, sim_ts = load_observations(args.sim_pkl)
        print(f"Loaded {len(sim_obs)} simulated observations (shape: {sim_obs.shape})")
    else:
        print("No sim data provided, plotting real data only")
        sim_obs, sim_ts = None, None

    # Plot comparison
    print(f"\nGenerating interactive comparison plots ({'imitation' if args.imitation else 'velocity'} mode)...")
    plot_comparison(real_obs, real_ts, sim_obs, sim_ts, imitation=args.imitation)

    return 0


if __name__ == "__main__":
    exit(main())
