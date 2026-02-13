#!/usr/bin/env python3
"""
Visualize real vs sim replay matching for a few key joints.
"""

import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def load_replay_data(pkl_path):
    """Load replay data from pickle file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Handle different pickle formats
    if isinstance(data, dict) and 'observations' in data and 'timestamps' in data:
        observations = data['observations']
        timestamps = data['timestamps']
    elif isinstance(data, list):
        timestamps = np.array([item['timestamp'] for item in data])
        observations = np.array([item['observation'] for item in data])
    else:
        raise ValueError(f"Unsupported pickle format: {type(data)}")

    return timestamps, observations


def main():
    # Load data
    real_pkl = Path("/home/antoine/MISC/mjlab_microduck/real_1_hang.pkl")
    sim_pkl = Path("/home/antoine/MISC/mjlab_microduck/sim_1_hang.pkl")

    print("Loading replay data...")
    real_ts, real_obs = load_replay_data(real_pkl)
    sim_ts, sim_obs = load_replay_data(sim_pkl)

    # Ensure same length
    min_len = min(len(real_obs), len(sim_obs))
    real_obs = real_obs[:min_len]
    sim_obs = sim_obs[:min_len]
    real_ts = real_ts[:min_len]

    # Joint names
    joint_names = [
        'L_hip_yaw', 'L_hip_roll', 'L_hip_pitch', 'L_knee', 'L_ankle',
        'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
        'R_hip_yaw', 'R_hip_roll', 'R_hip_pitch', 'R_knee', 'R_ankle'
    ]

    # Select joints to visualize (focus on worst offenders)
    selected_joints = [
        (3, 'L_knee'),      # Worst position error (8.2°)
        (4, 'L_ankle'),     # High position error (4.6°)
        (12, 'R_knee'),     # Worst position error (8.6°)
        (13, 'R_ankle'),    # High position error (5.6°)
    ]

    # Create subplots: 2 rows per joint (position + velocity)
    subplot_titles = []
    for _, name in selected_joints:
        subplot_titles.extend([f'{name} - Position', f'{name} - Velocity'])

    fig = make_subplots(
        rows=len(selected_joints) * 2,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.04,
    )

    row = 1
    for joint_idx, joint_name in selected_joints:
        # Position
        real_pos = real_obs[:, 11 + joint_idx]
        sim_pos = sim_obs[:, 11 + joint_idx]

        fig.add_trace(
            go.Scatter(x=real_ts, y=np.rad2deg(real_pos), name='Real',
                      line=dict(color='blue', width=2),
                      showlegend=(row == 1)),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=real_ts, y=np.rad2deg(sim_pos), name='Sim',
                      line=dict(color='red', width=2, dash='dash'),
                      showlegend=(row == 1)),
            row=row, col=1
        )
        fig.update_yaxes(title_text='Position (°)', row=row, col=1)

        # Velocity
        real_vel = real_obs[:, 25 + joint_idx]
        sim_vel = sim_obs[:, 25 + joint_idx]

        fig.add_trace(
            go.Scatter(x=real_ts, y=np.rad2deg(real_vel), name='Real',
                      line=dict(color='blue', width=2),
                      showlegend=False),
            row=row + 1, col=1
        )
        fig.add_trace(
            go.Scatter(x=real_ts, y=np.rad2deg(sim_vel), name='Sim',
                      line=dict(color='red', width=2, dash='dash'),
                      showlegend=False),
            row=row + 1, col=1
        )
        fig.update_yaxes(title_text='Velocity (°/s)', row=row + 1, col=1)
        fig.update_xaxes(title_text='Time (s)', row=row + 1, col=1)

        row += 2

    fig.update_layout(
        title_text='Replay Matching: Real vs Sim (Key Joints with Largest Errors)',
        title_font_size=20,
        height=400 * len(selected_joints),
        width=1400,
        showlegend=True,
        legend=dict(x=0.85, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified'
    )

    fig.show()

    return 0


if __name__ == "__main__":
    exit(main())
