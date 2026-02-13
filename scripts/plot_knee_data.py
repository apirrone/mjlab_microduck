#!/usr/bin/env python3
"""
Plot knee data (position and action) using Plotly.
Can compare real vs simulated data or plot a single dataset.
"""

import argparse
import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def load_knee_data(pkl_path: str):
    """
    Load knee data from pickle file.

    Expected formats:
    - Newest: [[timestamp, position, velocity, last_action], ...]
    - New: [[timestamp, position, last_action], ...]
    - Old: [[position, last_action], ...]

    Returns:
        positions: np.array of positions
        velocities: np.array of velocities (or None if not present)
        actions: np.array of actions
        timestamps: np.array of timestamps
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data)}")

    if len(data) == 0:
        raise ValueError("Empty data list")

    # Check format based on first item length
    if len(data[0]) == 4:
        # Newest format: [timestamp, position, velocity, last_action]
        timestamps = np.array([item[0] for item in data])
        positions = np.array([item[1] for item in data])
        velocities = np.array([item[2] for item in data])
        actions = np.array([item[3] for item in data])
    elif len(data[0]) == 3:
        # New format: [timestamp, position, last_action]
        timestamps = np.array([item[0] for item in data])
        positions = np.array([item[1] for item in data])
        velocities = None
        actions = np.array([item[2] for item in data])
    elif len(data[0]) == 2:
        # Old format: [position, last_action]
        positions = np.array([item[0] for item in data])
        velocities = None
        actions = np.array([item[1] for item in data])
        # Generate timestamps at 50Hz (0.02s intervals)
        timestamps = np.arange(len(data)) * 0.02
    else:
        raise ValueError(f"Unexpected data format: each item has {len(data[0])} elements")

    return positions, velocities, actions, timestamps


def plot_single_dataset(positions, velocities, actions, timestamps, title="Knee Data"):
    """Plot a single dataset (position, velocity, and action)."""

    num_rows = 3 if velocities is not None else 2
    subplot_titles = ['Knee Position', 'Knee Velocity', 'Knee Action'] if velocities is not None else ['Knee Position', 'Knee Action']

    fig = make_subplots(
        rows=num_rows, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08 if num_rows == 3 else 0.12,
        row_heights=[1] * num_rows,
    )

    # Position plot
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=positions,
            name='Position',
            line=dict(color='blue', width=2),
            mode='lines'
        ),
        row=1, col=1
    )

    # Velocity plot (if available)
    if velocities is not None:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=velocities,
                name='Velocity',
                line=dict(color='red', width=2),
                mode='lines'
            ),
            row=2, col=1
        )

    # Action plot
    action_row = 3 if velocities is not None else 2
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=actions,
            name='Action',
            line=dict(color='green', width=2),
            mode='lines'
        ),
        row=action_row, col=1
    )

    # Update axes
    fig.update_xaxes(title_text='Time (s)', row=num_rows, col=1)
    fig.update_yaxes(title_text='Position (rad)', row=1, col=1)
    if velocities is not None:
        fig.update_yaxes(title_text='Velocity (rad/s)', row=2, col=1)
    fig.update_yaxes(title_text='Action (rad)', row=action_row, col=1)

    # Update layout
    fig.update_layout(
        title_text=title,
        title_font_size=20,
        height=800,
        width=1200,
        showlegend=True,
        hovermode='x unified'
    )

    fig.show()


def plot_comparison(real_pos, real_vel, real_act, real_ts, sim_pos, sim_vel, sim_act, sim_ts):
    """Plot comparison between real and simulated knee data."""

    has_velocity = real_vel is not None and sim_vel is not None
    num_rows = 3 if has_velocity else 2
    subplot_titles = ['Knee Position (Real vs Sim)', 'Knee Velocity (Real vs Sim)', 'Knee Action (Real vs Sim)'] if has_velocity else ['Knee Position (Real vs Sim)', 'Knee Action (Real vs Sim)']

    fig = make_subplots(
        rows=num_rows, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08 if num_rows == 3 else 0.12,
        row_heights=[1] * num_rows,
    )

    # Position comparison
    fig.add_trace(
        go.Scatter(
            x=real_ts,
            y=real_pos,
            name='Real Position',
            line=dict(color='blue', width=2),
            mode='lines'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=sim_ts,
            y=sim_pos,
            name='Sim Position',
            line=dict(color='red', width=2, dash='dash'),
            mode='lines'
        ),
        row=1, col=1
    )

    # Velocity comparison (if available)
    if has_velocity:
        fig.add_trace(
            go.Scatter(
                x=real_ts,
                y=real_vel,
                name='Real Velocity',
                line=dict(color='blue', width=2),
                mode='lines'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=sim_ts,
                y=sim_vel,
                name='Sim Velocity',
                line=dict(color='red', width=2, dash='dash'),
                mode='lines'
            ),
            row=2, col=1
        )

    # Action comparison
    action_row = 3 if has_velocity else 2
    fig.add_trace(
        go.Scatter(
            x=real_ts,
            y=real_act,
            name='Real Action',
            line=dict(color='darkblue', width=2),
            mode='lines'
        ),
        row=action_row, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=sim_ts,
            y=sim_act,
            name='Sim Action',
            line=dict(color='darkred', width=2, dash='dash'),
            mode='lines'
        ),
        row=action_row, col=1
    )

    # Compute common y-ranges
    pos_min = min(real_pos.min(), sim_pos.min())
    pos_max = max(real_pos.max(), sim_pos.max())
    pos_margin = (pos_max - pos_min) * 0.1
    pos_range = [pos_min - pos_margin, pos_max + pos_margin]

    if has_velocity:
        vel_min = min(real_vel.min(), sim_vel.min())
        vel_max = max(real_vel.max(), sim_vel.max())
        vel_margin = (vel_max - vel_min) * 0.1
        vel_range = [vel_min - vel_margin, vel_max + vel_margin]

    act_min = min(real_act.min(), sim_act.min())
    act_max = max(real_act.max(), sim_act.max())
    act_margin = (act_max - act_min) * 0.1
    act_range = [act_min - act_margin, act_max + act_margin]

    # Update axes with common ranges
    fig.update_xaxes(title_text='Time (s)', row=num_rows, col=1)
    fig.update_yaxes(title_text='Position (rad)', range=pos_range, row=1, col=1)
    if has_velocity:
        fig.update_yaxes(title_text='Velocity (rad/s)', range=vel_range, row=2, col=1)
    fig.update_yaxes(title_text='Action (rad)', range=act_range, row=action_row, col=1)

    # Update layout
    fig.update_layout(
        title_text='Knee Data: Real vs Simulated Comparison',
        title_font_size=20,
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified'
    )

    fig.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot knee data (position and action) using Plotly"
    )
    parser.add_argument("pkl_file", type=str,
                       help="Path to .pkl file with knee data")
    parser.add_argument("--compare", type=str, default=None,
                       help="Optional: Path to second .pkl file for comparison (e.g., sim data)")

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.pkl_file).exists():
        print(f"Error: {args.pkl_file} not found")
        return 1

    # Load primary dataset
    print(f"Loading data from {args.pkl_file}...")
    positions, velocities, actions, timestamps = load_knee_data(args.pkl_file)
    print(f"Loaded {len(positions)} samples")
    print(f"Duration: {timestamps[-1]:.2f}s")
    print(f"Position range: [{positions.min():.3f}, {positions.max():.3f}] rad")
    if velocities is not None:
        print(f"Velocity range: [{velocities.min():.3f}, {velocities.max():.3f}] rad/s")
    print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}] rad")

    if args.compare:
        # Comparison mode
        if not Path(args.compare).exists():
            print(f"Error: {args.compare} not found")
            return 1

        print(f"\nLoading comparison data from {args.compare}...")
        sim_positions, sim_velocities, sim_actions, sim_timestamps = load_knee_data(args.compare)
        print(f"Loaded {len(sim_positions)} samples")
        print(f"Duration: {sim_timestamps[-1]:.2f}s")
        print(f"Position range: [{sim_positions.min():.3f}, {sim_positions.max():.3f}] rad")
        if sim_velocities is not None:
            print(f"Velocity range: [{sim_velocities.min():.3f}, {sim_velocities.max():.3f}] rad/s")
        print(f"Action range: [{sim_actions.min():.3f}, {sim_actions.max():.3f}] rad")

        print("\nGenerating comparison plot...")
        plot_comparison(positions, velocities, actions, timestamps,
                       sim_positions, sim_velocities, sim_actions, sim_timestamps)
    else:
        # Single dataset mode
        print("\nGenerating plot...")
        dataset_name = Path(args.pkl_file).stem.replace('_', ' ').title()
        plot_single_dataset(positions, velocities, actions, timestamps,
                          title=f"Knee Data: {dataset_name}")

    return 0


if __name__ == "__main__":
    exit(main())
