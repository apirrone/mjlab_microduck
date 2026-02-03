#!/usr/bin/env python3
"""
Replay observation script for MicroDuck robot.
Loads timestamped observations from a .pkl file and replays the actions on the robot.
"""

import argparse
import pickle
import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path


# Default pose for MicroDuck (from microduck_constants.py)
DEFAULT_POSE = np.array([
    0.0,   # left_hip_yaw
    0.0,   # left_hip_roll
    0.6,   # left_hip_pitch (flexed)
    -1.2,  # left_knee
    0.6,   # left_ankle
    0.0,   # neck_pitch
    0.0,   # head_pitch
    0.0,   # head_yaw
    0.0,   # head_roll
    0.0,   # right_hip_yaw
    0.0,   # right_hip_roll
    -0.6,  # right_hip_pitch
    1.2,   # right_knee
    -0.6,  # right_ankle
])


def load_observations(pkl_path: str):
    """Load observations from pickle file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        # Check if it's a dictionary with 'observations' and 'timestamps'
        if 'observations' in data and 'timestamps' in data:
            observations = data['observations']
            timestamps = data['timestamps']
        else:
            raise ValueError("Dictionary must contain 'observations' and 'timestamps' keys")
    elif isinstance(data, list):
        if len(data) == 0:
            raise ValueError("Empty data list")

        # Check if list contains dictionaries with 'timestamp' and 'observation' keys
        if isinstance(data[0], dict) and 'timestamp' in data[0] and 'observation' in data[0]:
            timestamps = [item['timestamp'] for item in data]
            observations = [item['observation'] for item in data]
        # Check if it's a list of (timestamp, observation) tuples
        elif isinstance(data[0], tuple):
            timestamps = [item[0] for item in data]
            observations = [item[1] for item in data]
        else:
            # Just observations, create timestamps at 50Hz (0.02s intervals)
            observations = data
            timestamps = [i * 0.02 for i in range(len(observations))]
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")

    return np.array(observations), np.array(timestamps)


def extract_actions_from_observations(observations: np.ndarray) -> np.ndarray:
    """
    Extract actions from observation vectors.

    Observation structure (51D or 53D):
    [0:3]    - Base angular velocity
    [3:6]    - Projected gravity
    [6:20]   - Joint positions relative (14D)
    [20:34]  - Joint velocities (14D)
    [34:48]  - Last action (14D) <- THIS IS WHAT WE WANT
    [48:51]  - Velocity command (3D)
    [51:53]  - Imitation phase [optional] (2D)
    """
    if observations.shape[1] < 48:
        raise ValueError(f"Observation dimension too small: {observations.shape[1]}, expected at least 48")

    # Extract actions from indices 34:48
    actions = observations[:, 34:48]
    return actions


def initialize_robot(model, data):
    """Initialize robot to default standing pose."""
    # Set joint positions to default pose (qpos[7:21] are the joint positions)
    data.qpos[7:21] = DEFAULT_POSE

    # Set base position slightly above ground to avoid penetration
    data.qpos[2] = 0.15  # z-position of base

    # Reset velocities
    data.qvel[:] = 0.0

    # Set initial control targets to default pose
    data.ctrl[:] = DEFAULT_POSE

    # Forward step to update physics
    mujoco.mj_forward(model, data)


def replay_observations_with_viewer(model, data, actions, timestamps, action_scale=0.3,
                                   real_time=True):
    """
    Replay actions on the robot with viewer.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        actions: Array of actions to replay (N x 14)
        timestamps: Array of timestamps for each action (N,)
        action_scale: Scaling factor for actions (default: 0.3)
        real_time: If True, replay at actual timing. If False, step as fast as possible.
    """
    print(f"Replaying {len(actions)} actions...")
    print(f"Duration: {timestamps[-1]:.2f}s")
    print(f"Action scale: {action_scale}")

    # Initialize robot
    initialize_robot(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        # Wait 1 second in default pose
        print("Waiting 1 second in default pose...")
        wait_steps = int(1.0 / model.opt.timestep)
        for _ in range(wait_steps):
            mujoco.mj_step(model, data)
            viewer.sync()
            if not viewer.is_running():
                return

        if real_time:
            time.sleep(1.0)

        print("Starting action replay...")
        start_time = time.time()

        # Replay actions
        action_idx = 0
        sim_time = 0.0

        while viewer.is_running() and action_idx < len(actions):
            step_start = time.time()

            # Find current action based on simulation time
            while action_idx < len(actions) - 1 and timestamps[action_idx + 1] <= sim_time:
                action_idx += 1

            # Apply action to robot
            action = actions[action_idx]
            motor_targets = DEFAULT_POSE + action * action_scale
            data.ctrl[:] = motor_targets

            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            sim_time += model.opt.timestep

            # Real-time pacing
            if real_time:
                elapsed = time.time() - step_start
                sleep_time = model.opt.timestep - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Progress update
            if action_idx % 50 == 0 and action_idx > 0:
                print(f"Progress: {action_idx}/{len(actions)} actions ({sim_time:.2f}s)")

        print("Replay complete!")
        time.sleep(2.0)  # Keep viewer open for 2 seconds after completion


def replay_observations_no_viewer(model, data, actions, timestamps, action_scale=0.3,
                                 real_time=True):
    """
    Replay actions on the robot without viewer (headless mode).

    Args:
        model: MuJoCo model
        data: MuJoCo data
        actions: Array of actions to replay (N x 14)
        timestamps: Array of timestamps for each action (N,)
        action_scale: Scaling factor for actions (default: 0.3)
        real_time: If True, replay at actual timing. If False, step as fast as possible.
    """
    print(f"Replaying {len(actions)} actions (no viewer)...")
    print(f"Duration: {timestamps[-1]:.2f}s")
    print(f"Action scale: {action_scale}")

    # Initialize robot
    initialize_robot(model, data)

    # Wait 1 second in default pose
    print("Waiting 1 second in default pose...")
    wait_steps = int(1.0 / model.opt.timestep)
    for _ in range(wait_steps):
        mujoco.mj_step(model, data)

    if real_time:
        time.sleep(1.0)

    print("Starting action replay...")
    start_time = time.time()

    # Replay actions
    action_idx = 0
    sim_time = 0.0

    while action_idx < len(actions):
        step_start = time.time()

        # Find current action based on simulation time
        while action_idx < len(actions) - 1 and timestamps[action_idx + 1] <= sim_time:
            action_idx += 1

        # Apply action to robot
        action = actions[action_idx]
        motor_targets = DEFAULT_POSE + action * action_scale
        data.ctrl[:] = motor_targets

        # Step simulation
        mujoco.mj_step(model, data)
        sim_time += model.opt.timestep

        # Real-time pacing
        if real_time:
            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Progress update
        if action_idx % 50 == 0 and action_idx > 0:
            print(f"Progress: {action_idx}/{len(actions)} actions ({sim_time:.2f}s)")

    print("Replay complete!")


def main():
    parser = argparse.ArgumentParser(description="Replay observations on MicroDuck robot")
    parser.add_argument("pkl_file", type=str, help="Path to .pkl file with observations")
    parser.add_argument("--model", type=str,
                       default="src/mjlab_microduck/robot/microduck/scene.xml",
                       help="Path to MuJoCo XML model file")
    parser.add_argument("--action-scale", type=float, default=0.3,
                       help="Scaling factor for actions (default: 0.3)")
    parser.add_argument("--no-viewer", action="store_true",
                       help="Run without visualization")
    parser.add_argument("--no-real-time", action="store_true",
                       help="Run as fast as possible without real-time pacing")

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.pkl_file).exists():
        print(f"Error: {args.pkl_file} not found")
        return 1

    if not Path(args.model).exists():
        print(f"Error: Model file {args.model} not found")
        return 1

    # Load observations
    print(f"Loading observations from {args.pkl_file}...")
    observations, timestamps = load_observations(args.pkl_file)
    print(f"Loaded {len(observations)} observations")
    print(f"Observation shape: {observations.shape}")
    print(f"Time range: {timestamps[0]:.3f}s to {timestamps[-1]:.3f}s")

    # Extract actions
    print("Extracting actions from observations...")
    actions = extract_actions_from_observations(observations)
    print(f"Extracted {len(actions)} actions (shape: {actions.shape})")

    # Load MuJoCo model
    print(f"Loading MuJoCo model from {args.model}...")
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    # Replay observations
    try:
        if args.no_viewer:
            replay_observations_no_viewer(
                model, data, actions, timestamps,
                action_scale=args.action_scale,
                real_time=not args.no_real_time
            )
        else:
            replay_observations_with_viewer(
                model, data, actions, timestamps,
                action_scale=args.action_scale,
                real_time=not args.no_real_time
            )
    except KeyboardInterrupt:
        print("\nReplay interrupted by user")

    return 0


if __name__ == "__main__":
    exit(main())
