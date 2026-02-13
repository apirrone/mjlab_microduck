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


def extract_actions_from_observations(observations: np.ndarray, imitation: bool = False) -> np.ndarray:
    """
    Extract actions from observation vectors.

    Standard observation structure (51D):
    [0:3]    - Base angular velocity
    [3:6]    - Projected gravity
    [6:20]   - Joint positions relative (14D)
    [20:34]  - Joint velocities (14D)
    [34:48]  - Last action (14D) <- THIS IS WHAT WE WANT
    [48:51]  - Velocity command (3D)

    Imitation observation structure (53D):
    [0:3]    - Velocity command (3D)
    [3:5]    - Phase (2D) - [cos(2π*phase), sin(2π*phase)]
    [5:8]    - Base angular velocity (3D)
    [8:11]   - Raw accelerometer (3D)
    [11:25]  - Joint positions relative (14D)
    [25:39]  - Joint velocities (14D)
    [39:53]  - Last action (14D) <- THIS IS WHAT WE WANT
    """
    if imitation:
        if observations.shape[1] < 53:
            raise ValueError(f"Imitation observation dimension too small: {observations.shape[1]}, expected 53")
        # Extract actions from indices 39:53 for imitation
        actions = observations[:, 39:53]
    else:
        if observations.shape[1] < 48:
            raise ValueError(f"Observation dimension too small: {observations.shape[1]}, expected at least 48")
        # Extract actions from indices 34:48 for standard
        actions = observations[:, 34:48]
    return actions


def construct_observation(data, last_action, imitation=False):
    """
    Construct observation from MuJoCo data state.

    Standard observation structure (51D):
    [0:3]    - Base angular velocity
    [3:6]    - Projected gravity
    [6:20]   - Joint positions relative (14D)
    [20:34]  - Joint velocities (14D)
    [34:48]  - Last action (14D)
    [48:51]  - Velocity command (3D) [zeros for replay]

    Imitation observation structure (53D):
    [0:3]    - Velocity command (3D) [zeros for replay]
    [3:5]    - Phase (2D) - [cos(2π*phase), sin(2π*phase)] [set to [1, 0] for phase=0]
    [5:8]    - Base angular velocity (3D)
    [8:11]   - Raw accelerometer (3D)
    [11:25]  - Joint positions relative (14D)
    [25:39]  - Joint velocities (14D)
    [39:53]  - Last action (14D)
    """
    if imitation:
        obs = np.zeros(53)

        # Velocity command (zeros for replay)
        obs[0:3] = 0.0

        # Phase [cos, sin] (set to [1, 0] for phase=0)
        obs[3:5] = [1.0, 0.0]

        # Base angular velocity (body frame)
        obs[5:8] = data.qvel[3:6]

        # Raw accelerometer (linear acceleration in body frame)
        # Approximate as projected gravity for stationary replay
        gravity = np.array([0, 0, -9.81])
        base_quat = data.qpos[3:7]  # [w, x, y, z]
        base_rot = np.zeros((3, 3))
        mujoco.mju_quat2Mat(base_rot.ravel(), base_quat)
        raw_accel = base_rot.T @ gravity
        obs[8:11] = raw_accel

        # Joint positions relative to default
        obs[11:25] = data.qpos[7:21] - DEFAULT_POSE

        # Joint velocities
        obs[25:39] = data.qvel[6:20]

        # Last action
        obs[39:53] = last_action

    else:
        obs = np.zeros(51)

        # Base angular velocity (body frame)
        obs[0:3] = data.qvel[3:6]

        # Projected gravity (rotate gravity vector to body frame)
        gravity = np.array([0, 0, -1])
        base_quat = data.qpos[3:7]  # [w, x, y, z]
        # Convert to rotation matrix and apply inverse rotation
        base_rot = np.zeros((3, 3))
        mujoco.mju_quat2Mat(base_rot.ravel(), base_quat)
        projected_gravity = base_rot.T @ gravity
        obs[3:6] = projected_gravity

        # Joint positions relative to default (qpos[7:21] are joint positions)
        obs[6:20] = data.qpos[7:21] - DEFAULT_POSE

        # Joint velocities (qvel[6:20] are joint velocities)
        obs[20:34] = data.qvel[6:20]

        # Last action
        obs[34:48] = last_action

        # Velocity command (zeros for replay)
        obs[48:51] = 0.0

    return obs


def initialize_robot(model, data, hang=False):
    """Initialize robot to default standing pose."""
    # Set joint positions to default pose (qpos[7:21] are the joint positions)
    data.qpos[7:21] = DEFAULT_POSE

    # Set base position slightly above ground to avoid penetration
    if hang:
        data.qpos[2] = 0.30  # z-position of base (hanging in air)
    else:
        data.qpos[2] = 0.15  # z-position of base

    # Reset velocities
    data.qvel[:] = 0.0

    # Set initial control targets to default pose
    data.ctrl[:] = DEFAULT_POSE

    # Forward step to update physics
    mujoco.mj_forward(model, data)


def replay_observations_with_viewer(model, data, actions, timestamps, action_scale=1.0,
                                   real_time=True, hang=False, record_output=None, decimation=4, imitation=False):
    """
    Replay actions on the robot with viewer.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        actions: Array of actions to replay (N x 14)
        timestamps: Array of timestamps for each action (N,)
        action_scale: Scaling factor for actions (default: 1.0)
        real_time: If True, replay at actual timing. If False, step as fast as possible.
        hang: If True, robot hangs in the air. If False, robot stands on ground.
        record_output: If provided, save recorded observations to this file path.
        imitation: If True, use imitation observation structure (53D instead of 51D).
    """
    print(f"Replaying {len(actions)} actions...")
    print(f"Duration: {timestamps[-1]:.2f}s")
    print(f"Action scale: {action_scale}")
    if hang:
        print("Mode: Hanging in air")
    if record_output:
        print(f"Recording observations to: {record_output}")

    # Initialize robot
    initialize_robot(model, data, hang=hang)

    # Save initial base position and orientation for hang mode
    if hang:
        base_pos = data.qpos[0:3].copy()
        base_quat = data.qpos[3:7].copy()

    # Lists to record observations
    recorded_observations = []
    recorded_timestamps = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        # Wait 1 second in default pose
        print("Waiting 1 second in default pose...")
        wait_steps = int(1.0 / model.opt.timestep)
        for _ in range(wait_steps):
            if hang:
                # Fix base position and orientation
                data.qpos[0:3] = base_pos
                data.qpos[3:7] = base_quat
                data.qvel[0:6] = 0.0
            mujoco.mj_step(model, data)
            viewer.sync()
            if not viewer.is_running():
                return recorded_observations, recorded_timestamps

        if real_time:
            time.sleep(1.0)

        print("Starting action replay...")
        start_time = time.time()
        control_dt = decimation * model.opt.timestep

        # Record all observations by replaying the action sequence
        # Key insight: obs[i] is recorded WITH last_action[i], then action[i+1] is applied
        for i in range(len(actions)):
            step_start = time.time()

            # Fix base position if hanging
            if hang:
                data.qpos[0:3] = base_pos
                data.qpos[3:7] = base_quat
                data.qvel[0:6] = 0.0

            # Record observation at step i with last_action = actions[i]
            if record_output:
                obs = construct_observation(data, actions[i], imitation=imitation)
                recorded_observations.append(obs)
                recorded_timestamps.append(timestamps[i])

            # Apply action for next step (action[i+1]) and simulate
            if i < len(actions) - 1:
                action = actions[i + 1]  # Next action to apply
                motor_targets = DEFAULT_POSE + action * action_scale
                data.ctrl[:] = motor_targets

                # Step simulation 'decimation' times (matches mjlab env.step and infer_policy.py)
                for _ in range(decimation):
                    if hang:
                        data.qpos[0:3] = base_pos
                        data.qpos[3:7] = base_quat
                        data.qvel[0:6] = 0.0

                    mujoco.mj_step(model, data)
                    viewer.sync()

                if not viewer.is_running():
                    break

                # Real-time pacing (sleep once per control step)
                if real_time:
                    elapsed = time.time() - step_start
                    sleep_time = control_dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            # Progress update
            if i % 50 == 0 and i > 0:
                print(f"Progress: {i}/{len(actions)} observations ({timestamps[i]:.2f}s)")

        print("Replay complete!")

        # Only keep viewer open if not recording (for visual inspection)
        if not record_output:
            time.sleep(2.0)  # Keep viewer open for 2 seconds after completion

    return recorded_observations, recorded_timestamps


def replay_observations_no_viewer(model, data, actions, timestamps, action_scale=1.0,
                                 real_time=True, hang=False, record_output=None, decimation=4, imitation=False):
    """
    Replay actions on the robot without viewer (headless mode).

    Args:
        model: MuJoCo model
        data: MuJoCo data
        actions: Array of actions to replay (N x 14)
        timestamps: Array of timestamps for each action (N,)
        action_scale: Scaling factor for actions (default: 1.0)
        real_time: If True, replay at actual timing. If False, step as fast as possible.
        hang: If True, robot hangs in the air. If False, robot stands on ground.
        record_output: If provided, save recorded observations to this file path.
        imitation: If True, use imitation observation structure (53D instead of 51D).
    """
    print(f"Replaying {len(actions)} actions (no viewer)...")
    print(f"Duration: {timestamps[-1]:.2f}s")
    print(f"Action scale: {action_scale}")
    if hang:
        print("Mode: Hanging in air")
    if record_output:
        print(f"Recording observations to: {record_output}")

    # Initialize robot
    initialize_robot(model, data, hang=hang)

    # Save initial base position and orientation for hang mode
    if hang:
        base_pos = data.qpos[0:3].copy()
        base_quat = data.qpos[3:7].copy()

    # Lists to record observations
    recorded_observations = []
    recorded_timestamps = []

    # Wait 1 second in default pose
    print("Waiting 1 second in default pose...")
    wait_steps = int(1.0 / model.opt.timestep)
    for _ in range(wait_steps):
        if hang:
            # Fix base position and orientation
            data.qpos[0:3] = base_pos
            data.qpos[3:7] = base_quat
            data.qvel[0:6] = 0.0
        mujoco.mj_step(model, data)

    if real_time:
        time.sleep(1.0)

    print("Starting action replay...")
    start_time = time.time()
    control_dt = decimation * model.opt.timestep

    # Record all observations by replaying the action sequence
    # Key insight: obs[i] is recorded WITH last_action[i], then action[i+1] is applied
    for i in range(len(actions)):
        step_start = time.time()

        # Fix base position if hanging
        if hang:
            data.qpos[0:3] = base_pos
            data.qpos[3:7] = base_quat
            data.qvel[0:6] = 0.0

        # Record observation at step i with last_action = actions[i]
        if record_output:
            obs = construct_observation(data, actions[i], imitation=imitation)
            recorded_observations.append(obs)
            recorded_timestamps.append(timestamps[i])

        # Apply action for next step (action[i+1]) and simulate
        if i < len(actions) - 1:
            action = actions[i + 1]  # Next action to apply
            motor_targets = DEFAULT_POSE + action * action_scale
            data.ctrl[:] = motor_targets

            # Step simulation 'decimation' times (matches mjlab env.step and infer_policy.py)
            for _ in range(decimation):
                if hang:
                    data.qpos[0:3] = base_pos
                    data.qpos[3:7] = base_quat
                    data.qvel[0:6] = 0.0

                mujoco.mj_step(model, data)

            # Real-time pacing (sleep once per control step)
            if real_time:
                elapsed = time.time() - step_start
                sleep_time = control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        # Progress update
        if i % 50 == 0 and i > 0:
            print(f"Progress: {i}/{len(actions)} observations ({timestamps[i]:.2f}s)")

    print("Replay complete!")

    return recorded_observations, recorded_timestamps


def main():
    parser = argparse.ArgumentParser(description="Replay observations on MicroDuck robot")
    parser.add_argument("pkl_file", type=str, help="Path to .pkl file with observations")
    parser.add_argument("--model", type=str,
                       default="src/mjlab_microduck/robot/microduck/scene.xml",
                       help="Path to MuJoCo XML model file")
    parser.add_argument("--action-scale", type=float, default=1.0,
                       help="Scaling factor for actions (default: 0.3)")
    parser.add_argument("--no-viewer", action="store_true",
                       help="Run without visualization")
    parser.add_argument("--no-real-time", action="store_true",
                       help="Run as fast as possible without real-time pacing")
    parser.add_argument("--hang", action="store_true",
                       help="Robot hangs in the air instead of standing on ground")
    parser.add_argument("--record-output", type=str,
                       help="Path to save recorded observations from simulation (e.g., sim_observations.pkl)")
    parser.add_argument("--imitation", action="store_true",
                       help="Use imitation observation structure (53D: command[3], phase[2], base_ang_vel[3], raw_accel[3], joints[28], actions[14])")

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
    if args.imitation:
        print("Using imitation observation structure (53D)")
    actions = extract_actions_from_observations(observations, imitation=args.imitation)
    print(f"Extracted {len(actions)} actions (shape: {actions.shape})")

    # Load MuJoCo model
    print(f"Loading MuJoCo model from {args.model}...")
    model = mujoco.MjModel.from_xml_path(args.model)

    # Override timestep to match mjlab (0.005s instead of XML's 0.002s)
    # mjlab velocity environments use timestep=0.005 for performance/stability
    model.opt.timestep = 0.005

    # Decimation to match mjlab (control at 50Hz while simulation at 200Hz)
    decimation = 4
    control_dt = decimation * model.opt.timestep  # 0.02s per control step

    data = mujoco.MjData(model)

    print(f"Simulation timestep: {model.opt.timestep}s (200Hz)")
    print(f"Control frequency: 50Hz (decimation: {decimation})")
    print(f"Control dt: {control_dt}s")

    # Replay observations
    try:
        if args.no_viewer:
            recorded_obs, recorded_ts = replay_observations_no_viewer(
                model, data, actions, timestamps,
                action_scale=args.action_scale,
                real_time=not args.no_real_time,
                hang=args.hang,
                record_output=args.record_output,
                decimation=decimation,
                imitation=args.imitation
            )
        else:
            recorded_obs, recorded_ts = replay_observations_with_viewer(
                model, data, actions, timestamps,
                action_scale=args.action_scale,
                real_time=not args.no_real_time,
                hang=args.hang,
                record_output=args.record_output,
                decimation=decimation,
                imitation=args.imitation
            )

        # Save recorded observations if requested
        if args.record_output and recorded_obs:
            print(f"\nSaving {len(recorded_obs)} recorded observations to {args.record_output}...")
            save_data = {
                'observations': np.array(recorded_obs),
                'timestamps': np.array(recorded_ts)
            }
            with open(args.record_output, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"Saved successfully to {args.record_output}")
            print(f"Observation shape: {np.array(recorded_obs).shape}")
            print(f"Time range: {recorded_ts[0]:.3f}s to {recorded_ts[-1]:.3f}s")

    except KeyboardInterrupt:
        print("\nReplay interrupted by user")

    return 0


if __name__ == "__main__":
    exit(main())
