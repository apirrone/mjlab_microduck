#!/usr/bin/env python3
"""
Get knee data from simulation (matching get_knee_data_real.py).
Robot hangs in the air and moves the left knee with a sine wave.
Records [timestamp, position, velocity, last_action] data.
"""

import time
import numpy as np
import pickle
import mujoco
import mujoco.viewer

# MuJoCo model path
MODEL_PATH = "src/mjlab_microduck/robot/microduck/scene.xml"

# Default pose for MicroDuck
DEFAULT_POSE = np.array([
    0.0,   # left_hip_yaw
    0.0,   # left_hip_roll
    0.6,   # left_hip_pitch
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

# Joint indices (0-indexed in control array)
LEFT_KNEE_IDX = 3
LEFT_HIP_PITCH_IDX = 2
LEFT_ANKLE_IDX = 4


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Record knee sine wave data from simulation")
    parser.add_argument("-f", "--frequency", type=float, default=1.0,
                       help="Sine wave frequency in Hz (default: 1.0)")
    parser.add_argument("-a", "--amplitude", type=float, default=1.0,
                       help="Sine wave amplitude in radians (default: 1.0)")
    parser.add_argument("-d", "--duration", type=float, default=3.0,
                       help="Recording duration in seconds (default: 3.0)")
    parser.add_argument("-o", "--output", type=str, default="knee_sin_sim.pkl",
                       help="Output pickle file (default: knee_sin_sim.pkl)")
    args = parser.parse_args()

    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)

    # Override timestep to match mjlab (0.005s for 200Hz simulation)
    model.opt.timestep = 0.005

    data = mujoco.MjData(model)

    print("Initializing robot...")
    # Set joint positions to default pose (qpos[7:21] are the joint positions)
    data.qpos[7:21] = DEFAULT_POSE

    # Set base position hanging in air
    data.qpos[2] = 0.30  # z-position of base

    # Reset velocities
    data.qvel[:] = 0.0

    # Set initial control targets to default pose
    data.ctrl[:] = DEFAULT_POSE

    # Forward step to update physics
    mujoco.mj_forward(model, data)

    # Save initial base position and orientation for hang mode
    base_pos = data.qpos[0:3].copy()
    base_quat = data.qpos[3:7].copy()

    print("Setting knee, hip_pitch, and ankle to 0...")
    # Set left knee, hip_pitch, and ankle to 0 (like in real script)
    data.ctrl[LEFT_KNEE_IDX] = 0.0
    data.ctrl[LEFT_HIP_PITCH_IDX] = 0.0
    data.ctrl[LEFT_ANKLE_IDX] = 0.0

    # Wait a bit for motors to settle
    for _ in range(int(0.5 / model.opt.timestep)):
        # Fix base position and orientation
        data.qpos[0:3] = base_pos
        data.qpos[3:7] = base_quat
        data.qvel[0:6] = 0.0
        mujoco.mj_step(model, data)

    # Sine wave parameters from CLI
    amplitude = args.amplitude
    frequency = args.frequency
    duration = args.duration
    control_freq = 50  # Hz
    control_dt = 1.0 / control_freq

    data_list = []
    last_action = 0.0

    print(f"Starting data collection for {duration}s at {control_freq}Hz...")
    print(f"Sine wave: amplitude={amplitude} rad, frequency={frequency} Hz")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        t0 = time.time()

        while True:
            step_start = time.time()
            t = time.time() - t0

            if t > duration:
                break

            # Fix base position and orientation (hanging in air)
            data.qpos[0:3] = base_pos
            data.qpos[3:7] = base_quat
            data.qvel[0:6] = 0.0

            # Read current knee position and velocity
            # qpos[7:21] are joint positions, so knee is at qpos[7 + LEFT_KNEE_IDX]
            # qvel[6:20] are joint velocities, so knee is at qvel[6 + LEFT_KNEE_IDX]
            pos = data.qpos[7 + LEFT_KNEE_IDX]
            vel = data.qvel[6 + LEFT_KNEE_IDX]

            # Record [timestamp, position, velocity, last_action]
            data_list.append([t, pos, vel, last_action])

            # Compute new action (sine wave)
            action = amplitude * np.sin(2 * np.pi * frequency * t)

            # Apply action to knee
            data.ctrl[LEFT_KNEE_IDX] = action

            # Update last_action
            last_action = action

            # Step simulation (decimation=4 for 50Hz control with 200Hz sim)
            decimation = 4
            for _ in range(decimation):
                # Fix base position every simulation step
                data.qpos[0:3] = base_pos
                data.qpos[3:7] = base_quat
                data.qvel[0:6] = 0.0
                mujoco.mj_step(model, data)
                viewer.sync()

            if not viewer.is_running():
                print("Viewer closed by user")
                break

            # Real-time pacing
            elapsed = time.time() - step_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Progress update
            if len(data_list) % 50 == 0:
                print(f"Progress: {len(data_list)} samples ({t:.2f}s)")

    print(f"Data collection complete! Collected {len(data_list)} samples")

    # Calculate actual control frequency
    if len(data_list) > 1:
        actual_duration = data_list[-1][0] - data_list[0][0]
        actual_freq = (len(data_list) - 1) / actual_duration
        print(f"Actual duration: {actual_duration:.3f}s")
        print(f"Actual control frequency: {actual_freq:.2f} Hz (target: 50 Hz)")

    # Save data to pickle file
    print(f"Saving data to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data_list, f)

    print(f"Saved successfully!")
    print(f"Data format: [[timestamp, position, last_action], ...]")
    print(f"Number of samples: {len(data_list)}")


if __name__ == "__main__":
    main()
