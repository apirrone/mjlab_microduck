#!/usr/bin/env python3
"""Debug script to test robot translations and display raw accelerometer data."""

import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

# Import the robot XML path
from src.mjlab_microduck.robot.microduck_constants import MICRODUCK_XML


def quat_rotate_inverse(quat, vec):
    """Rotate a vector by the inverse of a quaternion [w, x, y, z]."""
    w, x, y, z = quat
    vx, vy, vz = vec

    # Compute the rotation using quaternion conjugate (inverse for unit quaternions)
    t_w = -x * vx - y * vy - z * vz
    t_x = w * vx + y * vz - z * vy
    t_y = w * vy + z * vx - x * vz
    t_z = w * vz + x * vy - y * vx

    # Second: (q* * v) * q
    result_x = t_w * (-x) + t_x * w + t_y * (-z) - t_z * (-y)
    result_y = t_w * (-y) + t_y * w + t_z * (-x) - t_x * (-z)
    result_z = t_w * (-z) + t_z * w + t_x * (-y) - t_y * (-x)

    return np.array([result_x, result_y, result_z])


def get_body_frame_acceleration(model, data, body_name, prev_vel, dt):
    """Calculate linear acceleration in the body frame.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_name: Name of the body
        prev_vel: Previous world-frame velocity
        dt: Time step

    Returns:
        3D acceleration vector in body frame (simulating accelerometer reading)
    """
    # Get body ID
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    # Get current world-frame velocity
    current_vel = data.cvel[body_id, 3:6].copy()  # Linear velocity (last 3 components)

    # Calculate world-frame acceleration
    world_accel = (current_vel - prev_vel) / dt

    # Get body orientation (quaternion)
    quat = data.xquat[body_id]  # [w, x, y, z]

    # Rotate acceleration into body frame
    body_accel = quat_rotate_inverse(quat, world_accel)

    # Add gravity component (accelerometer measures specific force)
    # Gravity in world frame is [0, 0, -9.81]
    world_gravity = np.array([0.0, 0.0, -9.81])
    body_gravity = quat_rotate_inverse(quat, world_gravity)

    # Accelerometer reads specific force = acceleration - gravity
    # Or in other words: accelerometer reading = measured acceleration + body_frame_gravity
    accelerometer_reading = body_accel - body_gravity

    return accelerometer_reading, current_vel


def apply_velocity_impulse(data, joint_adr, direction, magnitude):
    """Apply a velocity impulse in a specific direction.

    Args:
        data: MuJoCo data
        joint_adr: Joint address in qvel
        direction: 'forward', 'backward', 'left', 'right', 'up', 'down'
        magnitude: Velocity magnitude in m/s
    """
    # Reset all velocities to zero
    data.qvel[joint_adr:joint_adr + 6] = 0.0

    # Apply velocity in the specified direction (world frame)
    if direction == 'forward':
        data.qvel[joint_adr + 0] = magnitude  # +X
    elif direction == 'backward':
        data.qvel[joint_adr + 0] = -magnitude  # -X
    elif direction == 'left':
        data.qvel[joint_adr + 1] = magnitude  # +Y
    elif direction == 'right':
        data.qvel[joint_adr + 1] = -magnitude  # -Y
    elif direction == 'up':
        data.qvel[joint_adr + 2] = magnitude  # +Z
    elif direction == 'down':
        data.qvel[joint_adr + 2] = -magnitude  # -Z


def main():
    # Load the model
    print(f"Loading model from: {MICRODUCK_XML}")
    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_XML))
    data = mujoco.MjData(model)

    # Find the freejoint index
    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    if freejoint_id < 0:
        print("Error: Could not find trunk_base_freejoint")
        return

    # Get the qpos and qvel addresses for the freejoint
    qpos_adr = model.jnt_qposadr[freejoint_id]
    qvel_adr = model.jnt_dofadr[freejoint_id]

    print("\n" + "="*80)
    print("MicroDuck Acceleration Debug Script")
    print("="*80)
    print("Launching MuJoCo viewer...")

    # Define translation sequences
    # Each entry is (name, direction, description)
    translations = [
        ("Forward (+X)", "forward", "Accelerating forward along X-axis"),
        ("Backward (-X)", "backward", "Accelerating backward along X-axis"),
        ("Left (+Y)", "left", "Accelerating left along Y-axis"),
        ("Right (-Y)", "right", "Accelerating right along Y-axis"),
        ("Up (+Z)", "up", "Accelerating upward along Z-axis"),
        ("Down (-Z)", "down", "Accelerating downward along Z-axis"),
    ]

    steps_per_translation = 30
    max_velocity = 2.0  # m/s
    dt = model.opt.timestep
    sleep_time = 0.05  # seconds between steps

    # Launch the passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Initial sync
        viewer.sync()
        time.sleep(1.0)  # Give time to see the initial state

        # Initialize robot at a fixed position and orientation
        data.qpos[qpos_adr + 0] = 0.0  # x
        data.qpos[qpos_adr + 1] = 0.0  # y
        data.qpos[qpos_adr + 2] = 0.3  # z (height)
        data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]  # identity quaternion

        # Get body ID for trunk_base
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

        for translation_name, direction, description in translations:
            print(f"\n{translation_name}: {description}")
            print("-" * 80)

            # Reset position and velocity
            data.qpos[qpos_adr + 0] = 0.0
            data.qpos[qpos_adr + 1] = 0.0
            data.qpos[qpos_adr + 2] = 0.3
            data.qvel[qvel_adr:qvel_adr + 6] = 0.0

            # Forward kinematics to initialize
            mujoco.mj_forward(model, data)

            # Initialize previous velocity
            prev_vel = data.cvel[body_id, 3:6].copy()

            # Acceleration phase: ramp up velocity
            print("\n  ACCELERATION PHASE:")
            for i in range(steps_per_translation):
                # Calculate current velocity (smooth ramp up)
                t = i / steps_per_translation
                current_magnitude = max_velocity * t

                # Apply velocity
                apply_velocity_impulse(data, qvel_adr, direction, current_magnitude)

                # Step simulation
                mujoco.mj_step(model, data)

                # Calculate accelerometer reading
                accel, prev_vel = get_body_frame_acceleration(
                    model, data, "trunk_base", prev_vel, dt
                )

                # Get position for display
                pos = data.qpos[qpos_adr:qpos_adr + 3]

                # Sync with viewer
                viewer.sync()

                # Print accelerometer data
                if i % 3 == 0:  # Print every 3 steps to reduce clutter
                    print(f"  Step {i:2d}: Pos=[{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] | "
                          f"Accel=[{accel[0]:7.3f}, {accel[1]:7.3f}, {accel[2]:7.3f}] m/s²")

                time.sleep(sleep_time)

            # Constant velocity phase
            print("\n  CONSTANT VELOCITY PHASE:")
            for i in range(10):
                # Maintain constant velocity
                apply_velocity_impulse(data, qvel_adr, direction, max_velocity)

                # Step simulation
                mujoco.mj_step(model, data)

                # Calculate accelerometer reading
                accel, prev_vel = get_body_frame_acceleration(
                    model, data, "trunk_base", prev_vel, dt
                )

                # Get position for display
                pos = data.qpos[qpos_adr:qpos_adr + 3]

                # Sync with viewer
                viewer.sync()

                # Print accelerometer data
                if i % 2 == 0:
                    print(f"  Step {i:2d}: Pos=[{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] | "
                          f"Accel=[{accel[0]:7.3f}, {accel[1]:7.3f}, {accel[2]:7.3f}] m/s²")

                time.sleep(sleep_time)

            # Deceleration phase: ramp down velocity
            print("\n  DECELERATION PHASE:")
            for i in range(steps_per_translation):
                # Calculate current velocity (smooth ramp down)
                t = i / steps_per_translation
                current_magnitude = max_velocity * (1 - t)

                # Apply velocity
                apply_velocity_impulse(data, qvel_adr, direction, current_magnitude)

                # Step simulation
                mujoco.mj_step(model, data)

                # Calculate accelerometer reading
                accel, prev_vel = get_body_frame_acceleration(
                    model, data, "trunk_base", prev_vel, dt
                )

                # Get position for display
                pos = data.qpos[qpos_adr:qpos_adr + 3]

                # Sync with viewer
                viewer.sync()

                # Print accelerometer data
                if i % 3 == 0:
                    print(f"  Step {i:2d}: Pos=[{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] | "
                          f"Accel=[{accel[0]:7.3f}, {accel[1]:7.3f}, {accel[2]:7.3f}] m/s²")

                time.sleep(sleep_time)

            # Stop and reset
            print(f"\n  Stopping and resetting...")
            data.qvel[qvel_adr:qvel_adr + 6] = 0.0
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(sleep_time * 4)

        print("\n" + "="*80)
        print("Debug complete! Close the viewer window to exit.")
        print("="*80)

        # Keep viewer open for a bit longer
        time.sleep(3.0)


if __name__ == "__main__":
    main()
