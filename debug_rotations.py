#!/usr/bin/env python3
"""Debug script to test robot rotations and display euler angles."""

import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

# Import the robot XML path
from src.mjlab_microduck.robot.microduck_constants import MICRODUCK_XML


def quat_from_euler(roll, pitch, yaw):
    """Convert euler angles (in radians) to quaternion [w, x, y, z]."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def euler_from_quat(quat):
    """Convert quaternion [w, x, y, z] to euler angles [roll, pitch, yaw] in radians."""
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def quat_rotate_inverse(quat, vec):
    """Rotate a vector by the inverse of a quaternion [w, x, y, z]."""
    w, x, y, z = quat
    vx, vy, vz = vec

    # Compute the rotation using quaternion conjugate (inverse for unit quaternions)
    # q* v q where q* is conjugate of q
    # This is equivalent to rotating by inverse quaternion

    # First: q* * v (treating v as pure quaternion [0, vx, vy, vz])
    t_w = -x * vx - y * vy - z * vz
    t_x = w * vx + y * vz - z * vy
    t_y = w * vy + z * vx - x * vz
    t_z = w * vz + x * vy - y * vx

    # Second: (q* * v) * q
    result_x = t_w * (-x) + t_x * w + t_y * (-z) - t_z * (-y)
    result_y = t_w * (-y) + t_y * w + t_z * (-x) - t_x * (-z)
    result_z = t_w * (-z) + t_z * w + t_x * (-y) - t_y * (-x)

    return np.array([result_x, result_y, result_z])


def get_projected_gravity(quat):
    """Get gravity vector projected into the body frame.

    Args:
        quat: Quaternion [w, x, y, z] representing body orientation

    Returns:
        3D vector representing gravity in body frame
    """
    # World gravity vector (pointing down in world frame)
    world_gravity = np.array([0.0, 0.0, -1.0])

    # Rotate gravity into body frame using inverse quaternion
    projected_gravity = quat_rotate_inverse(quat, world_gravity)

    return projected_gravity


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

    # Get the qpos address for the freejoint (first 7 values: x, y, z, qw, qx, qy, qz)
    qpos_adr = model.jnt_qposadr[freejoint_id]

    # Get the qvel address for the freejoint (first 6 values: vx, vy, vz, wx, wy, wz)
    qvel_adr = model.jnt_dofadr[freejoint_id]

    # Find the IMU angular velocity sensor
    imu_ang_vel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    if imu_ang_vel_id < 0:
        print("Error: Could not find imu_ang_vel sensor")
        return

    print("\n" + "="*60)
    print("MicroDuck Rotation Debug Script")
    print("="*60)
    print(f"IMU sensor ID: {imu_ang_vel_id}")
    print("Launching MuJoCo viewer...")

    # Define rotation sequences
    # Each entry is (name, max_angle, axis_name)
    rotations = [
        ("Pitch Forward (nose down)", 0.4, "pitch", 1),   # positive pitch - nose down, axis=1 (wy)
        ("Pitch Backward (nose up)", -0.4, "pitch", 1),   # negative pitch - nose up, axis=1 (wy)
        ("Roll Right", 0.4, "roll", 0),                   # positive roll - roll right, axis=0 (wx)
        ("Roll Left", -0.4, "roll", 0),                   # negative roll - roll left, axis=0 (wx)
        ("Yaw Left (CCW)", 0.4, "yaw", 2),                # positive yaw - turn left, axis=2 (wz)
        ("Yaw Right (CW)", -0.4, "yaw", 2),               # negative yaw - turn right, axis=2 (wz)
    ]

    steps_per_rotation = 20
    sleep_time = 0.1  # seconds between steps
    max_angular_velocity = 2.0  # rad/s

    # Launch the passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Initial sync
        viewer.sync()
        time.sleep(1.0)  # Give time to see the initial state

        for rotation_name, max_angle, axis, vel_axis in rotations:
            print(f"\n{rotation_name} (max angle: {np.degrees(max_angle):.1f}°)")
            print("-" * 60)

            for i in range(steps_per_rotation + 1):
                # Calculate current angle (smooth interpolation)
                t = i / steps_per_rotation
                # Use smooth acceleration/deceleration
                smooth_t = t * t * (3 - 2 * t)
                current_angle = max_angle * np.sin(smooth_t * np.pi)

                # Calculate angular velocity (derivative of angle)
                # For sine wave: d/dt[A*sin(ωt)] = A*ω*cos(ωt)
                # We're using sin(smooth_t * π), so derivative is based on smooth_t
                angular_velocity = max_angle * np.pi * np.cos(smooth_t * np.pi) * (6 * smooth_t * (1 - smooth_t)) / steps_per_rotation / sleep_time

                # Set base position at 0.3m height
                data.qpos[qpos_adr + 0] = 0.0  # x
                data.qpos[qpos_adr + 1] = 0.0  # y
                data.qpos[qpos_adr + 2] = 0.3  # z

                # Create rotation based on axis
                if axis == "roll":
                    quat = quat_from_euler(current_angle, 0.0, 0.0)
                elif axis == "pitch":
                    quat = quat_from_euler(0.0, current_angle, 0.0)
                elif axis == "yaw":
                    quat = quat_from_euler(0.0, 0.0, current_angle)

                # Set orientation (quaternion: w, x, y, z)
                data.qpos[qpos_adr + 3] = quat[0]  # qw
                data.qpos[qpos_adr + 4] = quat[1]  # qx
                data.qpos[qpos_adr + 5] = quat[2]  # qy
                data.qpos[qpos_adr + 6] = quat[3]  # qz

                # Set angular velocity in body frame
                # qvel for freejoint: [vx, vy, vz, wx, wy, wz]
                data.qvel[qvel_adr:qvel_adr + 6] = 0.0  # Reset all velocities
                data.qvel[qvel_adr + 3 + vel_axis] = angular_velocity  # Set angular velocity on correct axis

                # Forward kinematics
                mujoco.mj_forward(model, data)

                # Sync with viewer
                viewer.sync()

                # Get the resulting quaternion from the simulation
                result_quat = data.qpos[qpos_adr + 3:qpos_adr + 7]

                # Convert to euler angles
                euler = euler_from_quat(result_quat)
                roll_deg, pitch_deg, yaw_deg = np.degrees(euler)

                # Get projected gravity
                proj_grav = get_projected_gravity(result_quat)

                # Get angular velocity from IMU sensor (used as policy input)
                sensor_adr = model.sensor_adr[imu_ang_vel_id]
                ang_vel = data.sensordata[sensor_adr:sensor_adr + 3]

                # Print euler angles, projected gravity, and angular velocity
                print(f"Step {i:2d}/{steps_per_rotation}: "
                      f"Roll={roll_deg:7.2f}°  Pitch={pitch_deg:7.2f}°  Yaw={yaw_deg:7.2f}°  |  "
                      f"Proj Grav=[{proj_grav[0]:6.3f}, {proj_grav[1]:6.3f}, {proj_grav[2]:6.3f}]  |  "
                      f"Ang Vel=[{ang_vel[0]:7.3f}, {ang_vel[1]:7.3f}, {ang_vel[2]:7.3f}] rad/s")

                time.sleep(sleep_time)

            # Return to zero position
            print(f"Returning to zero...")
            data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]  # identity quaternion
            data.qpos[qpos_adr + 2] = 0.3  # Keep height at 0.3m
            data.qvel[qvel_adr:qvel_adr + 6] = 0.0  # Zero all velocities
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(sleep_time * 2)

        print("\n" + "="*60)
        print("Debug complete! Close the viewer window to exit.")
        print("="*60)

        # Keep viewer open for a bit longer
        time.sleep(3.0)


if __name__ == "__main__":
    main()
