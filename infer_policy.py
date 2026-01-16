#!/usr/bin/env python3
"""Simple script to run ONNX policy inference in MuJoCo with rendering."""

import argparse
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
from pathlib import Path

# Import the robot XML path
from src.mjlab_microduck.robot.microduck_constants import MICRODUCK_XML


class PolicyInference:
    def __init__(self, model, data, onnx_path):
        self.model = model
        self.data = data

        # Load ONNX model
        print(f"Loading ONNX model from: {onnx_path}")
        self.ort_session = ort.InferenceSession(onnx_path)

        # Get input/output names
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        print(f"Input: {self.input_name}, shape: {self.ort_session.get_inputs()[0].shape}")
        print(f"Output: {self.output_name}, shape: {self.ort_session.get_outputs()[0].shape}")

        # Find freejoint and get dimensions
        self.freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.trunk_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

        # Get IMU sensor IDs
        try:
            self.imu_ang_vel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        except:
            self.imu_ang_vel_id = None
            print("Warning: imu_ang_vel sensor not found")

        # Joint information
        self.n_joints = model.nu  # Number of actuators

        # Last action (for history)
        self.last_action = np.zeros(self.n_joints)

        # Command (lin_vel_x, lin_vel_y, ang_vel_z)
        self.command = np.zeros(3)

        # Delay buffers for ang_vel and projected_gravity
        self.ang_vel_buffer = []
        self.proj_grav_buffer = []
        self.delay_steps = 2  # max_lag

    def quat_rotate_inverse(self, quat, vec):
        """Rotate a vector by the inverse of a quaternion [w, x, y, z]."""
        w, x, y, z = quat
        vx, vy, vz = vec

        # Compute rotation using quaternion conjugate
        t_w = -x * vx - y * vy - z * vz
        t_x = w * vx + y * vz - z * vy
        t_y = w * vy + z * vx - x * vz
        t_z = w * vz + x * vy - y * vx

        result_x = t_w * (-x) + t_x * w + t_y * (-z) - t_z * (-y)
        result_y = t_w * (-y) + t_y * w + t_z * (-x) - t_x * (-z)
        result_z = t_w * (-z) + t_z * w + t_x * (-y) - t_y * (-x)

        return np.array([result_x, result_y, result_z])

    def get_projected_gravity(self):
        """Get gravity vector projected into body frame."""
        # Get body orientation quaternion
        quat = self.data.xquat[self.trunk_base_id]  # [w, x, y, z]

        # World gravity (pointing down)
        world_gravity = np.array([0.0, 0.0, -1.0])

        # Rotate into body frame
        return self.quat_rotate_inverse(quat, world_gravity)

    def get_base_ang_vel(self):
        """Get base angular velocity from IMU sensor."""
        if self.imu_ang_vel_id is not None:
            sensor_adr = self.model.sensor_adr[self.imu_ang_vel_id]
            return self.data.sensordata[sensor_adr:sensor_adr + 3].copy()
        else:
            # Fallback: use body angular velocity
            return self.data.cvel[self.trunk_base_id, :3].copy()

    def get_delayed_observation(self, current_obs, buffer):
        """Get delayed observation using buffer."""
        buffer.append(current_obs.copy())

        # Keep buffer size limited
        if len(buffer) > self.delay_steps + 1:
            buffer.pop(0)

        # Return delayed observation (delay_min_lag=1, so we use buffer[-2] if available)
        if len(buffer) >= 2:
            return buffer[-2]
        else:
            return current_obs

    def get_observations(self):
        """Collect observations matching policy input.

        Order: base_ang_vel, projected_gravity, joint_pos, joint_vel, actions, command
        """
        obs = []

        # Base angular velocity (with delay) - 3D
        ang_vel = self.get_base_ang_vel()
        ang_vel_delayed = self.get_delayed_observation(ang_vel, self.ang_vel_buffer)
        obs.append(ang_vel_delayed)

        # Projected gravity (with delay) - 3D
        proj_grav = self.get_projected_gravity()
        proj_grav_delayed = self.get_delayed_observation(proj_grav, self.proj_grav_buffer)
        obs.append(proj_grav_delayed)

        # Joint positions (relative to default) - n_joints
        # For simplicity, we'll use absolute positions
        # In a real setup, you'd subtract the default pose
        joint_pos = self.data.qpos[7:7 + self.n_joints].copy()  # Skip freejoint (7 DOFs)
        obs.append(joint_pos)

        # Joint velocities - n_joints
        joint_vel = self.data.qvel[6:6 + self.n_joints].copy()  # Skip freejoint (6 DOFs)
        obs.append(joint_vel)

        # Last action - n_joints
        obs.append(self.last_action)

        # Command (lin_vel_x, lin_vel_y, ang_vel_z) - 3D
        obs.append(self.command)

        # Concatenate all observations
        return np.concatenate(obs).astype(np.float32)

    def set_command(self, lin_vel_x=0.0, lin_vel_y=0.0, ang_vel_z=0.0):
        """Set velocity command."""
        self.command = np.array([lin_vel_x, lin_vel_y, ang_vel_z])
        print(f"Command set to: lin_vel_x={lin_vel_x:.2f}, lin_vel_y={lin_vel_y:.2f}, ang_vel_z={ang_vel_z:.2f}")

    def infer(self):
        """Run policy inference and return action."""
        # Get observations
        obs = self.get_observations()

        # Add batch dimension
        obs_batch = obs.reshape(1, -1)

        # Run inference
        action = self.ort_session.run([self.output_name], {self.input_name: obs_batch})[0]

        # Remove batch dimension
        action = action.squeeze(0)

        # Store for next step
        self.last_action = action.copy()

        return action

    def apply_action(self, action):
        """Apply action to MuJoCo controls."""
        # Actions are typically normalized joint position targets
        # The scale and offset depend on training configuration
        self.data.ctrl[:] = action


def main():
    parser = argparse.ArgumentParser(description="Run ONNX policy in MuJoCo")
    parser.add_argument("onnx_path", type=str, help="Path to ONNX policy file")
    parser.add_argument("--lin-vel-x", type=float, default=0.0, help="Linear velocity X command")
    parser.add_argument("--lin-vel-y", type=float, default=0.0, help="Linear velocity Y command")
    parser.add_argument("--ang-vel-z", type=float, default=0.0, help="Angular velocity Z command")
    parser.add_argument("--timestep", type=float, default=0.02, help="Control timestep (seconds)")
    args = parser.parse_args()

    # Load MuJoCo model
    print(f"Loading MuJoCo model from: {MICRODUCK_XML}")
    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_XML))
    data = mujoco.MjData(model)

    # Initialize policy
    policy = PolicyInference(model, data, args.onnx_path)
    policy.set_command(args.lin_vel_x, args.lin_vel_y, args.ang_vel_z)

    # Set initial position
    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = model.jnt_qposadr[freejoint_id]
    data.qpos[qpos_adr + 0] = 0.0  # x
    data.qpos[qpos_adr + 1] = 0.0  # y
    data.qpos[qpos_adr + 2] = 0.15  # z (height)
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]  # identity quaternion

    # Forward kinematics
    mujoco.mj_forward(model, data)

    print("\n" + "="*80)
    print("Starting policy inference with rendering")
    print("="*80)
    print(f"Control timestep: {args.timestep}s")
    print("Close viewer window to exit")
    print()

    # Control loop
    step_count = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        while viewer.is_running():
            # Run inference
            action = policy.infer()

            # Apply action
            policy.apply_action(action)

            # Step simulation multiple times (decimation)
            # Assuming policy runs at 50Hz (0.02s) and sim at 200Hz (0.005s)
            n_steps = int(args.timestep / model.opt.timestep)
            for _ in range(n_steps):
                mujoco.mj_step(model, data)

            # Update viewer
            viewer.sync()

            # Print info periodically
            step_count += 1
            if step_count % 100 == 0:
                pos = data.qpos[qpos_adr:qpos_adr + 3]
                print(f"Step {step_count}: Position=[{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]")

    print("\nInference stopped.")


if __name__ == "__main__":
    main()
