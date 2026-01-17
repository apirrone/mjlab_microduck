#!/usr/bin/env python3
"""Simple script to run ONNX policy inference in MuJoCo with rendering."""

import argparse
import csv
import time
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort

MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene.xml"

# Default pose used by the policy (legs flexed, standing position)
# This is the reference pose that:
# - Actions are offsets from (motor_target = DEFAULT_POSE + action * scale)
# - Joint observations are relative to (obs_joint_pos = current_pos - DEFAULT_POSE)
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
], dtype=np.float32)


class PolicyInference:
    def __init__(self, model, data, onnx_path, action_scale=1.0, use_imitation=False, reference_motion_path=None,
                 delay_min_lag=0, delay_max_lag=0):
        self.model = model
        self.data = data
        self.action_scale = action_scale
        self.use_imitation = use_imitation
        self.delay_min_lag = delay_min_lag
        self.delay_max_lag = delay_max_lag

        # Load ONNX model
        print(f"Loading ONNX model from: {onnx_path}")
        self.ort_session = ort.InferenceSession(onnx_path)

        # Get input/output names
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        input_shape = self.ort_session.get_inputs()[0].shape
        output_shape = self.ort_session.get_outputs()[0].shape
        print(f"Policy input: {self.input_name}, shape: {input_shape}")
        print(f"Policy output: {self.output_name}, shape: {output_shape}")

        # Get sensor IDs and body IDs
        self.imu_ang_vel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.trunk_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

        print(f"Sensors found:")
        print(f"  imu_ang_vel: id={self.imu_ang_vel_id}")
        print(f"Body IDs:")
        print(f"  trunk_base: id={self.trunk_base_id}")

        # Joint information
        self.n_joints = model.nu  # Number of actuators

        # Default pose for the policy (flexed legs)
        self.default_pose = DEFAULT_POSE[:self.n_joints]
        print(f"Number of actuators: {self.n_joints}")
        print(f"Default pose: {self.default_pose}")
        print(f"Action scale: {self.action_scale}")

        # Last action (for observation history)
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)

        # Command (lin_vel_x, lin_vel_y, ang_vel_z)
        self.command = np.zeros(3, dtype=np.float32)

        # Imitation learning phase tracking
        self.imitation_phase = 0.0
        self.gait_period = 0.72  # Default period in seconds
        if self.use_imitation:
            print(f"\nImitation mode enabled")
            if reference_motion_path:
                # Load reference motion to get the actual period
                import pickle
                try:
                    with open(reference_motion_path, 'rb') as f:
                        ref_data = pickle.load(f)
                    # Get period from any motion (they should all have similar periods)
                    first_key = list(ref_data.keys())[0]
                    self.gait_period = ref_data[first_key]['period']
                    print(f"  Loaded reference motion from: {reference_motion_path}")
                    print(f"  Using gait period: {self.gait_period:.3f}s")
                except Exception as e:
                    print(f"  Warning: Could not load reference motion: {e}")
                    print(f"  Using default period: {self.gait_period:.3f}s")
            else:
                print(f"  Using default gait period: {self.gait_period:.3f}s")

        # Action delay buffer (matches mjlab's DelayedActuatorCfg)
        self.use_delay = self.delay_max_lag > 0
        if self.use_delay:
            buffer_size = self.delay_max_lag + 1
            self.action_buffer = [np.zeros(self.n_joints, dtype=np.float32) for _ in range(buffer_size)]
            self.buffer_index = 0
            # Sample a fixed lag for single-environment inference (matches mjlab behavior)
            self.current_lag = np.random.randint(self.delay_min_lag, self.delay_max_lag + 1)
            print(f"\nActuator delay enabled:")
            print(f"  Min lag: {self.delay_min_lag} timesteps")
            print(f"  Max lag: {self.delay_max_lag} timesteps")
            print(f"  Sampled lag: {self.current_lag} timesteps")
            print(f"  Buffer size: {buffer_size}")
        else:
            self.action_buffer = None
            self.current_lag = 0

    def quat_rotate_inverse(self, quat, vec):
        """Rotate a vector by the inverse of a quaternion [w, x, y, z].

        Uses the formula from PyTorch's quat_apply_inverse:
        result = vec - w * t + xyz × t, where t = xyz × vec * 2
        """
        w = quat[0]
        xyz = quat[1:4]  # [x, y, z]

        # t = xyz × vec * 2
        t = np.cross(xyz, vec) * 2

        # result = vec - w * t + xyz × t
        return vec - w * t + np.cross(xyz, t)

    def get_projected_gravity(self):
        """Get gravity vector projected into body frame using body quaternion.

        This matches mdp.projected_gravity which uses asset.data.projected_gravity_b,
        computed from root_link_quat_w (body frame), NOT from sensors.
        """
        # Get body orientation quaternion from xquat (w, x, y, z format in MuJoCo)
        quat = self.data.xquat[self.trunk_base_id].copy().astype(np.float32)

        # World gravity (pointing down)
        world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # Rotate into body frame
        return self.quat_rotate_inverse(quat, world_gravity)

    def get_base_ang_vel(self):
        """Get base angular velocity from IMU gyro sensor.

        This matches mdp.builtin_sensor with sensor_name="robot/imu_ang_vel".
        """
        sensor_adr = self.model.sensor_adr[self.imu_ang_vel_id]
        return self.data.sensordata[sensor_adr:sensor_adr + 3].copy().astype(np.float32)

    def get_joint_pos_relative(self):
        """Get joint positions relative to default pose.

        Returns: current_pos - default_pose
        This matches mdp.joint_pos_rel.
        """
        # Skip freejoint (7 qpos DOFs: x, y, z, qw, qx, qy, qz)
        current_pos = self.data.qpos[7:7 + self.n_joints].copy().astype(np.float32)
        return current_pos - self.default_pose

    def get_joint_vel(self):
        """Get joint velocities relative to default (which is zero).

        This matches mdp.joint_vel_rel.
        """
        # Skip freejoint (6 qvel DOFs: vx, vy, vz, wx, wy, wz)
        # Default joint vel is 0, so relative vel = absolute vel
        return self.data.qvel[6:6 + self.n_joints].copy().astype(np.float32)

    def get_imitation_phase_obs(self):
        """Get imitation phase observation [cos(phase * 2π), sin(phase * 2π)]."""
        angle = self.imitation_phase * 2 * np.pi
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

    def update_phase(self, dt):
        """Update the gait phase based on elapsed time."""
        if self.use_imitation:
            self.imitation_phase += dt / self.gait_period
            self.imitation_phase = self.imitation_phase % 1.0  # Keep in [0, 1]

    def get_observations(self):
        """Collect observations matching policy input.

        Order (from velocity_env_cfg.py after deleting base_lin_vel):
        1. base_ang_vel (3D)
        2. projected_gravity (3D)
        3. joint_pos (14D) - relative to default
        4. joint_vel (14D) - relative to default (zero)
        5. actions (14D) - last action
        6. command (3D) - velocity command
        7. imitation_phase (2D) - [cos, sin] (only if use_imitation=True)

        Total: 51D (no imitation) or 53D (with imitation)
        """
        obs = []

        # Base angular velocity from sensor (NO delay, NO noise) - 3D
        ang_vel = self.get_base_ang_vel()
        obs.append(ang_vel)

        # Projected gravity from body frame (NO delay, NO noise) - 3D
        proj_grav = self.get_projected_gravity()
        obs.append(proj_grav)

        # Joint positions (relative to default pose, NO noise) - n_joints
        joint_pos_rel = self.get_joint_pos_relative()
        obs.append(joint_pos_rel)

        # Joint velocities (relative to zero, NO noise) - n_joints
        joint_vel = self.get_joint_vel()
        obs.append(joint_vel)

        # Last action - n_joints
        obs.append(self.last_action)

        # Command (lin_vel_x, lin_vel_y, ang_vel_z) - 3D
        obs.append(self.command)

        # Imitation phase (only if enabled) - 2D
        if self.use_imitation:
            phase_obs = self.get_imitation_phase_obs()
            obs.append(phase_obs)

        # Concatenate all observations
        return np.concatenate(obs).astype(np.float32)

    def set_command(self, lin_vel_x=0.0, lin_vel_y=0.0, ang_vel_z=0.0):
        """Set velocity command."""
        self.command = np.array([lin_vel_x, lin_vel_y, ang_vel_z], dtype=np.float32)
        print(f"Command: lin_vel_x={lin_vel_x:.2f}, lin_vel_y={lin_vel_y:.2f}, ang_vel_z={ang_vel_z:.2f}")

    def infer(self):
        """Run policy inference and return action."""
        # Get observations
        obs = self.get_observations()

        # Add batch dimension
        obs_batch = obs.reshape(1, -1)

        # Run inference
        action = self.ort_session.run([self.output_name], {self.input_name: obs_batch})[0]

        # Remove batch dimension
        action = action.squeeze(0).astype(np.float32)

        # Store for next step
        self.last_action = action.copy()

        return action

    def apply_action(self, action):
        """Apply action to MuJoCo controls with optional delay.

        Motor targets = default_pose + action * action_scale

        If delay is enabled, the action is buffered and a delayed action
        from T-lag timesteps ago is applied instead (matching mjlab's DelayedActuatorCfg).
        """
        if self.use_delay:
            # Add current action to circular buffer
            self.action_buffer[self.buffer_index] = action.copy()

            # Calculate index for delayed action (T - lag timesteps ago)
            # Buffer stores most recent actions in order: [t-2, t-1, t]
            # If buffer_index=2 and lag=1, we want action at index 1 (t-1)
            # If buffer_index=2 and lag=2, we want action at index 0 (t-2)
            delayed_index = (self.buffer_index - self.current_lag) % len(self.action_buffer)
            delayed_action = self.action_buffer[delayed_index]

            # Advance buffer index (circular)
            self.buffer_index = (self.buffer_index + 1) % len(self.action_buffer)

            # Use delayed action
            target_positions = self.default_pose + delayed_action * self.action_scale
        else:
            # No delay: use current action directly
            target_positions = self.default_pose + action * self.action_scale

        # Set control targets
        self.data.ctrl[:] = target_positions


def main():
    parser = argparse.ArgumentParser(description="Run ONNX policy in MuJoCo")
    parser.add_argument("onnx_path", type=str, help="Path to ONNX policy file")
    parser.add_argument("--lin-vel-x", type=float, default=0.0, help="Linear velocity X command (m/s)")
    parser.add_argument("--lin-vel-y", type=float, default=0.0, help="Linear velocity Y command (m/s)")
    parser.add_argument("--ang-vel-z", type=float, default=0.0, help="Angular velocity Z command (rad/s)")
    parser.add_argument("--action-scale", type=float, default=1.0, help="Action scale (default: 1.0)")
    parser.add_argument("--imitation", action="store_true", help="Enable imitation mode (adds phase observation)")
    parser.add_argument("--reference-motion", type=str, default=None, help="Path to reference motion .pkl file (for imitation)")
    parser.add_argument("--delay", type=int, nargs='*', default=None, help="Enable actuator delay: --delay MIN MAX (e.g., --delay 1 2 for mjlab default) or --delay LAG for fixed delay")
    parser.add_argument("--debug", action="store_true", help="Print observations and actions")
    parser.add_argument("--save-csv", type=str, default=None, help="Save observations and actions to CSV file")
    args = parser.parse_args()

    # Parse delay arguments
    delay_min_lag = 0
    delay_max_lag = 0
    if args.delay is not None:
        if len(args.delay) == 0:
            # --delay with no arguments: use mjlab default (1-2 timesteps)
            delay_min_lag = 1
            delay_max_lag = 2
        elif len(args.delay) == 1:
            # --delay LAG: fixed delay
            delay_min_lag = args.delay[0]
            delay_max_lag = args.delay[0]
        elif len(args.delay) == 2:
            # --delay MIN MAX: random delay in range
            delay_min_lag = args.delay[0]
            delay_max_lag = args.delay[1]
        else:
            print("Error: --delay accepts 0, 1, or 2 arguments")
            return

    # Load MuJoCo model
    print(f"Loading MuJoCo model from: {MICRODUCK_XML}")
    model = mujoco.MjModel.from_xml_path(MICRODUCK_XML)
    data = mujoco.MjData(model)

    # Initialize policy
    policy = PolicyInference(
        model, data, args.onnx_path,
        action_scale=args.action_scale,
        use_imitation=args.imitation,
        reference_motion_path=args.reference_motion,
        delay_min_lag=delay_min_lag,
        delay_max_lag=delay_max_lag
    )
    policy.set_command(args.lin_vel_x, args.lin_vel_y, args.ang_vel_z)

    # Set initial position to default pose
    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = model.jnt_qposadr[freejoint_id]

    # Set base position (match training config: z=0.12-0.13)
    data.qpos[qpos_adr + 0] = 0.0  # x
    data.qpos[qpos_adr + 1] = 0.0  # y
    data.qpos[qpos_adr + 2] = 0.125  # z (height) - middle of training range
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]  # identity quaternion

    # Set joint positions to default pose (flexed legs)
    data.qpos[7:7 + policy.n_joints] = policy.default_pose

    # Set controls to default pose (important for initial state)
    data.ctrl[:] = policy.default_pose

    # Forward kinematics
    mujoco.mj_forward(model, data)

    # Verify observation size
    test_obs = policy.get_observations()
    expected_obs_size = 3 + 3 + policy.n_joints + policy.n_joints + policy.n_joints + 3
    if policy.use_imitation:
        expected_obs_size += 2  # Add phase observation

    if test_obs.size != expected_obs_size:
        print(f"\nWARNING: Observation size mismatch!")
        print(f"  Expected: {expected_obs_size}")
        print(f"  Got: {test_obs.size}")
        breakdown = f"3(ang_vel) + 3(proj_grav) + {policy.n_joints}(joint_pos) + {policy.n_joints}(joint_vel) + {policy.n_joints}(last_action) + 3(command)"
        if policy.use_imitation:
            breakdown += " + 2(imitation_phase)"
        print(f"  Breakdown: {breakdown}")
        print()

    print("\n" + "="*80)
    print("MicroDuck Policy Inference (NO delays, NO noise for sim2sim)")
    print("="*80)
    print(f"Control frequency: 50 Hz (decimation: 4)")
    print(f"Simulation timestep: {model.opt.timestep}s")
    print(f"Observation size: {test_obs.size} (expected: {expected_obs_size})")
    if policy.use_imitation:
        print(f"Imitation mode: ENABLED (gait period: {policy.gait_period:.3f}s)")
    print("Close viewer window to exit")
    print()

    # Control loop matching mjlab timing
    # Simulation runs at model.opt.timestep (0.005s = 200Hz)
    # Control runs every 4 steps (0.02s = 50Hz)
    decimation = 4
    control_step_count = 0
    control_dt = decimation * model.opt.timestep  # Time per control step (0.02s)

    # Data collection for CSV
    csv_data = [] if args.save_csv else None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        start_time = time.time()

        while viewer.is_running():
            step_start = time.time()

            # Update phase for imitation learning
            policy.update_phase(control_dt)

            # Control loop: run inference and apply action
            action = policy.infer()
            policy.apply_action(action)

            control_step_count += 1

            # Save data for CSV if requested
            if csv_data is not None:
                obs = policy.get_observations()

                # Create row: step, time, obs(51), action(14)
                row = {
                    'step': control_step_count,
                    'time': control_step_count * control_dt,
                }

                # Add observations
                for i in range(obs.size):
                    row[f'obs_{i}'] = obs[i]

                # Add actions
                for i in range(action.size):
                    row[f'action_{i}'] = action[i]

                csv_data.append(row)

            # Debug: print observations and actions
            if args.debug:
                # Print every step for first 10 steps, then every 50
                should_print = control_step_count <= 10 or control_step_count % 50 == 0

                if should_print:
                    obs = policy.get_observations()
                    pos = data.qpos[qpos_adr:qpos_adr + 3]
                    quat = data.qpos[qpos_adr + 3:qpos_adr + 7]

                    print(f"\n{'='*70}")
                    print(f"Step {control_step_count} DEBUG:")
                    print(f"{'='*70}")
                    if policy.use_imitation:
                        print(f"Imitation phase: {policy.imitation_phase:.4f} (period: {policy.gait_period:.3f}s)")
                    print(f"Base state:")
                    print(f"  Position: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
                    print(f"  Quaternion: [{quat[0]:7.4f}, {quat[1]:7.4f}, {quat[2]:7.4f}, {quat[3]:7.4f}]")
                    print(f"\nObservation (shape {obs.shape}, total {obs.size}):")
                    print(f"  Ang vel [0:3]:        {obs[0:3]}")
                    print(f"  Proj grav [3:6]:      {obs[3:6]}")
                    print(f"  Joint pos [6:20]:     {obs[6:6+policy.n_joints]}")
                    print(f"  Joint vel [20:34]:    {obs[6+policy.n_joints:6+2*policy.n_joints]}")
                    print(f"  Last action [34:48]:  {obs[6+2*policy.n_joints:6+3*policy.n_joints]}")
                    cmd_end = 6+3*policy.n_joints+3
                    print(f"  Command [48:{cmd_end}]:      {obs[6+3*policy.n_joints:cmd_end]}")
                    if policy.use_imitation:
                        print(f"  Phase [cos,sin]:      {obs[cmd_end:]}")
                    print(f"\nAction output:")
                    print(f"  Raw action: {action}")
                    print(f"  Action min/max: [{action.min():.4f}, {action.max():.4f}]")
                    if policy.use_delay:
                        print(f"  Delay: {policy.current_lag} timesteps (buffered)")
                    print(f"  Applied ctrl (first 5): {data.ctrl[:5]}")
                    print(f"  Applied ctrl (last 5):  {data.ctrl[-5:]}")

            # Step simulation 'decimation' times (matches mjlab env.step behavior)
            for _ in range(decimation):
                mujoco.mj_step(model, data)

            # Sync viewer
            viewer.sync()

            # Sleep to maintain real-time pacing
            elapsed = time.time() - step_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("\nInference stopped.")

    # Save CSV if requested
    if csv_data is not None and len(csv_data) > 0:
        print(f"\nSaving {len(csv_data)} steps to: {args.save_csv}")

        with open(args.save_csv, 'w', newline='') as csvfile:
            fieldnames = csv_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(csv_data)

        print(f"CSV file saved successfully!")
        print(f"  Columns: {len(fieldnames)}")
        print(f"  Rows: {len(csv_data)}")


if __name__ == "__main__":
    main()
