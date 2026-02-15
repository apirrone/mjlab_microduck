#!/usr/bin/env python3
"""Simple script to run ONNX policy inference in MuJoCo with rendering."""

import argparse
import csv
import pickle
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
                 delay_min_lag=0, delay_max_lag=0, standing_onnx_path=None, switch_threshold=0.05,
                 use_projected_gravity=False):
        self.model = model
        self.data = data
        self.action_scale = action_scale
        self.use_imitation = use_imitation
        self.use_projected_gravity = use_projected_gravity
        self.delay_min_lag = delay_min_lag
        self.delay_max_lag = delay_max_lag
        self.switch_threshold = switch_threshold

        # Load walking policy (primary ONNX model)
        print(f"Loading walking policy from: {onnx_path}")
        self.walking_session = ort.InferenceSession(onnx_path)
        self.ort_session = self.walking_session  # Default to walking

        # Get input/output names from walking policy
        self.input_name = self.walking_session.get_inputs()[0].name
        self.output_name = self.walking_session.get_outputs()[0].name

        input_shape = self.walking_session.get_inputs()[0].shape
        output_shape = self.walking_session.get_outputs()[0].shape
        print(f"Walking policy input: {self.input_name}, shape: {input_shape}")
        print(f"Walking policy output: {self.output_name}, shape: {output_shape}")

        # Try to read gait period from ONNX metadata
        try:
            model_metadata = self.walking_session.get_modelmeta()
            if hasattr(model_metadata, 'custom_metadata_map') and 'gait_period' in model_metadata.custom_metadata_map:
                self.default_gait_period_from_onnx = float(model_metadata.custom_metadata_map['gait_period'])
                print(f"Found gait period in ONNX metadata: {self.default_gait_period_from_onnx:.4f}s")
            else:
                self.default_gait_period_from_onnx = None
        except Exception as e:
            print(f"Could not read gait period from ONNX metadata: {e}")
            self.default_gait_period_from_onnx = None

        # Load standing policy if provided
        self.standing_session = None
        if standing_onnx_path:
            print(f"\nLoading standing policy from: {standing_onnx_path}")
            self.standing_session = ort.InferenceSession(standing_onnx_path)
            standing_input_shape = self.standing_session.get_inputs()[0].shape
            standing_output_shape = self.standing_session.get_outputs()[0].shape
            print(f"Standing policy input: {self.standing_session.get_inputs()[0].name}, shape: {standing_input_shape}")
            print(f"Standing policy output: {self.standing_session.get_outputs()[0].name}, shape: {standing_output_shape}")
            print(f"Policy switching threshold: {switch_threshold} (command magnitude)")

        # Track which policy is active
        self.current_policy = "walking"

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
        self.gait_period = 0.5  # Fallback default

        if self.use_imitation:
            print(f"\nImitation mode enabled")

            # Priority 1: Load from reference motion file if provided
            if reference_motion_path:
                import pickle
                try:
                    with open(reference_motion_path, 'rb') as f:
                        ref_data = pickle.load(f)
                    # Get period from any motion (they should all have similar periods)
                    first_key = list(ref_data.keys())[0]
                    self.gait_period = ref_data[first_key]['period']
                    print(f"  Loaded gait period from reference motion file: {self.gait_period:.4f}s")
                    print(f"  Reference motion: {reference_motion_path}")
                except Exception as e:
                    print(f"  Warning: Could not load reference motion: {e}")
                    # Fall through to ONNX metadata

            # Priority 2: Use ONNX metadata if available and no reference motion
            if not reference_motion_path and self.default_gait_period_from_onnx is not None:
                self.gait_period = self.default_gait_period_from_onnx
                print(f"  Using gait period from ONNX metadata: {self.gait_period:.4f}s")

            # Priority 3: Fallback to default
            if reference_motion_path is None and self.default_gait_period_from_onnx is None:
                print(f"  Warning: No gait period found in ONNX or reference motion")
                print(f"  Using fallback default period: {self.gait_period:.4f}s")

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

    def get_raw_accelerometer(self):
        """Get raw accelerometer reading from MuJoCo sensor.

        Returns normalized raw accelerometer which includes gravity + linear acceleration.
        This matches what the real BNO055 accelerometer measures.
        Reads from the 'imu_accel' sensor in the MuJoCo model.
        """
        # Find the accelerometer sensor and read from sensordata
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        if sensor_id < 0:
            raise ValueError("Sensor 'imu_accel' not found in model")

        sensor_adr = self.model.sensor_adr[sensor_id]

        # Read accelerometer data (specific force measured by sensor)
        accel_raw = self.data.sensordata[sensor_adr:sensor_adr+3].copy().astype(np.float32)

        # MuJoCo accelerometer measures specific force (like real sensor)
        # Negate to match convention: when at rest upright, should point down
        accel_negated = -accel_raw

        # Normalize
        mag = np.linalg.norm(accel_negated)
        if mag > 0.1:
            return accel_negated / mag
        else:
            # Fallback to projected gravity
            quat = self.data.xquat[self.trunk_base_id].copy().astype(np.float32)
            world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            return self.quat_rotate_inverse(quat, world_gravity)

    def get_projected_gravity(self):
        """Get projected gravity in body frame.

        Returns the gravity vector projected into the robot's body frame,
        representing pure orientation without linear acceleration.
        """
        quat = self.data.xquat[self.trunk_base_id].copy().astype(np.float32)
        world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
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
        """Get imitation phase observation as [cos(2π*phase), sin(2π*phase)].

        This matches ImitationCommand.phase encoding.
        """
        phase_rad = self.imitation_phase * 2 * np.pi
        return np.array([np.cos(phase_rad), np.sin(phase_rad)], dtype=np.float32)

    def update_phase(self, dt):
        """Update the gait phase based on elapsed time."""
        if self.use_imitation:
            self.imitation_phase += dt / self.gait_period
            self.imitation_phase = self.imitation_phase % 1.0  # Keep in [0, 1]

    def get_observations(self):
        """Collect observations matching policy input.

        Order for velocity task (no imitation):
        1. base_ang_vel (3D)
        2. raw_accelerometer OR projected_gravity (3D)
        3. joint_pos (14D) - relative to default
        4. joint_vel (14D) - relative to default (zero)
        5. actions (14D) - last action
        6. command (3D) - velocity command
        Total: 51D

        Order for imitation task (use_imitation=True):
        1. command (3D) - velocity command
        2. phase (2D) - [cos(2π*phase), sin(2π*phase)]
        3. base_ang_vel (3D)
        4. raw_accelerometer OR projected_gravity (3D)
        5. joint_pos (14D) - relative to default
        6. joint_vel (14D) - relative to default (zero)
        7. actions (14D) - last action
        Total: 53D
        """
        obs = []

        if self.use_imitation:
            # Imitation task: command and phase come first
            # Command (lin_vel_x, lin_vel_y, ang_vel_z) - 3D
            obs.append(self.command)

            # Imitation phase [cos, sin] - 2D
            phase_obs = self.get_imitation_phase_obs()
            obs.append(phase_obs)

        # Base angular velocity from sensor (NO delay, NO noise) - 3D
        ang_vel = self.get_base_ang_vel()
        obs.append(ang_vel)

        # Gravity/accelerometer observation - 3D
        if self.use_projected_gravity:
            # Use projected gravity (orientation only)
            gravity = self.get_projected_gravity()
            obs.append(gravity)
        else:
            # Use raw accelerometer (includes linear acceleration)
            raw_accel = self.get_raw_accelerometer()
            obs.append(raw_accel)

        # Joint positions (relative to default pose, NO noise) - n_joints
        joint_pos_rel = self.get_joint_pos_relative()
        obs.append(joint_pos_rel)

        # Joint velocities (relative to zero, NO noise) - n_joints
        joint_vel = self.get_joint_vel()
        obs.append(joint_vel)

        # Last action - n_joints
        obs.append(self.last_action)

        if not self.use_imitation:
            # Velocity task: command comes last
            # Command (lin_vel_x, lin_vel_y, ang_vel_z) - 3D
            obs.append(self.command)

        # Concatenate all observations
        return np.concatenate(obs).astype(np.float32)

    def set_command(self, lin_vel_x=0.0, lin_vel_y=0.0, ang_vel_z=0.0):
        """Set velocity command and switch policy if needed."""
        self.command = np.array([lin_vel_x, lin_vel_y, ang_vel_z], dtype=np.float32)

        # Calculate command magnitude
        command_magnitude = np.sqrt(lin_vel_x**2 + lin_vel_y**2 + ang_vel_z**2)

        # Switch policy based on command magnitude if standing policy is available
        previous_policy = self.current_policy
        if self.standing_session is not None:
            if command_magnitude <= self.switch_threshold:
                self.current_policy = "standing"
                self.ort_session = self.standing_session
            else:
                self.current_policy = "walking"
                self.ort_session = self.walking_session

            # Print when policy switches
            if previous_policy != self.current_policy:
                print(f"Switched to {self.current_policy} policy (magnitude: {command_magnitude:.3f})")

        print(f"Command: lin_vel_x={lin_vel_x:.2f}, lin_vel_y={lin_vel_y:.2f}, ang_vel_z={ang_vel_z:.2f} [{self.current_policy}]")

    def infer(self):
        """Run policy inference and return action."""
        # Get observations
        obs = self.get_observations()

        # Add batch dimension
        obs_batch = obs.reshape(1, -1)

        # Run inference with active policy
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
            # target_positions = self.default_pose# + action * self.action_scale
            target_positions = self.default_pose + action * self.action_scale

        # Set control targets
        self.data.ctrl[:] = target_positions


def main():
    parser = argparse.ArgumentParser(description="Run ONNX policy in MuJoCo")
    parser.add_argument("onnx_path", type=str, help="Path to ONNX policy file (walking policy)")
    parser.add_argument("-s", "--standing", type=str, default=None, help="Path to standing policy ONNX file (optional)")
    parser.add_argument("--lin-vel-x", type=float, default=0.0, help="Linear velocity X command (m/s)")
    parser.add_argument("--lin-vel-y", type=float, default=0.0, help="Linear velocity Y command (m/s)")
    parser.add_argument("--ang-vel-z", type=float, default=0.0, help="Angular velocity Z command (rad/s)")
    parser.add_argument("--action-scale", type=float, default=1.0, help="Action scale (default: 1.0)")
    parser.add_argument("--imitation", action="store_true", help="Enable imitation mode (adds phase observation)")
    parser.add_argument("--reference-motion", type=str, default=None, help="Path to reference motion .pkl file (for imitation)")
    parser.add_argument("--raw-accelerometer", action="store_true", help="Use raw accelerometer instead of projected gravity (default: projected gravity)")
    parser.add_argument("--delay", type=int, nargs='*', default=None, help="Enable actuator delay: --delay MIN MAX (e.g., --delay 1 2 for mjlab default) or --delay LAG for fixed delay")
    parser.add_argument("--debug", action="store_true", help="Print observations and actions")
    parser.add_argument("--save-csv", type=str, default=None, help="Save observations and actions to CSV file")
    parser.add_argument("--record", type=str, default=None, help="Enable recording mode: save observations to pickle file on Ctrl+C")
    parser.add_argument("--switch-threshold", type=float, default=0.05, help="Command magnitude threshold for switching between standing and walking policy (default: 0.05)")
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

    # Override timestep to match mjlab (0.005s instead of XML's 0.002s)
    # mjlab velocity environments use timestep=0.005 for performance/stability
    model.opt.timestep = 0.005

    data = mujoco.MjData(model)

    # Initialize policy
    policy = PolicyInference(
        model, data, args.onnx_path,
        action_scale=args.action_scale,
        use_imitation=args.imitation,
        reference_motion_path=args.reference_motion,
        delay_min_lag=delay_min_lag,
        delay_max_lag=delay_max_lag,
        standing_onnx_path=args.standing,
        switch_threshold=args.switch_threshold,
        use_projected_gravity=not args.raw_accelerometer
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
    if policy.use_imitation:
        # Imitation task: command(3) + phase(2) + ang_vel(3) + proj_grav(3) + joint_pos + joint_vel + last_action
        expected_obs_size = 3 + 2 + 3 + 3 + policy.n_joints + policy.n_joints + policy.n_joints
        breakdown = f"3(command) + 2(phase) + 3(ang_vel) + 3(proj_grav) + {policy.n_joints}(joint_pos) + {policy.n_joints}(joint_vel) + {policy.n_joints}(last_action)"
    else:
        # Velocity task: ang_vel(3) + proj_grav(3) + joint_pos + joint_vel + last_action + command(3)
        expected_obs_size = 3 + 3 + policy.n_joints + policy.n_joints + policy.n_joints + 3
        breakdown = f"3(ang_vel) + 3(proj_grav) + {policy.n_joints}(joint_pos) + {policy.n_joints}(joint_vel) + {policy.n_joints}(last_action) + 3(command)"

    if test_obs.size != expected_obs_size:
        print(f"\nWARNING: Observation size mismatch!")
        print(f"  Expected: {expected_obs_size}")
        print(f"  Got: {test_obs.size}")
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
    if policy.standing_session is not None:
        print(f"Dual-policy mode: ENABLED")
        print(f"  Standing/Walking switch threshold: {policy.switch_threshold} (command magnitude)")
        print(f"  Initial policy: {policy.current_policy}")
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

    # Data collection for recording (pickle)
    recorded_observations = [] if args.record else None

    # Policy control for recording mode (start disabled, enable after 1s)
    policy_enabled = not args.record  # Disabled if recording, enabled otherwise
    policy_enable_time = None

    # Save original kp gains for restoring after standby
    original_kp = None
    if args.record:
        original_kp = model.actuator_gainprm[:, 0].copy()

    # Setup keyboard listener using pynput
    try:
        from pynput import keyboard as pynput_keyboard

        def on_press(key):
            try:
                # Handle special keys (arrows and space)
                if key == pynput_keyboard.Key.up:
                    policy.set_command(0.3, 0.0, 0.0)
                elif key == pynput_keyboard.Key.down:
                    policy.set_command(-0.2, 0.0, 0.0)
                elif key == pynput_keyboard.Key.right:
                    policy.set_command(0.0, 0.3, 0.0)
                elif key == pynput_keyboard.Key.left:
                    policy.set_command(0.0, -0.3, 0.0)
                elif key == pynput_keyboard.Key.space:
                    policy.set_command(0.0, 0.0, 0.0)
                # Handle character keys
                elif hasattr(key, 'char'):
                    if key.char == 'a' or key.char == 'A':
                        policy.set_command(0.0, 0.0, 2.0)
                    elif key.char == 'e' or key.char == 'E':
                        policy.set_command(0.0, 0.0, -2.0)
            except Exception as e:
                print(f"Key press error: {e}")

        listener = pynput_keyboard.Listener(on_press=on_press)
        listener.start()

        print("\nKeyboard controls enabled:")
        print("  UP arrow:    lin_vel_x = +0.5")
        print("  DOWN arrow:  lin_vel_x = -0.5")
        print("  RIGHT arrow: lin_vel_y = +0.5")
        print("  LEFT arrow:  lin_vel_y = -0.5")
        print("  A key:       ang_vel_z = -1.0")
        print("  E key:       ang_vel_z = +1.0")
        print("  SPACE:       Stop (all velocities = 0.0)")
        print("\nNote: Keyboard listener captures keys system-wide")

    except ImportError:
        print("\nKeyboard control unavailable:")
        print("  pynput not found. Install with: pip install pynput")
        print("  Or use command-line arguments: --lin-vel-x 0.5 --ang-vel-z 0.5")
    except Exception as e:
        print(f"\nCould not enable keyboard controls: {e}")
        import traceback
        traceback.print_exc()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        start_time = time.time()

        # Set policy enable time for recording mode
        if args.record:
            policy_enable_time = start_time + 1.0  # Enable after 1 second
            print("Recording mode: policy will be enabled after 1 second standby")

            # Set kp to 2.0 during standby to prevent falling (sim2real gap workaround)
            # Note: Real robot is stable at kp=0.28, but sim needs higher kp to hold position
            for i in range(model.nu):
                model.actuator_gainprm[i, 0] = 2.0  # kp
                model.actuator_biasprm[i, 1] = -2.0  # bias = -kp
            print("  Standby mode: kp set to 2.0 to prevent falling (sim needs this, real doesn't)")

        try:
            # Track previous step time for accurate phase updates
            prev_step_time = time.time()

            while viewer.is_running():
                step_start = time.time()

                # Enable policy after 1 second in recording mode
                if not policy_enabled and policy_enable_time is not None:
                    if step_start >= policy_enable_time:
                        policy_enabled = True

                        # Restore original kp gains (back to training kp=0.28)
                        if original_kp is not None:
                            for i in range(model.nu):
                                kp = original_kp[i]
                                model.actuator_gainprm[i, 0] = kp
                                model.actuator_biasprm[i, 1] = -kp
                            print("✓ Policy inference enabled (after 1s standby)")
                            print(f"  Restored original kp gains (range: [{original_kp.min():.2f}, {original_kp.max():.2f}])")

                # Calculate actual elapsed time since last step
                actual_dt = step_start - prev_step_time
                prev_step_time = step_start

                # Update phase for imitation learning using actual elapsed time
                # This ensures phase stays synchronized even if control loop runs faster/slower than target
                policy.update_phase(actual_dt)

                # Control loop: run inference and apply action (or hold default position during standby)
                if policy_enabled:
                    action = policy.infer()
                else:
                    # During standby, use zero actions (hold default position)
                    action = np.zeros(policy.n_joints, dtype=np.float32)
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

                # Record observations if requested
                if recorded_observations is not None:
                    obs = policy.get_observations()
                    timestamp = time.time() - start_time
                    recorded_observations.append({
                        'timestamp': timestamp,
                        'observation': obs.tolist()
                    })

                # Debug: print observations and actions
                if args.debug:
                    # Print every step for first 10 steps, then every 50
                    should_print = control_step_count <= 10 or control_step_count % 50 == 0

                    if should_print:
                        obs = policy.get_observations()
                        pos = data.qpos[qpos_adr:qpos_adr + 3]
                        quat = data.qpos[qpos_adr + 3:qpos_adr + 7]
                        # Use root link position (matches reward calculation)
                        com_height = pos[2]
    
                        print(f"\n{'='*70}")
                        print(f"Step {control_step_count} DEBUG:")
                        print(f"{'='*70}")
                        if policy.use_imitation:
                            print(f"Imitation phase: {policy.imitation_phase:.4f} (period: {policy.gait_period:.3f}s)")
                        print(f"Base state:")
                        print(f"  Position: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
                        print(f"  CoM height (root link): {com_height:7.4f}")
                        print(f"  Quaternion: [{quat[0]:7.4f}, {quat[1]:7.4f}, {quat[2]:7.4f}, {quat[3]:7.4f}]")
                        print(f"\nObservation (shape {obs.shape}, total {obs.size}):")
                        if policy.use_imitation:
                            # Imitation order: command, phase, ang_vel, proj_grav, joint_pos, joint_vel, last_action
                            print(f"  Command [0:3]:        {obs[0:3]}")
                            print(f"  Phase [3:5]:          {obs[3:5]} (cos, sin)")
                            print(f"  Ang vel [5:8]:        {obs[5:8]}")
                            print(f"  Proj grav [8:11]:     {obs[8:11]}")
                            joint_start = 11
                            print(f"  Joint pos [{joint_start}:{joint_start+policy.n_joints}]:     {obs[joint_start:joint_start+policy.n_joints]}")
                            print(f"  Joint vel [{joint_start+policy.n_joints}:{joint_start+2*policy.n_joints}]:    {obs[joint_start+policy.n_joints:joint_start+2*policy.n_joints]}")
                            print(f"  Last action [{joint_start+2*policy.n_joints}:{joint_start+3*policy.n_joints}]:  {obs[joint_start+2*policy.n_joints:joint_start+3*policy.n_joints]}")
                        else:
                            # Velocity order: ang_vel, proj_grav, joint_pos, joint_vel, last_action, command
                            print(f"  Ang vel [0:3]:        {obs[0:3]}")
                            print(f"  Proj grav [3:6]:      {obs[3:6]}")
                            print(f"  Joint pos [6:{6+policy.n_joints}]:     {obs[6:6+policy.n_joints]}")
                            print(f"  Joint vel [{6+policy.n_joints}:{6+2*policy.n_joints}]:    {obs[6+policy.n_joints:6+2*policy.n_joints]}")
                            print(f"  Last action [{6+2*policy.n_joints}:{6+3*policy.n_joints}]:  {obs[6+2*policy.n_joints:6+3*policy.n_joints]}")
                            cmd_end = 6+3*policy.n_joints+3
                            print(f"  Command [{6+3*policy.n_joints}:{cmd_end}]:      {obs[6+3*policy.n_joints:cmd_end]}")
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

        except KeyboardInterrupt:
            print("\n\nKeyboardInterrupt received (Ctrl+C). Saving data...")

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

    # Save recorded observations if requested
    if recorded_observations is not None and len(recorded_observations) > 0:
        print(f"\nSaving {len(recorded_observations)} recorded observations to: {args.record}")

        with open(args.record, 'wb') as f:
            pickle.dump(recorded_observations, f)

        print(f"✓ Recorded observations saved to {args.record}")
        print(f"  Observations: {len(recorded_observations)}")
        print(f"  Duration: {recorded_observations[-1]['timestamp']:.2f}s")


if __name__ == "__main__":
    main()
