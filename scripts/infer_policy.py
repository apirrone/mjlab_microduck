#!/usr/bin/env python3
"""Simple script to run ONNX policy inference in MuJoCo with rendering."""

import argparse
import csv
import math
import pickle
import time
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort

MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene.xml"
MICRODUCK_ROLLERS_XML = "src/mjlab_microduck/robot/microduck/robot_walk_rollers.xml"

# Body pose command constants (must match training constants)
BODY_CMD_MAX_Z = 0.03              # ±30 mm
BODY_CMD_MAX_ANGLE = math.radians(30)  # ±30°

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
    -0.5,   # neck_pitch
    0.5,   # head_pitch
    0.0,   # head_yaw
    0.0,   # head_roll
    0.0,   # right_hip_yaw
    0.0,   # right_hip_roll
    -0.6,  # right_hip_pitch
    1.2,   # right_knee
    -0.6,  # right_ankle
], dtype=np.float32)


class PolicyInference:
    def __init__(self, model, data, walking_onnx_path=None, action_scale=1.0, use_imitation=False,
                 reference_motion_path=None, delay_min_lag=0, delay_max_lag=0,
                 standing_onnx_path=None, switch_threshold=0.05,
                 use_projected_gravity=False, ground_pick_onnx_path=None, ground_pick_period=4.0):
        self.model = model
        self.data = data
        self.action_scale = action_scale
        self.use_imitation = use_imitation
        self.use_projected_gravity = use_projected_gravity
        self.delay_min_lag = delay_min_lag
        self.delay_max_lag = delay_max_lag
        self.switch_threshold = switch_threshold

        # Load walking policy
        self.walking_session = None
        self.default_gait_period_from_onnx = None
        if walking_onnx_path:
            print(f"Loading walking policy from: {walking_onnx_path}")
            self.walking_session = ort.InferenceSession(walking_onnx_path)
            w_input_shape = self.walking_session.get_inputs()[0].shape
            w_output_shape = self.walking_session.get_outputs()[0].shape
            print(f"Walking policy input: {self.walking_session.get_inputs()[0].name}, shape: {w_input_shape}")
            print(f"Walking policy output: {self.walking_session.get_outputs()[0].name}, shape: {w_output_shape}")

            # Try to read gait period from ONNX metadata
            try:
                model_metadata = self.walking_session.get_modelmeta()
                if hasattr(model_metadata, 'custom_metadata_map') and 'gait_period' in model_metadata.custom_metadata_map:
                    self.default_gait_period_from_onnx = float(model_metadata.custom_metadata_map['gait_period'])
                    print(f"Found gait period in ONNX metadata: {self.default_gait_period_from_onnx:.4f}s")
            except Exception as e:
                print(f"Could not read gait period from ONNX metadata: {e}")

        # Load standing policy
        self.standing_session = None
        if standing_onnx_path:
            print(f"\nLoading standing policy from: {standing_onnx_path}")
            self.standing_session = ort.InferenceSession(standing_onnx_path)
            s_input_shape = self.standing_session.get_inputs()[0].shape
            s_output_shape = self.standing_session.get_outputs()[0].shape
            print(f"Standing policy input: {self.standing_session.get_inputs()[0].name}, shape: {s_input_shape}")
            print(f"Standing policy output: {self.standing_session.get_outputs()[0].name}, shape: {s_output_shape}")
            if self.walking_session:
                print(f"Policy switching threshold: {switch_threshold} (vel command magnitude)")

        # Load ground pick policy
        self.ground_pick_session = None
        self.ground_pick_mode = False
        self.ground_pick_phase = 0.0
        self.ground_pick_period = ground_pick_period
        if ground_pick_onnx_path:
            print(f"\nLoading ground pick policy from: {ground_pick_onnx_path}")
            self.ground_pick_session = ort.InferenceSession(ground_pick_onnx_path)
            gp_input_shape = self.ground_pick_session.get_inputs()[0].shape
            print(f"Ground pick policy input shape: {gp_input_shape}")

        # Validate at least one policy loaded
        if not self.walking_session and not self.standing_session:
            raise ValueError("At least one of --walking or --standing must be provided")

        # Determine initial active session and policy
        if self.walking_session:
            self.current_policy = "walking"
            self.ort_session = self.walking_session
        else:
            self.current_policy = "standing"
            self.ort_session = self.standing_session

        # Get input/output names from active session
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        # Get sensor IDs and body IDs
        self.imu_ang_vel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.trunk_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

        print(f"Sensors found:")
        print(f"  imu_ang_vel: id={self.imu_ang_vel_id}")
        print(f"Body IDs:")
        print(f"  trunk_base: id={self.trunk_base_id}")

        # Joint information
        self.n_joints = model.nu

        # For robots with passive/interspersed joints (e.g. roller skates), the actuated
        # joints are not contiguous in qpos/qvel. Compute the correct indices from the
        # actuator transmission joint IDs so extraction works for any joint ordering.
        self.joint_qpos_indices = [
            int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)
        ]
        self.joint_qvel_indices = [
            int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)
        ]

        # Default pose for the policy (flexed legs)
        self.default_pose = DEFAULT_POSE[:self.n_joints]
        print(f"Number of actuators: {self.n_joints}")
        print(f"Default pose: {self.default_pose}")
        print(f"Action scale: {self.action_scale}")

        # Last action (for observation history)
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)

        # Velocity command [lin_vel_x, lin_vel_y, ang_vel_z] — controls walking / policy switching
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        # Body pose command [Δz (m), Δpitch (rad), Δroll (rad)] — physical units
        self.body_cmd = np.zeros(3, dtype=np.float32)
        # Normalized obs command (set by _update_command)
        self.command = np.zeros(3, dtype=np.float32)

        # Body pose mode (like head mode but for standing body pose control)
        self.body_pose_mode = False
        self.body_cmd_step_z = 0.001              # 1 mm per keypress
        self.body_cmd_step_angle = math.radians(1) # 1° per keypress

        # Head control mode
        self.head_mode = False
        self.head_offset = np.zeros(4, dtype=np.float32)
        self.head_max = 2.5
        self.head_step = 0.1

        # Imitation learning phase tracking
        self.imitation_phase = 0.0
        self.gait_period = 0.5

        if self.use_imitation:
            print(f"\nImitation mode enabled")

            if reference_motion_path:
                import pickle
                try:
                    with open(reference_motion_path, 'rb') as f:
                        ref_data = pickle.load(f)
                    first_key = list(ref_data.keys())[0]
                    self.gait_period = ref_data[first_key]['period']
                    print(f"  Loaded gait period from reference motion file: {self.gait_period:.4f}s")
                    print(f"  Reference motion: {reference_motion_path}")
                except Exception as e:
                    print(f"  Warning: Could not load reference motion: {e}")

            if not reference_motion_path and self.default_gait_period_from_onnx is not None:
                self.gait_period = self.default_gait_period_from_onnx
                print(f"  Using gait period from ONNX metadata: {self.gait_period:.4f}s")

            if reference_motion_path is None and self.default_gait_period_from_onnx is None:
                print(f"  Warning: No gait period found in ONNX or reference motion")
                print(f"  Using fallback default period: {self.gait_period:.4f}s")

        # Action delay buffer
        self.use_delay = self.delay_max_lag > 0
        if self.use_delay:
            buffer_size = self.delay_max_lag + 1
            self.action_buffer = [np.zeros(self.n_joints, dtype=np.float32) for _ in range(buffer_size)]
            self.buffer_index = 0
            self.current_lag = np.random.randint(self.delay_min_lag, self.delay_max_lag + 1)
            print(f"\nActuator delay enabled:")
            print(f"  Min lag: {self.delay_min_lag} timesteps")
            print(f"  Max lag: {self.delay_max_lag} timesteps")
            print(f"  Sampled lag: {self.current_lag} timesteps")
            print(f"  Buffer size: {buffer_size}")
        else:
            self.action_buffer = None
            self.current_lag = 0

    def _update_command(self):
        """Update self.command (fed into obs) based on current policy and commands."""
        if self.current_policy == "walking":
            self.command = self.vel_cmd.copy()
        elif self.current_policy == "standing":
            # Normalize body pose cmd to match training's body_pose_cmd_obs
            self.command = np.array([
                self.body_cmd[0] / BODY_CMD_MAX_Z,
                self.body_cmd[1] / BODY_CMD_MAX_ANGLE,
                self.body_cmd[2] / BODY_CMD_MAX_ANGLE,
            ], dtype=np.float32)
        # ground_pick: command is set directly by update_ground_pick_phase

    def _update_policy_session(self):
        """Switch between walking and standing sessions based on vel_cmd magnitude."""
        if not (self.walking_session and self.standing_session):
            return  # Only one policy loaded, no switching
        if self.ground_pick_mode:
            return  # Don't switch during ground pick

        magnitude = float(np.linalg.norm(self.vel_cmd))
        new_policy = "standing" if magnitude <= self.switch_threshold else "walking"
        if new_policy != self.current_policy:
            self.current_policy = new_policy
            self.ort_session = self.standing_session if new_policy == "standing" else self.walking_session
            print(f"Switched to {self.current_policy} policy (vel magnitude: {magnitude:.3f})")
            self._update_command()

    def set_vel_cmd(self, lin_vel_x=0.0, lin_vel_y=0.0, ang_vel_z=0.0):
        """Set velocity command (used for walking / policy switching)."""
        self.vel_cmd = np.array([lin_vel_x, lin_vel_y, ang_vel_z], dtype=np.float32)
        self._update_policy_session()
        self._update_command()
        print(f"Vel cmd: [{lin_vel_x:.2f}, {lin_vel_y:.2f}, {ang_vel_z:.2f}] [{self.current_policy}]")

    def toggle_body_pose_mode(self):
        """Toggle body pose control mode on/off."""
        self.body_pose_mode = not self.body_pose_mode
        if self.body_pose_mode:
            print("Body pose mode: ON")
            print(f"  UP/DOWN: Δz ±{self.body_cmd_step_z*1000:.0f}mm  (max ±{BODY_CMD_MAX_Z*1000:.0f}mm)")
            print(f"  LEFT/RIGHT: Δpitch ±1°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
            print(f"  A/E: Δroll ±1°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
            print(f"  SPACE: reset body pose to zero")
            print(f"  Current: z={self.body_cmd[0]*1000:.1f}mm  pitch={math.degrees(self.body_cmd[1]):.1f}°  roll={math.degrees(self.body_cmd[2]):.1f}°")
        else:
            print("Body pose mode: OFF")

    def _print_body_cmd(self):
        print(f"Body cmd: z={self.body_cmd[0]*1000:.1f}mm  pitch={math.degrees(self.body_cmd[1]):.1f}°  roll={math.degrees(self.body_cmd[2]):.1f}°")

    def quat_rotate_inverse(self, quat, vec):
        """Rotate a vector by the inverse of a quaternion [w, x, y, z]."""
        w = quat[0]
        xyz = quat[1:4]
        t = np.cross(xyz, vec) * 2
        return vec - w * t + np.cross(xyz, t)

    def get_raw_accelerometer(self):
        """Get raw accelerometer reading from MuJoCo sensor."""
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        if sensor_id < 0:
            raise ValueError("Sensor 'imu_accel' not found in model")

        sensor_adr = self.model.sensor_adr[sensor_id]
        accel_raw = self.data.sensordata[sensor_adr:sensor_adr+3].copy().astype(np.float32)
        accel_negated = -accel_raw
        mag = np.linalg.norm(accel_negated)
        if mag > 0.1:
            return accel_negated / mag
        else:
            quat = self.data.xquat[self.trunk_base_id].copy().astype(np.float32)
            world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            return self.quat_rotate_inverse(quat, world_gravity)

    def get_projected_gravity(self):
        """Get projected gravity in body frame."""
        quat = self.data.xquat[self.trunk_base_id].copy().astype(np.float32)
        world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return self.quat_rotate_inverse(quat, world_gravity)

    def get_base_ang_vel(self):
        """Get base angular velocity from IMU gyro sensor."""
        sensor_adr = self.model.sensor_adr[self.imu_ang_vel_id]
        return self.data.sensordata[sensor_adr:sensor_adr + 3].copy().astype(np.float32)

    def get_joint_pos_relative(self):
        """Get joint positions relative to default pose."""
        current_pos = self.data.qpos[self.joint_qpos_indices].copy().astype(np.float32)
        return current_pos - self.default_pose

    def get_joint_vel(self):
        """Get joint velocities."""
        return self.data.qvel[self.joint_qvel_indices].copy().astype(np.float32)

    def get_imitation_phase_obs(self):
        """Get imitation phase observation as [cos(2π*phase), sin(2π*phase)]."""
        phase_rad = self.imitation_phase * 2 * np.pi
        return np.array([np.cos(phase_rad), np.sin(phase_rad)], dtype=np.float32)

    def update_phase(self, dt):
        """Update the gait phase based on elapsed time."""
        if self.use_imitation:
            self.imitation_phase += dt / self.gait_period
            self.imitation_phase = self.imitation_phase % 1.0

    def get_observations(self):
        """Collect observations matching policy input.

        Order for velocity/standing task (no imitation):
        1. base_ang_vel (3D)
        2. raw_accelerometer OR projected_gravity (3D)
        3. joint_pos (14D) - relative to default
        4. joint_vel (14D)
        5. actions (14D) - last action
        6. command (3D) - vel cmd (walking) or normalized body pose cmd (standing)
        Total: 51D

        Order for imitation task (use_imitation=True):
        1. command (3D)
        2. phase (2D)
        3. base_ang_vel (3D)
        4. raw_accelerometer OR projected_gravity (3D)
        5. joint_pos (14D)
        6. joint_vel (14D)
        7. actions (14D)
        Total: 53D
        """
        obs = []

        if self.use_imitation:
            obs.append(self.command)
            obs.append(self.get_imitation_phase_obs())

        obs.append(self.get_base_ang_vel())

        if self.use_projected_gravity:
            obs.append(self.get_projected_gravity())
        else:
            obs.append(self.get_raw_accelerometer())

        obs.append(self.get_joint_pos_relative())
        obs.append(self.get_joint_vel())
        obs.append(self.last_action)

        if not self.use_imitation:
            obs.append(self.command)

        return np.concatenate(obs).astype(np.float32)

    def trigger_ground_pick(self):
        """Start one ground pick cycle. Automatically returns to walking when done."""
        if self.ground_pick_session is None:
            print("Ground pick unavailable: no --ground-pick policy loaded")
            return
        if self.ground_pick_mode:
            print("Ground pick already in progress")
            return
        self.ground_pick_mode = True
        self.ground_pick_phase = 0.0
        self.ort_session = self.ground_pick_session
        self.current_policy = "ground_pick"
        print(f"Ground pick: started (period={self.ground_pick_period:.1f}s)")

    def _end_ground_pick(self):
        """Switch back after a ground pick cycle completes."""
        self.ground_pick_mode = False
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        if self.walking_session:
            self.current_policy = "walking"
            self.ort_session = self.walking_session
        else:
            self.current_policy = "standing"
            self.ort_session = self.standing_session
        self._update_command()
        print(f"Ground pick: done → back to {self.current_policy}")

    def update_ground_pick_phase(self, dt: float):
        """Advance the ground pick phase; auto-exit when one full cycle completes."""
        if not self.ground_pick_mode:
            return
        new_phase = self.ground_pick_phase + dt / self.ground_pick_period
        if new_phase >= 0.7:
            self._end_ground_pick()
            return
        self.ground_pick_phase = new_phase
        self.command[0] = np.cos(2 * np.pi * self.ground_pick_phase)
        self.command[1] = np.sin(2 * np.pi * self.ground_pick_phase)
        self.command[2] = 0.0

    def toggle_head_mode(self):
        """Toggle head control mode on/off."""
        self.head_mode = not self.head_mode
        if self.head_mode:
            print("Head mode: ON")
            print(f"  Z/S: neck_pitch  |  UP/DOWN: head_pitch  |  LEFT/RIGHT: head_yaw  |  A/E: head_roll  |  SPACE: reset  (max ±{self.head_max:.2f} rad)")
        else:
            print("Head mode: OFF")

    def infer(self):
        """Run policy inference and return action."""
        obs = self.get_observations()
        obs_batch = obs.reshape(1, -1)
        action = self.ort_session.run([self.output_name], {self.input_name: obs_batch})[0]
        action = action.squeeze(0).astype(np.float32)
        self.last_action = action.copy()
        return action

    def apply_action(self, action):
        """Apply action to MuJoCo controls with optional delay."""
        if self.use_delay:
            self.action_buffer[self.buffer_index] = action.copy()
            delayed_index = (self.buffer_index - self.current_lag) % len(self.action_buffer)
            delayed_action = self.action_buffer[delayed_index]
            self.buffer_index = (self.buffer_index + 1) % len(self.action_buffer)
            target_positions = self.default_pose + delayed_action * self.action_scale
        else:
            target_positions = self.default_pose + action * self.action_scale

        self.data.ctrl[:] = target_positions
        self.data.ctrl[5:9] += self.head_offset


def main():
    parser = argparse.ArgumentParser(description="Run ONNX policy in MuJoCo")
    parser.add_argument("--roller", action="store_true", help="Use roller skate robot XML (robot_walk_rollers.xml)")
    parser.add_argument("--walking", type=str, default=None, help="Path to walking policy ONNX file")
    parser.add_argument("--standing", "-s", type=str, default=None, help="Path to standing policy ONNX file")
    parser.add_argument("--ground-pick", type=str, default=None, help="Path to ground pick policy ONNX file (press G to activate)")
    parser.add_argument("--lin-vel-x", type=float, default=0.0, help="Initial linear velocity X command (m/s)")
    parser.add_argument("--lin-vel-y", type=float, default=0.0, help="Initial linear velocity Y command (m/s)")
    parser.add_argument("--ang-vel-z", type=float, default=0.0, help="Initial angular velocity Z command (rad/s)")
    parser.add_argument("--action-scale", type=float, default=1.0, help="Action scale (default: 1.0)")
    parser.add_argument("--imitation", action="store_true", help="Enable imitation mode (adds phase observation)")
    parser.add_argument("--reference-motion", type=str, default=None, help="Path to reference motion .pkl file (for imitation)")
    parser.add_argument("--raw-accelerometer", action="store_true", help="Use raw accelerometer instead of projected gravity")
    parser.add_argument("--delay", type=int, nargs='*', default=None, help="Enable actuator delay: --delay MIN MAX or --delay LAG")
    parser.add_argument("--debug", action="store_true", help="Print observations and actions")
    parser.add_argument("--save-csv", type=str, default=None, help="Save observations and actions to CSV file")
    parser.add_argument("--record", type=str, default=None, help="Enable recording mode: save observations to pickle file on Ctrl+C")
    parser.add_argument("--switch-threshold", type=float, default=0.05, help="Vel command magnitude threshold for walking/standing switch (default: 0.05)")
    parser.add_argument("--ground-pick-period", type=float, default=4.0, help="Ground pick phase period in seconds (default: 4.0)")
    args = parser.parse_args()

    if not args.walking and not args.standing:
        parser.error("At least one of --walking or --standing must be provided")

    # Parse delay arguments
    delay_min_lag = 0
    delay_max_lag = 0
    if args.delay is not None:
        if len(args.delay) == 0:
            delay_min_lag = 1
            delay_max_lag = 2
        elif len(args.delay) == 1:
            delay_min_lag = args.delay[0]
            delay_max_lag = args.delay[0]
        elif len(args.delay) == 2:
            delay_min_lag = args.delay[0]
            delay_max_lag = args.delay[1]
        else:
            print("Error: --delay accepts 0, 1, or 2 arguments")
            return

    # Load MuJoCo model
    xml_path = MICRODUCK_ROLLERS_XML if args.roller else MICRODUCK_XML
    print(f"Loading MuJoCo model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)

    # Initialize policy
    policy = PolicyInference(
        model, data,
        walking_onnx_path=args.walking,
        action_scale=args.action_scale,
        use_imitation=args.imitation,
        reference_motion_path=args.reference_motion,
        delay_min_lag=delay_min_lag,
        delay_max_lag=delay_max_lag,
        standing_onnx_path=args.standing,
        switch_threshold=args.switch_threshold,
        use_projected_gravity=not args.raw_accelerometer,
        ground_pick_onnx_path=args.ground_pick,
        ground_pick_period=args.ground_pick_period,
    )
    policy.set_vel_cmd(args.lin_vel_x, args.lin_vel_y, args.ang_vel_z)

    # Set initial position to default pose
    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = model.jnt_qposadr[freejoint_id]
    data.qpos[qpos_adr + 0] = 0.0
    data.qpos[qpos_adr + 1] = 0.0
    data.qpos[qpos_adr + 2] = 0.1385 if args.roller else 0.125  # rollers add 13.5mm height
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = policy.default_pose[i]
    data.ctrl[:] = policy.default_pose
    mujoco.mj_forward(model, data)

    # Verify observation size
    test_obs = policy.get_observations()
    if policy.use_imitation:
        expected_obs_size = 3 + 2 + 3 + 3 + policy.n_joints + policy.n_joints + policy.n_joints
        breakdown = f"3(command) + 2(phase) + 3(ang_vel) + 3(proj_grav) + {policy.n_joints}(joint_pos) + {policy.n_joints}(joint_vel) + {policy.n_joints}(last_action)"
    else:
        expected_obs_size = 3 + 3 + policy.n_joints + policy.n_joints + policy.n_joints + 3
        breakdown = f"3(ang_vel) + 3(proj_grav) + {policy.n_joints}(joint_pos) + {policy.n_joints}(joint_vel) + {policy.n_joints}(last_action) + 3(command)"

    if test_obs.size != expected_obs_size:
        print(f"\nWARNING: Observation size mismatch!")
        print(f"  Expected: {expected_obs_size}")
        print(f"  Got: {test_obs.size}")
        print(f"  Breakdown: {breakdown}")
        print()

    print("\n" + "="*80)
    print("MicroDuck Policy Inference")
    print("="*80)
    print(f"Control frequency: 50 Hz (decimation: 4)")
    print(f"Simulation timestep: {model.opt.timestep}s")
    print(f"Observation size: {test_obs.size} (expected: {expected_obs_size})")
    if policy.use_imitation:
        print(f"Imitation mode: ENABLED (gait period: {policy.gait_period:.3f}s)")
    if policy.walking_session:
        print(f"Walking policy: loaded")
    if policy.standing_session:
        print(f"Standing policy: loaded  (body pose: z=±{BODY_CMD_MAX_Z*1000:.0f}mm, pitch/roll=±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
    if policy.walking_session and policy.standing_session:
        print(f"  Switch threshold: {policy.switch_threshold} (vel cmd magnitude)")
    if policy.ground_pick_session:
        print(f"Ground pick policy: loaded  (press G)")
    print(f"Active policy: {policy.current_policy}")
    print("Close viewer window to exit")
    print()

    decimation = 4
    control_step_count = 0
    control_dt = decimation * model.opt.timestep

    csv_data = [] if args.save_csv else None
    recorded_observations = [] if args.record else None
    policy_enabled = not args.record
    policy_enable_time = None
    original_kp = None
    if args.record:
        original_kp = model.actuator_gainprm[:, 0].copy()

    try:
        from pynput import keyboard as pynput_keyboard

        def on_press(key):
            try:
                if key == pynput_keyboard.Key.up:
                    if policy.head_mode:
                        policy.head_offset[1] = np.clip(policy.head_offset[1] + policy.head_step, -policy.head_max, policy.head_max)
                        print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                    elif policy.body_pose_mode:
                        policy.body_cmd[0] = np.clip(policy.body_cmd[0] + policy.body_cmd_step_z, -BODY_CMD_MAX_Z, BODY_CMD_MAX_Z)
                        policy._update_command()
                        policy._print_body_cmd()
                    else:
                        policy.set_vel_cmd(0.5, 0.0, 0.0)
                elif key == pynput_keyboard.Key.down:
                    if policy.head_mode:
                        policy.head_offset[1] = np.clip(policy.head_offset[1] - policy.head_step, -policy.head_max, policy.head_max)
                        print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                    elif policy.body_pose_mode:
                        policy.body_cmd[0] = np.clip(policy.body_cmd[0] - policy.body_cmd_step_z, -BODY_CMD_MAX_Z, BODY_CMD_MAX_Z)
                        policy._update_command()
                        policy._print_body_cmd()
                    else:
                        policy.set_vel_cmd(-0.5, 0.0, 0.0)
                elif key == pynput_keyboard.Key.right:
                    if policy.head_mode:
                        policy.head_offset[2] = np.clip(policy.head_offset[2] - policy.head_step, -policy.head_max, policy.head_max)
                        print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                    elif policy.body_pose_mode:
                        policy.body_cmd[1] = np.clip(policy.body_cmd[1] - policy.body_cmd_step_angle, -BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE)
                        policy._update_command()
                        policy._print_body_cmd()
                    else:
                        policy.set_vel_cmd(0.0, -0.5, 0.0)
                elif key == pynput_keyboard.Key.left:
                    if policy.head_mode:
                        policy.head_offset[2] = np.clip(policy.head_offset[2] + policy.head_step, -policy.head_max, policy.head_max)
                        print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                    elif policy.body_pose_mode:
                        policy.body_cmd[1] = np.clip(policy.body_cmd[1] + policy.body_cmd_step_angle, -BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE)
                        policy._update_command()
                        policy._print_body_cmd()
                    else:
                        policy.set_vel_cmd(0.0, 0.5, 0.0)
                elif key == pynput_keyboard.Key.space:
                    if policy.head_mode:
                        policy.head_offset[:] = 0.0
                        print("Head offset reset to zero")
                    elif policy.body_pose_mode:
                        policy.body_cmd[:] = 0.0
                        policy._update_command()
                        print("Body pose cmd reset to zero")
                    else:
                        policy.set_vel_cmd(0.0, 0.0, 0.0)
                elif hasattr(key, 'char'):
                    if key.char == 'g' or key.char == 'G':
                        policy.trigger_ground_pick()
                    elif key.char == 'h' or key.char == 'H':
                        policy.toggle_head_mode()
                    elif key.char == 'b' or key.char == 'B':
                        policy.toggle_body_pose_mode()
                    elif key.char == 'a' or key.char == 'A':
                        if policy.head_mode:
                            policy.head_offset[3] = np.clip(policy.head_offset[3] + policy.head_step, -policy.head_max, policy.head_max)
                            print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                        elif policy.body_pose_mode:
                            policy.body_cmd[2] = np.clip(policy.body_cmd[2] + policy.body_cmd_step_angle, -BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE)
                            policy._update_command()
                            policy._print_body_cmd()
                        else:
                            policy.set_vel_cmd(0.0, 0.0, 4.0)
                    elif key.char == 'e' or key.char == 'E':
                        if policy.head_mode:
                            policy.head_offset[3] = np.clip(policy.head_offset[3] - policy.head_step, -policy.head_max, policy.head_max)
                            print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                        elif policy.body_pose_mode:
                            policy.body_cmd[2] = np.clip(policy.body_cmd[2] - policy.body_cmd_step_angle, -BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE)
                            policy._update_command()
                            policy._print_body_cmd()
                        else:
                            policy.set_vel_cmd(0.0, 0.0, -4.0)
                    elif key.char == 'z' or key.char == 'Z':
                        if policy.head_mode:
                            policy.head_offset[0] = np.clip(policy.head_offset[0] + policy.head_step, -policy.head_max, policy.head_max)
                            print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                    elif key.char == 's' or key.char == 'S':
                        if policy.head_mode:
                            policy.head_offset[0] = np.clip(policy.head_offset[0] - policy.head_step, -policy.head_max, policy.head_max)
                            print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
            except Exception as e:
                print(f"Key press error: {e}")

        listener = pynput_keyboard.Listener(on_press=on_press)
        listener.start()

        print("\nKeyboard controls:")
        print("  [ Velocity mode (default) ]")
        print("  UP/DOWN arrow:    lin_vel_x ±0.5")
        print("  LEFT/RIGHT arrow: lin_vel_y ±0.5")
        print("  A / E:            ang_vel_z ±4.0")
        print("  SPACE:            stop (zero velocity)")
        print("  G:                trigger ground pick (requires --ground-pick)")
        print("  [ Body pose mode — press B to toggle (requires --standing) ]")
        print(f"  UP/DOWN arrow:    Δz ±1mm  (max ±{BODY_CMD_MAX_Z*1000:.0f}mm)")
        print(f"  LEFT/RIGHT arrow: Δpitch ±1°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
        print(f"  A / E:            Δroll ±1°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
        print("  SPACE:            reset body pose to zero")
        print("  [ Head mode — press H to toggle ]")
        print("  Z / S:            neck_pitch ±step")
        print("  UP/DOWN arrow:    head_pitch ±step")
        print("  LEFT/RIGHT arrow: head_yaw ±step")
        print("  A / E:            head_roll ±step")
        print("  SPACE:            reset head offset to zero")
        print("\nNote: Keyboard listener captures keys system-wide")

    except ImportError:
        print("\nKeyboard control unavailable: pynput not found. Install with: pip install pynput")
    except Exception as e:
        print(f"\nCould not enable keyboard controls: {e}")
        import traceback
        traceback.print_exc()

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.sync()
        start_time = time.time()

        if args.record:
            policy_enable_time = start_time + 1.0
            print("Recording mode: policy will be enabled after 1 second standby")
            for i in range(model.nu):
                model.actuator_gainprm[i, 0] = 2.0
                model.actuator_biasprm[i, 1] = -2.0
            print("  Standby mode: kp set to 2.0")

        try:
            prev_step_time = time.time()

            while viewer.is_running():
                step_start = time.time()

                if not policy_enabled and policy_enable_time is not None:
                    if step_start >= policy_enable_time:
                        policy_enabled = True
                        if original_kp is not None:
                            for i in range(model.nu):
                                kp = original_kp[i]
                                model.actuator_gainprm[i, 0] = kp
                                model.actuator_biasprm[i, 1] = -kp
                            print("Policy inference enabled (after 1s standby)")
                            print(f"  Restored original kp gains (range: [{original_kp.min():.2f}, {original_kp.max():.2f}])")

                actual_dt = step_start - prev_step_time
                prev_step_time = step_start

                policy.update_phase(actual_dt)
                policy.update_ground_pick_phase(actual_dt)

                if policy_enabled:
                    action = policy.infer()
                else:
                    action = np.zeros(policy.n_joints, dtype=np.float32)
                policy.apply_action(action)

                control_step_count += 1

                if csv_data is not None:
                    obs = policy.get_observations()
                    row = {'step': control_step_count, 'time': control_step_count * control_dt}
                    for i in range(obs.size):
                        row[f'obs_{i}'] = obs[i]
                    for i in range(action.size):
                        row[f'action_{i}'] = action[i]
                    csv_data.append(row)

                if recorded_observations is not None:
                    obs = policy.get_observations()
                    timestamp = time.time() - start_time
                    recorded_observations.append({'timestamp': timestamp, 'observation': obs.tolist()})

                if args.debug:
                    should_print = control_step_count <= 10 or control_step_count % 50 == 0
                    if should_print:
                        obs = policy.get_observations()
                        pos = data.qpos[qpos_adr:qpos_adr + 3]
                        quat = data.qpos[qpos_adr + 3:qpos_adr + 7]
                        com_height = pos[2]

                        print(f"\n{'='*70}")
                        print(f"Step {control_step_count} DEBUG:")
                        print(f"{'='*70}")
                        if policy.use_imitation:
                            print(f"Imitation phase: {policy.imitation_phase:.4f} (period: {policy.gait_period:.3f}s)")
                        print(f"Active policy: {policy.current_policy}")
                        print(f"Base state:")
                        print(f"  Position: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
                        print(f"  CoM height: {com_height:7.4f}")
                        print(f"  Quaternion: [{quat[0]:7.4f}, {quat[1]:7.4f}, {quat[2]:7.4f}, {quat[3]:7.4f}]")
                        print(f"\nObservation (shape {obs.shape}, total {obs.size}):")
                        if policy.use_imitation:
                            print(f"  Command [0:3]:        {obs[0:3]}")
                            print(f"  Phase [3:5]:          {obs[3:5]} (cos, sin)")
                            print(f"  Ang vel [5:8]:        {obs[5:8]}")
                            print(f"  Proj grav [8:11]:     {obs[8:11]}")
                            joint_start = 11
                            print(f"  Joint pos [{joint_start}:{joint_start+policy.n_joints}]:     {obs[joint_start:joint_start+policy.n_joints]}")
                            print(f"  Joint vel [{joint_start+policy.n_joints}:{joint_start+2*policy.n_joints}]:    {obs[joint_start+policy.n_joints:joint_start+2*policy.n_joints]}")
                            print(f"  Last action [{joint_start+2*policy.n_joints}:{joint_start+3*policy.n_joints}]:  {obs[joint_start+2*policy.n_joints:joint_start+3*policy.n_joints]}")
                        else:
                            print(f"  Ang vel [0:3]:        {obs[0:3]}")
                            print(f"  Proj grav [3:6]:      {obs[3:6]}")
                            print(f"  Joint pos [6:{6+policy.n_joints}]:     {obs[6:6+policy.n_joints]}")
                            print(f"  Joint vel [{6+policy.n_joints}:{6+2*policy.n_joints}]:    {obs[6+policy.n_joints:6+2*policy.n_joints]}")
                            print(f"  Last action [{6+2*policy.n_joints}:{6+3*policy.n_joints}]:  {obs[6+2*policy.n_joints:6+3*policy.n_joints]}")
                            cmd_end = 6+3*policy.n_joints+3
                            print(f"  Command [{6+3*policy.n_joints}:{cmd_end}]:      {obs[6+3*policy.n_joints:cmd_end]}")
                        if policy.current_policy == "standing":
                            print(f"  Body cmd (raw): z={policy.body_cmd[0]*1000:.1f}mm  pitch={math.degrees(policy.body_cmd[1]):.1f}°  roll={math.degrees(policy.body_cmd[2]):.1f}°")
                        print(f"\nAction output:")
                        print(f"  Raw action: {action}")
                        print(f"  Action min/max: [{action.min():.4f}, {action.max():.4f}]")
                        if policy.use_delay:
                            print(f"  Delay: {policy.current_lag} timesteps (buffered)")
                        print(f"  Applied ctrl (first 5): {data.ctrl[:5]}")
                        print(f"  Applied ctrl (last 5):  {data.ctrl[-5:]}")

                for _ in range(decimation):
                    mujoco.mj_step(model, data)

                viewer.sync()

                elapsed = time.time() - step_start
                sleep_time = control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\nKeyboardInterrupt received (Ctrl+C). Saving data...")

    print("\nInference stopped.")

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

    if recorded_observations is not None and len(recorded_observations) > 0:
        print(f"\nSaving {len(recorded_observations)} recorded observations to: {args.record}")
        with open(args.record, 'wb') as f:
            pickle.dump(recorded_observations, f)
        print(f"Recorded observations saved to {args.record}")
        print(f"  Observations: {len(recorded_observations)}")
        print(f"  Duration: {recorded_observations[-1]['timestamp']:.2f}s")


if __name__ == "__main__":
    main()
