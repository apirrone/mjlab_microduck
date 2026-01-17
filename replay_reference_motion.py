#!/usr/bin/env python3
"""Replay reference motion in MuJoCo to verify motion playback correctness."""

import argparse
import pickle
import time
import numpy as np
import mujoco
import mujoco.viewer

MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene.xml"


class ReferenceMotionPlayer:
    def __init__(self, model, data, reference_motion_path, motion_key=None):
        self.model = model
        self.data = data

        # Load reference motion
        print(f"Loading reference motion from: {reference_motion_path}")
        with open(reference_motion_path, 'rb') as f:
            self.ref_data = pickle.load(f)

        # Pre-compute motion categories for quick keyboard selection
        self._categorize_motions()

        # Select motion
        if motion_key is None:
            # Use first motion if not specified
            motion_key = list(self.ref_data.keys())[0]
        elif motion_key not in self.ref_data:
            print(f"Motion key '{motion_key}' not found!")
            print(f"Available motions: {list(self.ref_data.keys())}")
            raise ValueError(f"Invalid motion key: {motion_key}")

        # Joint information
        self.n_joints = model.nu
        print(f"  Number of actuators: {self.n_joints}")

        # Phase tracking
        self.phase = 0.0

        # Store initial base position for hang mode
        self.initial_base_pos = None
        self.initial_base_quat = None

        # Load the selected motion
        self.motion_key = motion_key
        self.load_motion(motion_key)

    def _categorize_motions(self):
        """Categorize motions by velocity for keyboard shortcuts."""
        self.motion_categories = {
            'forward': None,
            'backward': None,
            'left': None,
            'right': None,
            'rotate_left': None,
            'rotate_right': None,
            'stand': None,
        }

        max_dx = -float('inf')
        min_dx = float('inf')
        max_dy = -float('inf')
        min_dy = float('inf')
        max_dtheta = -float('inf')
        min_dtheta = float('inf')
        min_total_vel = float('inf')

        for key, motion in self.ref_data.items():
            dx = motion['Placo']['dx']
            dy = motion['Placo']['dy']
            dtheta = motion['Placo']['dtheta']

            # Find maximum forward (positive dx)
            if dx > max_dx:
                max_dx = dx
                self.motion_categories['forward'] = key

            # Find maximum backward (negative dx)
            if dx < min_dx:
                min_dx = dx
                self.motion_categories['backward'] = key

            # Find maximum left (positive dy)
            if dy > max_dy:
                max_dy = dy
                self.motion_categories['left'] = key

            # Find maximum right (negative dy)
            if dy < min_dy:
                min_dy = dy
                self.motion_categories['right'] = key

            # Find maximum rotate left (positive dtheta)
            if dtheta > max_dtheta:
                max_dtheta = dtheta
                self.motion_categories['rotate_left'] = key

            # Find maximum rotate right (negative dtheta)
            if dtheta < min_dtheta:
                min_dtheta = dtheta
                self.motion_categories['rotate_right'] = key

            # Find standing motion (closest to zero velocity)
            total_vel = abs(dx) + abs(dy) + abs(dtheta)
            if total_vel < min_total_vel:
                min_total_vel = total_vel
                self.motion_categories['stand'] = key

        print(f"\nMotion categories:")
        for category, key in self.motion_categories.items():
            if key:
                motion = self.ref_data[key]
                print(f"  {category:15s}: {key:20s} (dx={motion['Placo']['dx']:6.2f}, dy={motion['Placo']['dy']:6.2f}, dtheta={motion['Placo']['dtheta']:6.2f})")

    def load_motion(self, motion_key):
        """Load a specific motion."""
        self.motion_key = motion_key
        self.motion = self.ref_data[motion_key]
        self.period = self.motion['period']
        self.coeffs = self.motion['coefficients']
        self.num_dims = len(self.coeffs)

        print(f"\nLoaded motion: {motion_key}")
        print(f"  Period: {self.motion['period']:.3f}s")
        print(f"  FPS: {self.motion['fps']}")
        print(f"  Placo dx: {self.motion['Placo']['dx']:6.2f}")
        print(f"  Placo dy: {self.motion['Placo']['dy']:6.2f}")
        print(f"  Placo dtheta: {self.motion['Placo']['dtheta']:6.2f}")

    def evaluate_motion_at_phase(self, phase):
        """Evaluate polynomial at given phase [0, 1]."""
        # Clamp phase to [0, 1]
        phase = np.clip(phase, 0.0, 1.0)

        # Evaluate all polynomials using Horner's method
        results = np.zeros(self.num_dims, dtype=np.float32)

        for dim in range(self.num_dims):
            poly_coeffs = np.array(self.coeffs[f"dim_{dim}"], dtype=np.float32)
            # Horner's method: evaluate from highest to lowest degree
            result = 0.0
            for i in range(len(poly_coeffs) - 1, -1, -1):
                result = result * phase + poly_coeffs[i]
            results[dim] = result

        # Split results
        # Order: joints_pos (14), joints_vel (14), foot_contacts (2), base_linear_vel (3), base_angular_vel (3)
        idx = 0
        joints_pos = results[idx:idx+14]
        idx += 14
        joints_vel = results[idx:idx+14]
        idx += 14
        foot_contacts = results[idx:idx+2]
        idx += 2
        base_linear_vel = results[idx:idx+3]
        idx += 3
        base_angular_vel = results[idx:idx+3]

        return {
            'joints_pos': joints_pos,
            'joints_vel': joints_vel,
            'foot_contacts': foot_contacts,
            'base_linear_vel': base_linear_vel,
            'base_angular_vel': base_angular_vel,
        }

    def update_phase(self, dt):
        """Update phase based on elapsed time."""
        self.phase += dt / self.period
        self.phase = self.phase % 1.0  # Keep in [0, 1]

    def set_initial_base_pose(self, pos, quat):
        """Store initial base position for hang mode."""
        self.initial_base_pos = pos.copy()
        self.initial_base_quat = quat.copy()

    def apply_reference_motion(self, freeze_base=False):
        """Apply current reference motion to robot."""
        ref = self.evaluate_motion_at_phase(self.phase)

        if freeze_base and self.initial_base_pos is not None:
            # Reset base position to initial (freeze in space)
            # First 3 qpos: position
            self.data.qpos[0:3] = self.initial_base_pos
            # Next 4 qpos: quaternion
            self.data.qpos[3:7] = self.initial_base_quat
            # First 6 qvel: base velocities (zero them out)
            self.data.qvel[0:6] = 0.0

        # Set joint positions directly (no PD control, just kinematic replay)
        # Skip freejoint (7 qpos DOFs)
        self.data.qpos[7:7 + self.n_joints] = ref['joints_pos']

        # Set joint velocities
        # Skip freejoint (6 qvel DOFs)
        self.data.qvel[6:6 + self.n_joints] = ref['joints_vel']

        return ref


def main():
    parser = argparse.ArgumentParser(description="Replay reference motion in MuJoCo")
    parser.add_argument("reference_motion", type=str, help="Path to reference motion .pkl file")
    parser.add_argument("--motion-key", type=str, default=None, help="Motion key (e.g., '0.01_0.0_-0.4')")
    parser.add_argument("--list-motions", action="store_true", help="List available motion keys and exit")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--hang", action="store_true", help="Keep robot hanging in air (disable gravity)")
    args = parser.parse_args()

    # Load reference data to list motions
    if args.list_motions:
        with open(args.reference_motion, 'rb') as f:
            ref_data = pickle.load(f)
        print(f"\nAvailable motions in {args.reference_motion}:")
        for key in sorted(ref_data.keys()):
            motion = ref_data[key]
            print(f"  {key:20s} - period: {motion['period']:.3f}s, dx: {motion['Placo']['dx']:6.2f}, dy: {motion['Placo']['dy']:6.2f}, dtheta: {motion['Placo']['dtheta']:6.2f}")
        return

    # Load MuJoCo model
    print(f"Loading MuJoCo model from: {MICRODUCK_XML}")
    model = mujoco.MjModel.from_xml_path(MICRODUCK_XML)
    data = mujoco.MjData(model)

    # Disable gravity if hanging mode
    if args.hang:
        model.opt.gravity[:] = [0, 0, 0]
        print("Gravity disabled (hanging mode)")

    # Initialize player
    player = ReferenceMotionPlayer(model, data, args.reference_motion, args.motion_key)

    # Set initial position
    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = model.jnt_qposadr[freejoint_id]

    # Set base position
    if args.hang:
        # Hang in air
        data.qpos[qpos_adr + 0] = 0.0  # x
        data.qpos[qpos_adr + 1] = 0.0  # y
        data.qpos[qpos_adr + 2] = 0.5  # z (higher in air)
    else:
        # On ground
        data.qpos[qpos_adr + 0] = 0.0  # x
        data.qpos[qpos_adr + 1] = 0.0  # y
        data.qpos[qpos_adr + 2] = 0.125  # z

    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]  # identity quaternion

    # Store initial base pose for hang mode
    player.set_initial_base_pose(
        data.qpos[qpos_adr:qpos_adr + 3],
        data.qpos[qpos_adr + 3:qpos_adr + 7]
    )

    # Apply first frame
    player.apply_reference_motion(freeze_base=args.hang)

    # Forward kinematics
    mujoco.mj_forward(model, data)

    print("\n" + "="*80)
    print("Reference Motion Replay")
    print("="*80)
    print(f"Motion: {player.motion_key}")
    print(f"Period: {player.period:.3f}s")
    print(f"Playback speed: {args.speed}x")
    print(f"Control frequency: 50 Hz (dt = 0.02s)")
    print(f"Simulation timestep: {model.opt.timestep}s")
    if args.hang:
        print("Mode: HANGING (no gravity)")
    else:
        print("Mode: ON GROUND (with gravity)")
    print("\n" + "="*80)
    print("Keyboard Controls:")
    print("="*80)
    print("  ↑ (Up Arrow)    - Forward motion (max positive dx)")
    print("  ↓ (Down Arrow)  - Backward motion (max negative dx)")
    print("  ← (Left Arrow)  - Left translation (max positive dy)")
    print("  → (Right Arrow) - Right translation (max negative dy)")
    print("  A               - Rotate left (max positive dtheta)")
    print("  E               - Rotate right (max negative dtheta)")
    print("  S               - Stand (zero velocity motion)")
    print("  Space           - Pause/resume")
    print("="*80)
    print("Close viewer window to exit")
    print()

    # Control loop
    decimation = 4  # Match training: 50 Hz control
    control_dt = decimation * model.opt.timestep  # 0.02s
    actual_dt = control_dt / args.speed  # Adjust for playback speed
    step_count = 0

    # Keyboard callback state
    motion_changed = [False]  # Use list to allow modification in nested function

    def key_callback(keycode):
        """Handle keyboard input for motion selection."""
        # Arrow key codes (GLFW)
        KEY_UP = 265
        KEY_DOWN = 264
        KEY_LEFT = 263
        KEY_RIGHT = 262
        KEY_A = 65
        KEY_E = 69
        KEY_S = 83

        motion_key = None

        if keycode == KEY_UP:
            motion_key = player.motion_categories['forward']
            print("\n🔼 Switching to FORWARD motion")
        elif keycode == KEY_DOWN:
            motion_key = player.motion_categories['backward']
            print("\n🔽 Switching to BACKWARD motion")
        elif keycode == KEY_LEFT:
            motion_key = player.motion_categories['left']
            print("\n⬅️  Switching to LEFT motion")
        elif keycode == KEY_RIGHT:
            motion_key = player.motion_categories['right']
            print("\n➡️  Switching to RIGHT motion")
        elif keycode == KEY_A or keycode == KEY_A + 32:  # A or a
            motion_key = player.motion_categories['rotate_left']
            print("\n🔄 Switching to ROTATE LEFT motion")
        elif keycode == KEY_E or keycode == KEY_E + 32:  # E or e
            motion_key = player.motion_categories['rotate_right']
            print("\n🔃 Switching to ROTATE RIGHT motion")
        elif keycode == KEY_S or keycode == KEY_S + 32:  # S or s
            motion_key = player.motion_categories['stand']
            print("\n🧍 Switching to STAND motion")

        if motion_key and motion_key != player.motion_key:
            player.load_motion(motion_key)
            player.phase = 0.0  # Reset phase when changing motion
            motion_changed[0] = True

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            # Update phase
            player.update_phase(control_dt)

            # Apply reference motion
            ref = player.apply_reference_motion(freeze_base=args.hang)

            if args.hang:
                # Pure kinematic replay: just update kinematics without dynamics
                mujoco.mj_forward(model, data)
            else:
                # Normal simulation with dynamics
                for _ in range(decimation):
                    mujoco.mj_step(model, data)

            # Sync viewer
            viewer.sync()

            step_count += 1

            # Print status every second or when motion changes
            if step_count % 50 == 0 or motion_changed[0]:
                motion_info = f"{player.motion_key:20s} | dx={player.motion['Placo']['dx']:6.2f} dy={player.motion['Placo']['dy']:6.2f} dtheta={player.motion['Placo']['dtheta']:6.2f}"
                print(f"Step {step_count:5d} | Phase: {player.phase:.4f} | "
                      f"Contacts: L={ref['foot_contacts'][0]:.2f} R={ref['foot_contacts'][1]:.2f} | "
                      f"Motion: {motion_info}")
                motion_changed[0] = False

            # Sleep to maintain timing
            elapsed = time.time() - step_start
            sleep_time = actual_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("\nReplay stopped.")


if __name__ == "__main__":
    main()
