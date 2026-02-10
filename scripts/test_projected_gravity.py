#!/usr/bin/env python3
"""Test projected gravity computation in MuJoCo by tilting the robot.

This script creates scenarios with known orientations and shows what
projected gravity the policy observes. Compare with real robot data.
"""

import numpy as np
import mujoco
import argparse
from pathlib import Path


def quat_to_euler(quat):
    """Convert quaternion [w,x,y,z] to Euler angles [roll, pitch, yaw] in degrees."""
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.degrees([roll, pitch, yaw])


def compute_projected_gravity(model, data):
    """Compute projected gravity as the policy sees it."""
    # Get robot body orientation (quaternion)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    quat = data.xquat[body_id].copy()  # [w, x, y, z]
    
    # World gravity (pointing down)
    world_gravity = np.array([0, 0, -1.0])
    
    # Rotate world gravity into body frame using quaternion
    # This matches what the IMU should do
    w, x, y, z = quat
    
    # Quaternion rotation formula: v' = q * v * q^(-1)
    # Optimized version
    gx, gy, gz = world_gravity
    
    cx = y * gz - z * gy
    cy = z * gx - x * gz
    cz = x * gy - y * gx
    
    cx2 = cy * z - cz * y + w * cx
    cy2 = cz * x - cx * z + w * cy
    cz2 = cx * y - cy * x + w * cz
    
    proj_grav = np.array([
        gx + 2.0 * cx2,
        gy + 2.0 * cy2,
        gz + 2.0 * cz2,
    ])
    
    # Normalize to unit vector
    proj_grav = proj_grav / np.linalg.norm(proj_grav)
    
    return proj_grav, quat


def test_orientations(model_path):
    """Test different robot orientations and show projected gravity."""
    # Load model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    # Disable gravity so robot stays in place
    model.opt.gravity[:] = 0
    
    print("=" * 80)
    print("PROJECTED GRAVITY TEST - MuJoCo Simulation")
    print("=" * 80)
    print()
    print("Legend:")
    print("  Quaternion: [w, x, y, z] - orientation of trunk_base")
    print("  Euler: [roll, pitch, yaw] in degrees")
    print("  Proj Gravity: [x, y, z] - what the policy observes")
    print("  Expected upright: [0, 0, -1] (gravity pointing down in body frame)")
    print()
    
    # Test cases: (name, quaternion [w, x, y, z])
    test_cases = [
        ("Upright (identity)", [1, 0, 0, 0]),
        ("Pitch forward 15°", [0.9914, 0, 0.1305, 0]),  # Rotate around Y
        ("Pitch forward 30°", [0.9659, 0, 0.2588, 0]),
        ("Pitch backward 15°", [0.9914, 0, -0.1305, 0]),
        ("Roll right 15°", [0.9914, 0.1305, 0, 0]),    # Rotate around X
        ("Roll left 15°", [0.9914, -0.1305, 0, 0]),
    ]
    
    for name, quat in test_cases:
        # Set robot orientation
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        data.qpos[3:7] = quat  # Set base quaternion
        
        # Forward kinematics
        mujoco.mj_forward(model, data)
        
        # Compute projected gravity
        proj_grav, actual_quat = compute_projected_gravity(model, data)
        euler = quat_to_euler(actual_quat)
        
        print(f"{name}:")
        print(f"  Quaternion: [{actual_quat[0]:6.3f}, {actual_quat[1]:6.3f}, {actual_quat[2]:6.3f}, {actual_quat[3]:6.3f}]")
        print(f"  Euler:      [roll={euler[0]:6.1f}°, pitch={euler[1]:6.1f}°, yaw={euler[2]:6.1f}°]")
        print(f"  Proj Grav:  [{proj_grav[0]:7.4f}, {proj_grav[1]:7.4f}, {proj_grav[2]:7.4f}]")
        print()
    
    print("=" * 80)
    print()
    print("Compare these values with your real robot:")
    print("1. Hold robot in similar orientations")
    print("2. Record observations (use plotting script)")
    print("3. Check if signs and magnitudes match")
    print()
    print("Common issues to check:")
    print("  - Sign flips in any axis (coordinate frame mismatch)")
    print("  - Axis swaps (X/Y/Z might be different)")
    print("  - Quaternion conjugate (inverse rotation)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test projected gravity in MuJoCo")
    parser.add_argument(
        "--model",
        type=str,
        default="src/mjlab_microduck/assets/microduck_v2/microduck_v2.xml",
        help="Path to MuJoCo XML model",
    )
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print(f"Current directory: {Path.cwd()}")
        exit(1)
    
    test_orientations(model_path)
