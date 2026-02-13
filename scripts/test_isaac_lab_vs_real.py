#!/usr/bin/env python3
"""Compare Isaac Lab's observation computation with real robot implementation."""

import numpy as np
import mujoco
from pathlib import Path


def quat_rotate_vec(quat, vec):
    """Rotate vector by quaternion [w,x,y,z]."""
    w, qx, qy, qz = quat
    vx, vy, vz = vec
    
    cx = qy * vz - qz * vy
    cy = qz * vx - qx * vz
    cz = qx * vy - qy * vx
    
    cx2 = cy * qz - cz * qy + w * cx
    cy2 = cz * qx - cx * qz + w * cy
    cz2 = cx * qy - cy * qx + w * cz
    
    return np.array([vx + 2.0 * cx2, vy + 2.0 * cy2, vz + 2.0 * cz2])


def test_observations():
    model_path = Path("src/mjlab_microduck/robot/microduck/scene.xml")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    # Disable gravity
    model.opt.gravity[:] = 0
    
    print("=" * 90)
    print("ISAAC LAB vs REAL ROBOT - Observation Computation")
    print("=" * 90)
    print()
    
    # Test different orientations
    test_cases = [
        ("Upright", [1, 0, 0, 0]),
        ("Pitch forward 15°", [0.9914, 0, 0.1305, 0]),
        ("Roll right 15°", [0.9914, 0.1305, 0, 0]),
    ]
    
    for name, quat in test_cases:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        
        # Set orientation
        data.qpos[3:7] = quat
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        
        print(f"{name}:")
        print(f"  Quaternion: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
        print()
        
        # Method 1: Isaac Lab's projected_gravity_b (from xmat)
        # Isaac Lab uses the rotation matrix to rotate gravity vector
        xmat = data.xmat[body_id].reshape(3, 3)  # Body rotation matrix (world to body)
        world_gravity = np.array([0, 0, -1])
        proj_grav_isaac = xmat @ world_gravity  # Rotate world gravity into body frame
        
        print(f"  Method 1 (Isaac Lab - rotation matrix):")
        print(f"    Projected gravity: [{proj_grav_isaac[0]:7.4f}, {proj_grav_isaac[1]:7.4f}, {proj_grav_isaac[2]:7.4f}]")
        
        # Method 2: Quaternion rotation (real robot implementation)
        proj_grav_quat = quat_rotate_vec(quat, world_gravity)
        proj_grav_quat = proj_grav_quat / np.linalg.norm(proj_grav_quat)
        
        print(f"  Method 2 (Real robot - quaternion):")
        print(f"    Projected gravity: [{proj_grav_quat[0]:7.4f}, {proj_grav_quat[1]:7.4f}, {proj_grav_quat[2]:7.4f}]")
        
        # Compare
        diff = proj_grav_isaac - proj_grav_quat
        print(f"  Difference (Isaac - Real): [{diff[0]:7.4f}, {diff[1]:7.4f}, {diff[2]:7.4f}]")
        
        if np.allclose(proj_grav_isaac, proj_grav_quat, atol=1e-4):
            print("  ✓ MATCH")
        else:
            print("  ✗ MISMATCH!")
        print()
        
        # Check angular velocity
        # Method 1: Isaac Lab's root_link_ang_vel_b
        # This is just the body angular velocity in body frame
        ang_vel_isaac = data.sensor('angular-velocity').data.copy()
        
        # Method 2: Real robot (read from gyro)
        # Should be the same!
        print(f"  Angular velocity (sensor 'angular-velocity'):")
        print(f"    [{ang_vel_isaac[0]:7.3f}, {ang_vel_isaac[1]:7.3f}, {ang_vel_isaac[2]:7.3f}] rad/s")
        print()
    
    print("=" * 90)
    print("CONCLUSION:")
    print("=" * 90)
    print()
    print("1. Projected gravity:")
    print("   - Isaac Lab uses rotation matrix (xmat) to rotate world gravity")
    print("   - Real robot uses quaternion rotation")
    print("   - These should be equivalent (check for sign mismatches)")
    print()
    print("2. Angular velocity:")
    print("   - Both use gyro sensor reading in body frame")
    print("   - Should match directly (check BNO055 axis mapping)")
    print()


if __name__ == "__main__":
    test_observations()
