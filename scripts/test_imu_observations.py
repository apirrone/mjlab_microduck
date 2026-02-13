#!/usr/bin/env python3
"""Test IMU observations (angular velocity + projected gravity) in MuJoCo.

Compare coordinate frames and signs between simulation and real robot.
"""

import numpy as np
import mujoco
import argparse
from pathlib import Path


def quat_to_euler(quat):
    """Convert quaternion [w,x,y,z] to Euler angles [roll, pitch, yaw] in degrees."""
    w, x, y, z = quat
    
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.degrees([roll, pitch, yaw])


def compute_observations(model, data):
    """Compute angular velocity and projected gravity as policy sees them."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    
    # Angular velocity in body frame (what gyro measures)
    ang_vel = data.sensor('angular-velocity').data.copy()
    
    # Projected gravity (rotate world gravity into body frame)
    quat = data.xquat[body_id].copy()  # [w, x, y, z]
    w, x, y, z = quat
    world_gravity = np.array([0, 0, -1.0])
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
    proj_grav = proj_grav / np.linalg.norm(proj_grav)
    
    return ang_vel, proj_grav, quat


def test_orientations(model_path):
    """Test robot orientations and show expected observations."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    # Disable gravity
    model.opt.gravity[:] = 0
    
    print("=" * 90)
    print("IMU OBSERVATIONS TEST - MuJoCo Simulation")
    print("=" * 90)
    print()
    print("Legend:")
    print("  ω (ang_vel): Angular velocity in body frame [rad/s]")
    print("  g (proj_grav): Projected gravity (unit vector)")
    print()
    print("Sign conventions (right-hand rule):")
    print("  ω_x > 0: Rolling RIGHT")
    print("  ω_y > 0: Pitching FORWARD (nose down)")
    print("  ω_z > 0: Yawing LEFT (counter-clockwise from above)")
    print()
    print("Gravity conventions:")
    print("  g_x < 0: Tilted FORWARD")
    print("  g_x > 0: Tilted BACKWARD")
    print("  g_y < 0: Tilted RIGHT")
    print("  g_y > 0: Tilted LEFT")
    print()
    
    # Static poses (no angular velocity)
    print("-" * 90)
    print("STATIC POSES (no rotation, just orientation)")
    print("-" * 90)
    print()
    
    static_tests = [
        ("Upright", [1, 0, 0, 0]),
        ("Pitch forward 15°", [0.9914, 0, 0.1305, 0]),
        ("Pitch backward 15°", [0.9914, 0, -0.1305, 0]),
        ("Roll right 15°", [0.9914, 0.1305, 0, 0]),
        ("Roll left 15°", [0.9914, -0.1305, 0, 0]),
    ]
    
    for name, quat in static_tests:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        data.qpos[3:7] = quat
        data.qvel[:] = 0  # Zero velocity
        mujoco.mj_forward(model, data)
        
        ang_vel, proj_grav, actual_quat = compute_observations(model, data)
        euler = quat_to_euler(actual_quat)
        
        print(f"{name}:")
        print(f"  Orientation: [roll={euler[0]:6.1f}°, pitch={euler[1]:6.1f}°, yaw={euler[2]:6.1f}°]")
        print(f"  ω = [{ang_vel[0]:7.3f}, {ang_vel[1]:7.3f}, {ang_vel[2]:7.3f}] rad/s  (should be ~0)")
        print(f"  g = [{proj_grav[0]:7.4f}, {proj_grav[1]:7.4f}, {proj_grav[2]:7.4f}]")
        print()
    
    # Dynamic tests (rotating)
    print("-" * 90)
    print("ROTATING MOTIONS (what angular velocity looks like)")
    print("-" * 90)
    print()
    print("When you tilt the robot, angular velocity appears briefly during the motion.")
    print("Example: Tilting forward → positive ω_y during the tilt → stops when held")
    print()
    
    rotation_tests = [
        ("Pitching FORWARD (nose down)", [0, 1, 0], "ω_y > 0, g_x becomes negative"),
        ("Pitching BACKWARD (nose up)", [0, -1, 0], "ω_y < 0, g_x becomes positive"),
        ("Rolling RIGHT", [1, 0, 0], "ω_x > 0, g_y becomes negative"),
        ("Rolling LEFT", [-1, 0, 0], "ω_x < 0, g_y becomes positive"),
        ("Yawing LEFT", [0, 0, 1], "ω_z > 0, g_x and g_y unchanged"),
    ]
    
    for name, ang_vel_dir, description in rotation_tests:
        # Set small angular velocity
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        data.qpos[3:7] = [1, 0, 0, 0]  # Start upright
        data.qvel[3:6] = np.array(ang_vel_dir) * 2.0  # 2 rad/s rotation
        mujoco.mj_forward(model, data)
        
        ang_vel, proj_grav, _ = compute_observations(model, data)
        
        print(f"{name}:")
        print(f"  ω = [{ang_vel[0]:7.3f}, {ang_vel[1]:7.3f}, {ang_vel[2]:7.3f}] rad/s")
        print(f"  Expected: {description}")
        print()
    
    print("=" * 90)
    print()
    print("TO COMPARE WITH REAL ROBOT:")
    print("1. Record observations while manually tilting the robot")
    print("2. Check angular velocity SIGNS during tilting motion")
    print("3. Check projected gravity SIGNS in held positions")
    print("4. Verify signs match the conventions above")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="src/mjlab_microduck/robot/microduck/scene.xml",
        help="Path to MuJoCo XML model",
    )
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        exit(1)
    
    test_orientations(model_path)
