#!/usr/bin/env python3
"""Check for sim2sim differences between mjlab and vanilla MuJoCo."""

import numpy as np
import mujoco

# Load model
MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene.xml"
model = mujoco.MjModel.from_xml_path(MICRODUCK_XML)
data = mujoco.MjData(model)

# Set to default pose
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

freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
qpos_adr = model.jnt_qposadr[freejoint_id]

# Set base
data.qpos[qpos_adr:qpos_adr+3] = [0, 0, 0.125]
data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]

# Set joints
data.qpos[7:7+14] = DEFAULT_POSE
data.ctrl[:] = DEFAULT_POSE

# Forward
mujoco.mj_forward(model, data)

print("="*70)
print("SIM2SIM DIAGNOSTIC")
print("="*70)

print("\n1. MODEL PROPERTIES:")
print(f"   Timestep (XML default): {model.opt.timestep:.6f}s")
print(f"   NOTE: mjlab overrides this to 0.005s for velocity environments!")
print(f"   Number of actuators: {model.nu}")
print(f"   Number of joints: {model.njnt}")
print(f"   Number of DOFs: {model.nv}")

print("\n2. ACTUATOR PROPERTIES:")
for i in range(model.nu):
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    joint_id = model.actuator_trnid[i, 0]
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)

    # Get gain (kp) and bias
    gain = model.actuator_gainprm[i, 0]  # kp for position actuators
    bias = model.actuator_biasprm[i, 1]  # kv for position actuators
    forcerange = model.actuator_forcerange[i]

    # Get DOF address for this joint (joints can have multiple DOFs)
    dof_adr = model.jnt_dofadr[joint_id]

    # Get joint properties from DOF
    damping = model.dof_damping[dof_adr]
    armature = model.dof_armature[dof_adr]
    frictionloss = model.dof_frictionloss[dof_adr]

    print(f"   {i:2d}. {actuator_name:20s} -> {joint_name:20s} (joint_id={joint_id}, dof_adr={dof_adr})")
    print(f"       Actuator: kp={gain:6.2f}, kv={bias:6.2f}, force=[{forcerange[0]:6.2f}, {forcerange[1]:6.2f}]")
    print(f"       Joint:    damping={damping:6.3f}, armature={armature:6.4f}, friction={frictionloss:6.3f}")

print("\n3. SENSOR CHECK:")
# Check gyro sensor
imu_ang_vel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
sensor_adr = model.sensor_adr[imu_ang_vel_id]
sensor_data = data.sensordata[sensor_adr:sensor_adr+3]

# Get body angular velocity in body frame (for comparison)
trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
trunk_quat = data.xquat[trunk_id]  # Body quaternion
trunk_ang_vel_world = data.qvel[3:6]  # Angular velocity in world frame (from freejoint)

# Rotate to body frame manually
def quat_rotate_inverse(quat, vec):
    """Rotate vector by quaternion inverse."""
    w, x, y, z = quat
    t2 = w * x
    t3 = w * y
    t4 = w * z
    t5 = -x * x
    t6 = x * y
    t7 = x * z
    t8 = -y * y
    t9 = y * z
    t10 = -z * z

    return np.array([
        2 * ((t8 + t10) * vec[0] + (t6 - t4) * vec[1] + (t3 + t7) * vec[2]) + vec[0],
        2 * ((t4 + t6) * vec[0] + (t5 + t10) * vec[1] + (t9 - t2) * vec[2]) + vec[1],
        2 * ((t7 - t3) * vec[0] + (t2 + t9) * vec[1] + (t5 + t8) * vec[2]) + vec[2]
    ])

trunk_ang_vel_body = quat_rotate_inverse(trunk_quat, trunk_ang_vel_world)

print(f"   IMU gyro sensor:       {sensor_data}")
print(f"   Computed body ang_vel: {trunk_ang_vel_body}")
print(f"   Difference:            {sensor_data - trunk_ang_vel_body}")
print(f"   World frame ang_vel:   {trunk_ang_vel_world}")

print("\n4. DEFAULT JOINT POSITIONS:")
print(f"   model.qpos0[7:21] (default):   {model.qpos0[7:21]}")
print(f"   data.qpos[7:21] (current):     {data.qpos[7:21]}")
print(f"   DEFAULT_POSE (policy offset):  {DEFAULT_POSE}")

print("\n5. ACTION SPACE:")
print(f"   Action dimension: {model.nu}")
print(f"   Control range check:")
for i in range(min(5, model.nu)):
    ctrlrange = model.actuator_ctrlrange[i]
    print(f"      Actuator {i}: [{ctrlrange[0]:7.3f}, {ctrlrange[1]:7.3f}]")

print("\n6. KEY DIFFERENCES TO CHECK:")
print("   [x] Actuator delay: mjlab uses DelayedActuatorCfg(delay_min_lag=1, delay_max_lag=2)")
print("   [x] Observation noise: mjlab adds noise during training (check if disabled in play)")
print("   [x] Observation delay: some observations have delay_min_lag/max_lag/update_period")
print("   [ ] Action clipping: check if actions are clipped differently")
print("   [ ] Default pose offset: check if mjlab uses same offset")

print("\n" + "="*70)
