#!/usr/bin/env python3
"""
Analyze the sim2real gap by comparing joint responses to actions.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open('data_hang.pkl', 'rb') as f:
    real_data = pickle.load(f)
with open('sim_observations.pkl', 'rb') as f:
    sim_data = pickle.load(f)

# Extract observations - handle both formats
if isinstance(real_data, list):
    real_obs = np.array([item['observation'] for item in real_data])
    real_ts = np.array([item['timestamp'] for item in real_data])
elif isinstance(real_data, dict):
    real_obs = real_data['observations']
    real_ts = real_data['timestamps']

if isinstance(sim_data, list):
    sim_obs = np.array([item['observation'] for item in sim_data])
    sim_ts = np.array([item['timestamp'] for item in sim_data])
elif isinstance(sim_data, dict):
    sim_obs = sim_data['observations']
    sim_ts = sim_data['timestamps']

# Extract relevant parts (ignore IMU since robot was held)
# Joint positions: [6:20]
# Joint velocities: [20:34]
# Actions: [34:48]

real_pos = real_obs[:, 6:20]
real_vel = real_obs[:, 20:34]
real_actions = real_obs[:, 34:48]

sim_pos = sim_obs[:, 6:20]
sim_vel = sim_obs[:, 20:34]
sim_actions = sim_obs[:, 34:48]

# Verify actions match (they should!)
print("="*60)
print("ACTION VERIFICATION")
print("="*60)
print(f"Actions match: {np.allclose(real_actions, sim_actions, atol=1e-6)}")
print(f"Max action difference: {np.max(np.abs(real_actions - sim_actions)):.8f}")

# Analyze position response to actions
print("\n" + "="*60)
print("POSITION RESPONSE ANALYSIS")
print("="*60)

# For each joint, compute the ratio of position change to action magnitude
joint_names = [
    'L_hip_yaw', 'L_hip_roll', 'L_hip_pitch', 'L_knee', 'L_ankle',
    'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
    'R_hip_yaw', 'R_hip_roll', 'R_hip_pitch', 'R_knee', 'R_ankle'
]

print("\nRMS joint position (relative to default):")
print(f"{'Joint':<15} {'Real RMS':>10} {'Sim RMS':>10} {'Ratio':>10}")
print("-"*60)
for i, name in enumerate(joint_names):
    real_rms = np.sqrt(np.mean(real_pos[:, i]**2))
    sim_rms = np.sqrt(np.mean(sim_pos[:, i]**2))
    ratio = real_rms / sim_rms if sim_rms > 1e-6 else np.inf
    print(f"{name:<15} {real_rms:>10.4f} {sim_rms:>10.4f} {ratio:>10.2f}")

print("\nRMS joint velocity:")
print(f"{'Joint':<15} {'Real RMS':>10} {'Sim RMS':>10} {'Ratio':>10}")
print("-"*60)
for i, name in enumerate(joint_names):
    real_rms = np.sqrt(np.mean(real_vel[:, i]**2))
    sim_rms = np.sqrt(np.mean(sim_vel[:, i]**2))
    ratio = real_rms / sim_rms if sim_rms > 1e-6 else np.inf
    print(f"{name:<15} {real_rms:>10.4f} {sim_rms:>10.4f} {ratio:>10.2f}")

# Estimate effective gain by comparing position response
print("\n" + "="*60)
print("EFFECTIVE GAIN ESTIMATION")
print("="*60)
print("\nAssuming sim needs higher kp, estimate the ratio:")

# For each joint, compute correlation between action and resulting position
print(f"\n{'Joint':<15} {'Pos/Action Real':>15} {'Pos/Action Sim':>15} {'Gain Ratio':>12}")
print("-"*75)
for i, name in enumerate(joint_names):
    # Skip if no significant action
    if np.std(real_actions[:, i]) < 0.01:
        continue

    # Compute mean absolute position response per unit action
    real_response = np.mean(np.abs(real_pos[:, i])) / (np.mean(np.abs(real_actions[:, i])) + 1e-9)
    sim_response = np.mean(np.abs(sim_pos[:, i])) / (np.mean(np.abs(sim_actions[:, i])) + 1e-9)
    gain_ratio = real_response / sim_response if sim_response > 1e-6 else np.inf

    print(f"{name:<15} {real_response:>15.4f} {sim_response:>15.4f} {gain_ratio:>12.2f}")

# Overall recommendation
all_ratios = []
for i in range(14):
    real_rms = np.sqrt(np.mean(real_pos[:, i]**2))
    sim_rms = np.sqrt(np.mean(sim_pos[:, i]**2))
    if sim_rms > 1e-6:
        all_ratios.append(real_rms / sim_rms)

if all_ratios:
    median_ratio = np.median(all_ratios)
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print(f"Median position response ratio (real/sim): {median_ratio:.2f}")
    print(f"This suggests sim needs kp increased by factor of ~{median_ratio:.2f}")
    print(f"If using kp=0.28 (firmware kp=100), try kp={0.28*median_ratio:.2f}")
    print(f"If using kp=0.57 (firmware kp=200), try kp={0.57*median_ratio:.2f}")
