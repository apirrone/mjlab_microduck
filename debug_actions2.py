#!/usr/bin/env python3
"""Debug action matching for second dataset."""

import pickle
import numpy as np

# Load both datasets
with open('data_hang2.pkl', 'rb') as f:
    real_data = pickle.load(f)
with open('sim_observations2.pkl', 'rb') as f:
    sim_data = pickle.load(f)

# Extract observations
if isinstance(real_data, list):
    real_obs = np.array([item['observation'] for item in real_data])
elif isinstance(real_data, dict):
    real_obs = real_data['observations']

if isinstance(sim_data, list):
    sim_obs = np.array([item['observation'] for item in sim_data])
elif isinstance(sim_data, dict):
    sim_obs = sim_data['observations']

# Extract actions
real_actions = real_obs[:, 34:48]
sim_actions = sim_obs[:, 34:48]

# Extract joint positions
real_pos = real_obs[:, 6:20]
sim_pos = sim_obs[:, 6:20]

print("="*60)
print("ACTION VERIFICATION (data_hang2)")
print("="*60)
print(f"Real obs shape: {real_obs.shape}")
print(f"Sim obs shape:  {sim_obs.shape}")
print(f"\nActions match: {np.allclose(real_actions, sim_actions, atol=1e-6)}")
print(f"Max action difference: {np.max(np.abs(real_actions - sim_actions)):.8f}")

print("\n" + "="*60)
print("FIRST 3 ACTIONS COMPARISON")
print("="*60)
for i in range(min(3, len(real_actions))):
    print(f"\nStep {i}:")
    print(f"  Real action: {real_actions[i]}")
    print(f"  Sim action:  {sim_actions[i]}")
    print(f"  Difference:  {real_actions[i] - sim_actions[i]}")

print("\n" + "="*60)
print("JOINT POSITION STATISTICS")
print("="*60)
print(f"{'Joint':<15} {'Real Mean':>10} {'Sim Mean':>10} {'Real Std':>10} {'Sim Std':>10}")
print("-"*60)

joint_names = [
    'L_hip_yaw', 'L_hip_roll', 'L_hip_pitch', 'L_knee', 'L_ankle',
    'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
    'R_hip_yaw', 'R_hip_roll', 'R_hip_pitch', 'R_knee', 'R_ankle'
]

for i, name in enumerate(joint_names):
    real_mean = np.mean(real_pos[:, i])
    sim_mean = np.mean(sim_pos[:, i])
    real_std = np.std(real_pos[:, i])
    sim_std = np.std(sim_pos[:, i])
    print(f"{name:<15} {real_mean:>10.4f} {sim_mean:>10.4f} {real_std:>10.4f} {sim_std:>10.4f}")

# Check if there's a systematic offset
print("\n" + "="*60)
print("POSITION OFFSET ANALYSIS")
print("="*60)
position_diff = real_pos - sim_pos
print(f"Mean position offset per joint:")
for i, name in enumerate(joint_names):
    mean_diff = np.mean(position_diff[:, i])
    print(f"  {name:<15}: {mean_diff:>8.4f} rad")

# Check correlation between actions and positions
print("\n" + "="*60)
print("ACTION MAGNITUDE vs POSITION RESPONSE")
print("="*60)
print(f"Real action RMS: {np.sqrt(np.mean(real_actions**2)):.4f}")
print(f"Sim action RMS:  {np.sqrt(np.mean(sim_actions**2)):.4f}")
print(f"Real position RMS: {np.sqrt(np.mean(real_pos**2)):.4f}")
print(f"Sim position RMS:  {np.sqrt(np.mean(sim_pos**2)):.4f}")
print(f"\nPosition response ratio (Real/Sim): {np.sqrt(np.mean(real_pos**2)) / np.sqrt(np.mean(sim_pos**2)):.2f}")
