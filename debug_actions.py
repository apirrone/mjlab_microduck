#!/usr/bin/env python3
"""Debug script to check action values in observations."""

import pickle
import numpy as np

# Load real observations
with open('data_hang.pkl', 'rb') as f:
    real_data = pickle.load(f)

# Load sim observations
with open('sim_observations.pkl', 'rb') as f:
    sim_data = pickle.load(f)

# Extract observations
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

# Extract actions (indices 34:48)
real_actions = real_obs[:, 34:48]
sim_actions = sim_obs[:, 34:48]

print("="*60)
print("REAL OBSERVATIONS")
print("="*60)
print(f"Shape: {real_obs.shape}")
print(f"Timestamps: {real_ts[:5]} ... {real_ts[-5:]}")
print(f"\nFirst 5 observations, action part [34:48]:")
for i in range(min(5, len(real_actions))):
    print(f"  [{i}] {real_actions[i]}")

print("\n" + "="*60)
print("SIM OBSERVATIONS")
print("="*60)
print(f"Shape: {sim_obs.shape}")
print(f"Timestamps: {sim_ts[:5]} ... {sim_ts[-5:]}")
print(f"\nFirst 5 observations, action part [34:48]:")
for i in range(min(5, len(sim_actions))):
    print(f"  [{i}] {sim_actions[i]}")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"Actions match: {np.allclose(real_actions, sim_actions, atol=1e-6)}")
print(f"Max difference: {np.max(np.abs(real_actions - sim_actions))}")
print(f"\nDifference in first 5:")
for i in range(min(5, len(real_actions))):
    diff = np.abs(real_actions[i] - sim_actions[i])
    print(f"  [{i}] max_diff={np.max(diff):.6f}")
