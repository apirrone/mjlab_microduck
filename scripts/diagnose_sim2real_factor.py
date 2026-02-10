#!/usr/bin/env python3
"""
Diagnostic script to find the factor-of-2 discrepancy between sim and real.
Compares observations, actions, and control gains.
"""

import argparse
import pickle
import numpy as np


def analyze_knee_data(real_pkl, sim_pkl):
    """Analyze knee sine wave data to check for systematic differences."""

    with open(real_pkl, 'rb') as f:
        real_data = pickle.load(f)
    with open(sim_pkl, 'rb') as f:
        sim_data = pickle.load(f)

    # Extract data (format: [timestamp, position, last_action])
    real_ts = np.array([d[0] for d in real_data])
    real_pos = np.array([d[1] for d in real_data])
    real_act = np.array([d[2] for d in real_data])

    sim_ts = np.array([d[0] for d in sim_data])
    sim_pos = np.array([d[1] for d in sim_data])
    sim_act = np.array([d[2] for d in sim_data])

    print("="*80)
    print("KNEE SINE WAVE ANALYSIS")
    print("="*80)

    # Action statistics (should be identical - both use same sine formula)
    print("\n1. ACTION COMPARISON (should be identical):")
    print(f"  Real action range: [{real_act.min():.4f}, {real_act.max():.4f}]")
    print(f"  Sim action range:  [{sim_act.min():.4f}, {sim_act.max():.4f}]")
    print(f"  Action amplitude ratio (sim/real): {sim_act.std() / real_act.std():.4f}")

    # Position response comparison
    print("\n2. POSITION RESPONSE (shows system dynamics):")
    print(f"  Real position range: [{real_pos.min():.4f}, {real_pos.max():.4f}]")
    print(f"  Sim position range:  [{sim_pos.min():.4f}, {sim_pos.max():.4f}]")
    print(f"  Position amplitude ratio (sim/real): {sim_pos.std() / real_pos.std():.4f}")

    # Phase lag analysis (indicates delay/damping differences)
    print("\n3. PHASE LAG ANALYSIS:")

    # Find peak indices for action and position
    real_act_peaks = find_peaks(real_act)
    real_pos_peaks = find_peaks(real_pos)
    sim_act_peaks = find_peaks(sim_act)
    sim_pos_peaks = find_peaks(sim_pos)

    if len(real_act_peaks) > 0 and len(real_pos_peaks) > 0:
        real_lag = np.mean(real_ts[real_pos_peaks] - real_ts[real_act_peaks[:len(real_pos_peaks)]])
        print(f"  Real phase lag: {real_lag*1000:.1f} ms ({real_lag*50:.1f} timesteps @ 50Hz)")

    if len(sim_act_peaks) > 0 and len(sim_pos_peaks) > 0:
        sim_lag = np.mean(sim_ts[sim_pos_peaks] - sim_ts[sim_act_peaks[:len(sim_pos_peaks)]])
        print(f"  Sim phase lag:  {sim_lag*1000:.1f} ms ({sim_lag*50:.1f} timesteps @ 50Hz)")

    # Transfer function gain (position_amplitude / action_amplitude)
    real_gain = sim_pos.std() / real_act.std()
    sim_gain = sim_pos.std() / sim_act.std()

    print("\n4. TRANSFER FUNCTION GAIN (position/action):")
    print(f"  Real: {real_gain:.4f}")
    print(f"  Sim:  {sim_gain:.4f}")
    print(f"  Ratio (sim/real): {sim_gain / real_gain:.4f}")

    if abs(sim_gain / real_gain - 2.0) < 0.2:
        print("\n  ⚠️  WARNING: Sim gain is ~2x real gain!")
        print("  This could explain the factor-of-2 issue.")

    return real_gain, sim_gain


def find_peaks(signal, threshold_percentile=75):
    """Find peaks in signal using simple threshold."""
    threshold = np.percentile(signal, threshold_percentile)
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            peaks.append(i)
    return np.array(peaks)


def check_observation_normalization():
    """Check if there's any observation normalization that could affect action scaling."""
    print("\n" + "="*80)
    print("OBSERVATION CONFIGURATION CHECK")
    print("="*80)

    try:
        from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg, MicroduckRlCfg

        cfg = make_microduck_velocity_env_cfg(play=False)

        print("\n1. ACTION SCALE:")
        print(f"  Training action scale: {cfg.actions['joint_pos'].scale}")

        print("\n2. OBSERVATION NORMALIZATION:")
        print(f"  Actor obs normalization: {MicroduckRlCfg.policy.actor_obs_normalization}")
        print(f"  Critic obs normalization: {MicroduckRlCfg.policy.critic_obs_normalization}")

        if MicroduckRlCfg.policy.actor_obs_normalization:
            print("\n  ⚠️  WARNING: Observation normalization is enabled!")
            print("  This could cause action magnitude differences if not applied in deployment.")

        print("\n3. ACTUATOR DELAY:")
        from mjlab_microduck.robot.microduck_constants import actuators
        print(f"  Delay range: {actuators.delay_min_lag}-{actuators.delay_max_lag} timesteps")
        print(f"  At 50Hz, this is {actuators.delay_min_lag*20}-{actuators.delay_max_lag*20} ms")

    except Exception as e:
        print(f"  Could not load config: {e}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose sim2real factor-of-2 discrepancy")
    parser.add_argument("--knee-real", type=str, default="knee_sin.pkl",
                       help="Real knee data pickle")
    parser.add_argument("--knee-sim", type=str, default="knee_sin_sim.pkl",
                       help="Sim knee data pickle")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("SIM2REAL FACTOR-OF-2 DIAGNOSTIC")
    print("="*80)

    # Analyze knee data if files exist
    import os
    if os.path.exists(args.knee_real) and os.path.exists(args.knee_sim):
        real_gain, sim_gain = analyze_knee_data(args.knee_real, args.knee_sim)
    else:
        print(f"\nKnee data files not found. Skipping sine wave analysis.")
        print(f"  Real: {args.knee_real}")
        print(f"  Sim:  {args.knee_sim}")

    # Check configuration
    check_observation_normalization()

    print("\n" + "="*80)
    print("POTENTIAL ISSUES TO CHECK:")
    print("="*80)
    print("""
1. **Action Bounds**: Are policy actions clipped during training?
   - Check: rsl_rl clip_actions parameter
   - If actions are clipped to [-1, 1] in training but not in deployment

2. **Observation Scaling**: Are observations normalized?
   - Check: actor_obs_normalization in MicroduckRlCfg
   - If enabled, you need to apply same normalization in deployment

3. **Action Delay**: Does real robot have more delay than sim?
   - Sim: 2-5 timesteps (40-100ms @ 50Hz)
   - Real: Unknown - measure this!

4. **Motor Calibration**: Double-check BAM identification
   - Verify voltage during identification matches deployment
   - Check if max_pwm, error_gain are correct

5. **Control Frequency Mismatch**: Verify both run at exactly 50Hz
   - Real: Check actual timing
   - Sim: Check decimation settings
    """)

    print("\n" + "="*80)
    print("RECOMMENDED NEXT STEPS:")
    print("="*80)
    print("""
1. Run knee sine tests at DIFFERENT frequencies (0.5Hz, 1Hz, 2Hz)
   - If gain ratio changes with frequency → delay mismatch
   - If gain ratio constant → systematic scaling error

2. Record a policy rollout in sim and save observations
   - Compare observation magnitudes to what policy expects

3. Add logging to main.rs to record observation statistics
   - Compare mean, std of observations during real deployment

4. Check if policy action distribution has changed
   - Plot histogram of policy outputs in sim vs real
    """)


if __name__ == "__main__":
    main()
