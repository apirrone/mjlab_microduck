#!/usr/bin/env python3
"""Quick test to identify which randomization causes instability.

This script creates a minimal environment and tests each randomization
to see if it produces valid observations.
"""

import sys
import torch

sys.path.insert(0, 'src')

def test_randomization(name: str, enable_flags: dict) -> bool:
    """Test a single randomization configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    # Temporarily modify the config
    import mjlab_microduck.tasks.microduck_velocity_env_cfg as cfg_module

    # Save original values
    original_flags = {}
    for flag_name in enable_flags:
        original_flags[flag_name] = getattr(cfg_module, flag_name)

    # Set test values
    for flag_name, value in enable_flags.items():
        setattr(cfg_module, flag_name, value)

    try:
        # Reload and create environment
        from importlib import reload
        reload(cfg_module)

        from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
        cfg = make_microduck_velocity_env_cfg()

        # Import environment after setting flags
        from mjlab.envs import ManagerBasedRlEnv

        # Create minimal environment
        cfg.scene.num_envs = 4  # Small number for quick test
        env = ManagerBasedRlEnv(cfg)

        # Run a few steps
        print("  Running 10 resets and 50 steps...")
        for reset_iter in range(10):
            obs, _ = env.reset()

            # Check for NaN/Inf in observations
            if torch.isnan(obs).any() or torch.isinf(obs).any():
                print(f"  ❌ FAILED: Invalid observations after reset {reset_iter}")
                return False

            for step in range(50):
                # Random actions
                actions = torch.randn_like(env.action_manager.action)
                obs, _, _, _, _ = env.step(actions)

                # Check for NaN/Inf
                if torch.isnan(obs).any() or torch.isinf(obs).any():
                    print(f"  ❌ FAILED: Invalid observations at reset {reset_iter}, step {step}")
                    return False

        print(f"  ✓ PASSED: No instability detected")
        env.close()
        return True

    except Exception as e:
        print(f"  ❌ FAILED: {type(e).__name__}: {e}")
        return False
    finally:
        # Restore original values
        for flag_name, value in original_flags.items():
            setattr(cfg_module, flag_name, value)


def main():
    print("🔍 DOMAIN RANDOMIZATION STABILITY TEST")
    print("Testing each randomization individually...\n")

    base_config = {
        'ENABLE_COM_RANDOMIZATION': True,
        'ENABLE_KP_RANDOMIZATION': True,
        'ENABLE_KD_RANDOMIZATION': False,
        'ENABLE_MASS_RANDOMIZATION': False,
        'ENABLE_INERTIA_RANDOMIZATION': False,
        'ENABLE_JOINT_FRICTION_RANDOMIZATION': False,
        'ENABLE_JOINT_DAMPING_RANDOMIZATION': False,
        'ENABLE_EXTERNAL_FORCE_DISTURBANCES': True,
    }

    # Test baseline (known working)
    print("\n" + "="*60)
    print("BASELINE TEST (CoM + Kp + External Forces)")
    print("="*60)
    if not test_randomization("Baseline", base_config):
        print("\n❌ BASELINE FAILED - Something is wrong with the base config!")
        return 1

    # Test each additional randomization
    tests = [
        ("Motor Kd", {'ENABLE_KD_RANDOMIZATION': True}),
        ("Body Mass", {'ENABLE_MASS_RANDOMIZATION': True}),
        ("Body Inertia", {'ENABLE_INERTIA_RANDOMIZATION': True}),
        ("Joint Friction", {'ENABLE_JOINT_FRICTION_RANDOMIZATION': True}),
        ("Joint Damping", {'ENABLE_JOINT_DAMPING_RANDOMIZATION': True}),
    ]

    results = {}
    for test_name, flags in tests:
        test_config = base_config.copy()
        test_config.update(flags)
        results[test_name] = test_randomization(test_name, test_config)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")

    failed = [name for name, passed in results.items() if not passed]
    if failed:
        print(f"\n⚠️  Problematic randomizations: {', '.join(failed)}")
        print("   These should be kept disabled or ranges should be reduced.")
        return 1
    else:
        print("\n✓ All randomizations passed! Safe to enable all.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
