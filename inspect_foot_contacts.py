#!/usr/bin/env python3
"""Inspect foot contacts data in reference motion file."""

import pickle
from pathlib import Path
import numpy as np

# Try to import matplotlib, but continue without it if not available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, skipping visualization")

# Load reference motion file
motion_file = Path("src/mjlab_microduck/data/reference_motion.pkl")
print(f"Loading motion file: {motion_file}")

with open(motion_file, 'rb') as f:
    data = pickle.load(f)

print(f"\nMotion file structure:")
print(f"  Type: {type(data)}")
if isinstance(data, dict):
    print(f"  Keys: {list(data.keys())}")

    # Check if it's the velocity-indexed format
    if 'velocity_points' in data:
        print(f"\n=== Velocity-indexed motion library ===")
        velocity_points = data['velocity_points']
        motion_data_list = data['motion_data_list']
        print(f"Number of motions: {len(motion_data_list)}")
        print(f"Velocity points shape: {np.array(velocity_points).shape}")
        print(f"First few velocity points (dx, dy, dtheta):")
        for i, vel in enumerate(velocity_points[:5]):
            print(f"  Motion {i}: {vel}")

        # Analyze first motion
        first_motion = np.array(motion_data_list[0])
        print(f"\nFirst motion data shape: {first_motion.shape}")
        print(f"  Number of frames: {first_motion.shape[0]}")
        print(f"  Frame size: {first_motion.shape[1]}")

        # Check if there's slice info
        if 'slices' in data:
            slices = data['slices']
            print(f"\nData slices in each frame:")
            for key, slice_obj in slices.items():
                size = slice_obj.stop - slice_obj.start
                print(f"  {key}: [{slice_obj.start}:{slice_obj.stop}] (size: {size})")

            # Extract foot contacts if available
            if 'foot_contacts' in slices:
                foot_slice = slices['foot_contacts']
                foot_contacts = first_motion[:, foot_slice]

                print(f"\n=== Foot Contacts Analysis ===")
                print(f"Foot contacts shape: {foot_contacts.shape}")
                print(f"  Frames: {foot_contacts.shape[0]}")
                print(f"  Number of feet: {foot_contacts.shape[1]}")

                print(f"\nFoot contacts statistics:")
                print(f"  Min value: {foot_contacts.min():.3f}")
                print(f"  Max value: {foot_contacts.max():.3f}")
                print(f"  Mean value: {foot_contacts.mean():.3f}")
                print(f"  Unique values: {np.unique(foot_contacts)}")

                print(f"\nFirst 20 frames of foot contacts:")
                print("  Frame | Left Foot | Right Foot")
                print("  ------|-----------|------------")
                for i in range(min(20, len(foot_contacts))):
                    left = foot_contacts[i, 0]
                    right = foot_contacts[i, 1] if foot_contacts.shape[1] > 1 else 0
                    left_str = "■" if left > 0.5 else "□"
                    right_str = "■" if right > 0.5 else "□"
                    print(f"  {i:5d} | {left:9.2f} {left_str} | {right:10.2f} {right_str}")

                # Plot foot contacts for multiple motions
                if HAS_MATPLOTLIB:
                    print(f"\nGenerating foot contacts visualization...")
                    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
                    fig.suptitle('Foot Contacts Across Different Velocities', fontsize=14)

                    # Plot 6 different motions
                    for idx in range(min(6, len(motion_data_list))):
                        motion = np.array(motion_data_list[idx])
                        contacts = motion[:, foot_slice]
                        vel = velocity_points[idx]

                        ax = axes[idx // 2, idx % 2]
                        frames = np.arange(len(contacts))

                        if contacts.shape[1] >= 2:
                            ax.plot(frames, contacts[:, 0], 'b-', label='Left Foot', linewidth=2)
                            ax.plot(frames, contacts[:, 1], 'r-', label='Right Foot', linewidth=2)
                        else:
                            ax.plot(frames, contacts[:, 0], 'b-', label='Contact', linewidth=2)

                        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
                        ax.set_xlabel('Frame')
                        ax.set_ylabel('Contact Value')
                        ax.set_title(f'Motion {idx}: vel=({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})')
                        ax.legend(loc='upper right')
                        ax.grid(True, alpha=0.3)
                        ax.set_ylim(-0.1, 1.1)

                    plt.tight_layout()
                    output_file = "foot_contacts_visualization.png"
                    plt.savefig(output_file, dpi=150)
                    print(f"Saved visualization to: {output_file}")

                # Count contact patterns
                print(f"\nContact patterns analysis:")
                for idx in range(min(10, len(motion_data_list))):
                    motion = np.array(motion_data_list[idx])
                    contacts = motion[:, foot_slice]
                    vel = velocity_points[idx]

                    # Count frames where each foot is in contact (> 0.5)
                    left_contact_frames = np.sum(contacts[:, 0] > 0.5)
                    right_contact_frames = np.sum(contacts[:, 1] > 0.5) if contacts.shape[1] > 1 else 0
                    both_contact_frames = np.sum((contacts[:, 0] > 0.5) & (contacts[:, 1] > 0.5)) if contacts.shape[1] > 1 else 0

                    print(f"  Motion {idx} (vel={vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}):")
                    print(f"    Left contact: {left_contact_frames}/{len(contacts)} frames ({100*left_contact_frames/len(contacts):.1f}%)")
                    if contacts.shape[1] > 1:
                        print(f"    Right contact: {right_contact_frames}/{len(contacts)} frames ({100*right_contact_frames/len(contacts):.1f}%)")
                        print(f"    Both feet: {both_contact_frames}/{len(contacts)} frames ({100*both_contact_frames/len(contacts):.1f}%)")
            else:
                print(f"\nWARNING: No 'foot_contacts' slice found in data!")
        else:
            print(f"\nWARNING: No 'slices' information found in data!")
            print(f"Available data keys: {data.keys()}")
    else:
        # Different format - velocity string keys
        print(f"\n=== Dictionary format (velocity string keys) ===")
        print(f"Number of motions: {len(data)}")
        print(f"First 10 keys: {list(data.keys())[:10]}")

        # Parse first motion
        first_key = list(data.keys())[0]
        first_motion_data = data[first_key]

        print(f"\nFirst motion key: {first_key}")
        print(f"First motion data type: {type(first_motion_data)}")
        print(f"First motion data keys: {list(first_motion_data.keys()) if isinstance(first_motion_data, dict) else 'N/A'}")

        if isinstance(first_motion_data, dict):
            # Check for foot contacts
            if 'foot_contacts' in first_motion_data:
                foot_contacts = np.array(first_motion_data['foot_contacts'])

                print(f"\n=== Foot Contacts Analysis ===")
                print(f"Foot contacts shape: {foot_contacts.shape}")
                print(f"  Frames: {foot_contacts.shape[0]}")
                if len(foot_contacts.shape) > 1:
                    print(f"  Number of feet: {foot_contacts.shape[1]}")

                print(f"\nFoot contacts statistics:")
                print(f"  Min value: {foot_contacts.min():.3f}")
                print(f"  Max value: {foot_contacts.max():.3f}")
                print(f"  Mean value: {foot_contacts.mean():.3f}")
                print(f"  Unique values: {np.unique(foot_contacts)}")
                print(f"  Data type: {foot_contacts.dtype}")

                print(f"\nFirst 30 frames of foot contacts:")
                print("  Frame | Left Foot | Right Foot")
                print("  ------|-----------|------------")
                for i in range(min(30, len(foot_contacts))):
                    if len(foot_contacts.shape) > 1 and foot_contacts.shape[1] >= 2:
                        left = foot_contacts[i, 0]
                        right = foot_contacts[i, 1]
                    else:
                        left = foot_contacts[i]
                        right = 0
                    left_str = "■" if left > 0.5 else "□"
                    right_str = "■" if right > 0.5 else "□"
                    print(f"  {i:5d} | {left:9.3f} {left_str} | {right:10.3f} {right_str}")

                # Sample a few different velocities
                print(f"\n=== Sample across different velocities ===")
                sample_keys = list(data.keys())[::len(data)//min(10, len(data))]

                for key in sample_keys[:10]:
                    motion_data = data[key]
                    if 'foot_contacts' in motion_data:
                        contacts = np.array(motion_data['foot_contacts'])

                        # Parse velocity from key
                        vel_parts = key.split('_')

                        # Count contact frames
                        if len(contacts.shape) > 1 and contacts.shape[1] >= 2:
                            left_contact = np.sum(contacts[:, 0] > 0.5)
                            right_contact = np.sum(contacts[:, 1] > 0.5)
                            both_contact = np.sum((contacts[:, 0] > 0.5) & (contacts[:, 1] > 0.5))

                            print(f"\n  Velocity key: {key}")
                            print(f"    Frames: {len(contacts)}")
                            print(f"    Left contact:  {left_contact}/{len(contacts)} ({100*left_contact/len(contacts):.1f}%)")
                            print(f"    Right contact: {right_contact}/{len(contacts)} ({100*right_contact/len(contacts):.1f}%)")
                            print(f"    Both feet:     {both_contact}/{len(contacts)} ({100*both_contact/len(contacts):.1f}%)")
                        else:
                            contact = np.sum(contacts > 0.5)
                            print(f"\n  Velocity key: {key}")
                            print(f"    Frames: {len(contacts)}")
                            print(f"    Contact: {contact}/{len(contacts)} ({100*contact/len(contacts):.1f}%)")
            else:
                print(f"\nWARNING: No 'foot_contacts' key found in motion data!")
                print(f"Available keys: {list(first_motion_data.keys())}")

print("\nDone!")
