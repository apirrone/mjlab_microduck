# BeyondMimic Tracking for MicroDuck

This guide explains how to use the BeyondMimic implementation from mjlab with your MicroDuck reference motions.

## Overview

**BeyondMimic vs. Current Imitation Learning:**

| Feature | Current Imitation | BeyondMimic |
|---------|------------------|-------------|
| Motion Format | Polynomial coefficients | Sampled trajectories (.npz) |
| Body Tracking | Joint positions only | Full body poses + velocities |
| Curriculum | Fixed reference motion | Adaptive sampling from hard frames |
| Reference Motion | Evaluated at continuous phase | Sampled discrete timesteps |
| Implementation | Custom reward function | mjlab's tracking task |

**When to use BeyondMimic:**
- You want full body tracking (not just joints)
- You want adaptive curriculum learning (focuses on hard parts of motion)
- You want to leverage mjlab's BeyondMimic implementation
- You want to track multiple bodies (e.g., trunk, feet)

**When to use Current Imitation:**
- You prefer the paper's reward structure exactly
- You want continuous phase-based motion evaluation
- Simpler setup (no conversion needed)

## Quick Start: Holonomous Walking (Multiple Motions)

For training a full omnidirectional walk that covers all velocity commands:

```bash
# 1. Convert all reference motions
uv run python convert_to_beyondmimic.py ./src/mjlab_microduck/data/reference_motions.pkl \
    --output-dir ./beyondmimic_motions

# 2. Concatenate motions covering holonomous space
uv run python concatenate_motions.py ./beyondmimic_motions \
    --output holonomous_walk.npz \
    --strategy edges

# 3. Upload motion to wandb as an artifact
uv run uv run python upload_motion_artifact.py holonomous_walk.npz \
    --artifact-name microduck-holonomous-walk \
    --project mjlab_microduck

# 4. Train using the wandb artifact
#    The script will print the full artifact path (entity/project/name:latest)
uv run train Mjlab-BeyondMimic-MicroDuck \
    --env.scene.num-envs 2048 \
    --registry-name <entity>/mjlab_microduck/microduck-holonomous-walk:latest
```

**Note:** BeyondMimic tasks require motion files to be stored as wandb artifacts. The training script downloads the motion from wandb, ensuring versioning and reproducibility.

**Concatenation Strategies:**
- `edges` (default): Selects motions at velocity extremes (forward, back, left, right, rotate) - ~7 motions
- `grid`: Samples a grid covering velocity space - ~20-50 motions
- `all`: Uses all available motions - can be 100+ motions

## Important: Registry Name

BeyondMimic tasks require a `--registry-name` parameter when training. This is used for:
- Wandb artifact tracking and versioning
- Linking trained models to their motion data
- Organizing different motion configurations

**Examples:**
- `--registry-name microduck-holonomous-walk` for omnidirectional walking
- `--registry-name microduck-forward-only` for forward-only motion
- `--registry-name microduck-v1` for versioning

You can use any descriptive name - it's just a label for organizing your experiments.

## Setup

### Step 1: Convert Reference Motions

First, convert your polynomial reference motions to BeyondMimic format:

```bash
uv run python convert_to_beyondmimic.py ./src/mjlab_microduck/data/reference_motions.pkl \
    --output-dir ./beyondmimic_motions \
    --fps 50 \
    --body-names trunk_base
```

**Options:**
- `--output-dir`: Where to save .npz files (default: `./beyondmimic_motions`)
- `--fps`: Sampling rate in Hz (default: 50, matches training frequency)
- `--body-names`: Bodies to track (default: `trunk_base`, can add more like `left_foot right_foot`)
- `--motion-keys`: Convert specific motions only (default: all)

This will create one .npz file per motion, e.g.:
```
beyondmimic_motions/
  0.01_0.0_-0.4.npz
  0.01_0.0_0.0.npz
  0.05_0.0_0.0.npz
  ...
```

### Step 2: Upload Motion to Wandb

BeyondMimic tasks require motion files to be stored as wandb artifacts:

```bash
uv run python upload_motion_artifact.py ./beyondmimic_motions/0.01_0.0_-0.4.npz \
    --artifact-name microduck-forward-walk \
    --project mjlab_microduck \
    --entity <your-wandb-entity>  # Optional, uses default if omitted
```

The script will output the full artifact path to use for training.

### Step 3: Train with BeyondMimic

Use the artifact path from the upload step:

```bash
uv run train Mjlab-BeyondMimic-MicroDuck \
    --env.scene.num-envs 2048 \
    --registry-name <entity>/mjlab_microduck/microduck-forward-walk:latest
```

**Example:** If your wandb entity is `pollen-robotics`:
```bash
uv run train Mjlab-BeyondMimic-MicroDuck \
    --env.scene.num-envs 2048 \
    --registry-name pollen-robotics/mjlab_microduck/microduck-forward-walk:latest
```

**Note:** To train on multiple motions for holonomous walking, concatenate them first (see section below).

### Step 4: Play/Inference

```bash
uv run play Mjlab-BeyondMimic-MicroDuck \
    --wandb-run-path <your-run-path> \
    --registry-name <entity>/mjlab_microduck/microduck-forward-walk:latest
```

**Note:** The motion file is automatically downloaded from the artifact during inference.

## Motion File Format

BeyondMimic expects .npz files with:

```python
{
    'joint_pos':       (time_steps, num_joints)        # Joint positions
    'joint_vel':       (time_steps, num_joints)        # Joint velocities
    'body_pos_w':      (time_steps, num_bodies, 3)    # Body positions (world frame)
    'body_quat_w':     (time_steps, num_bodies, 4)    # Body orientations (quaternions)
    'body_lin_vel_w':  (time_steps, num_bodies, 3)    # Body linear velocities
    'body_ang_vel_w':  (time_steps, num_bodies, 3)    # Body angular velocities
}
```

For MicroDuck:
- `num_joints = 14` (5 per leg + 4 for neck/head)
- `num_bodies` depends on `--body-names` (default: 1 for trunk_base)
- `time_steps = period * fps` (e.g., 0.72s * 50Hz = 36 timesteps)

## Adaptive Sampling

BeyondMimic's key feature is **adaptive sampling** - it learns which parts of the motion are hardest and samples those more frequently during training.

**Sampling Modes:**
- `adaptive` (default): Learns difficulty distribution, focuses on hard frames
- `uniform`: Random sampling across all frames
- `start`: Always starts from beginning of motion

**Adaptive Parameters:**
```python
cfg.commands["motion"].sampling_mode = "adaptive"
cfg.commands["motion"].adaptive_kernel_size = 5        # Smoothing window
cfg.commands["motion"].adaptive_lambda = 0.8           # Decay factor
cfg.commands["motion"].adaptive_uniform_ratio = 0.1    # Uniform sampling ratio
cfg.commands["motion"].adaptive_alpha = 0.001          # Learning rate
```

The agent automatically tracks which frames are hardest (high tracking error) and samples those more often.

## Rewards

BeyondMimic uses **exponential rewards** for smooth gradients:

```python
reward = exp(-error^2 / (2 * std^2))
```

**Reward Components:**
- `motion_global_root_pos` (0.5): Global anchor position tracking
- `motion_global_root_ori` (0.5): Global anchor orientation tracking
- `motion_body_pos` (1.0): Relative body position tracking
- `motion_body_ori` (1.0): Relative body orientation tracking
- `motion_body_lin_vel` (1.0): Body linear velocity tracking
- `motion_body_ang_vel` (1.0): Body angular velocity tracking
- `action_rate_l2` (-0.1): Action smoothness
- `joint_limit` (-10.0): Joint limit penalty
- `self_collisions` (-10.0): Self collision penalty

**Tuning std parameters:**
Larger `std` = more forgiving reward, smaller `std` = stricter tracking.

## Tracking Multiple Bodies

To track additional bodies (e.g., feet for better contact tracking):

```bash
uv run python convert_to_beyondmimic.py ./src/mjlab_microduck/data/reference_motions.pkl \
    --body-names trunk_base left_foot right_foot
```

Then update the config:
```python
cfg.commands["motion"].body_names = ("trunk_base", "left_foot", "right_foot")
```

This will track all three bodies' poses and velocities.

## Holonomous Walking (Training on Multiple Motions)

To train a policy that can walk in all directions (omnidirectional/holonomous), you need to train on multiple reference motions simultaneously.

### Why Concatenate Motions?

A single reference motion only covers one velocity command (e.g., "walk forward at 0.2 m/s"). For holonomous walking, you need:
- Forward and backward motion
- Left and right strafing
- Rotation left and right
- Combinations of these

BeyondMimic trains on one .npz file at a time, so we concatenate multiple motions into a single file.

### Method 1: Edge Motions (Recommended)

Select motions at the extremes of your velocity space:

```bash
uv run python concatenate_motions.py ./beyondmimic_motions \
    --output holonomous_walk.npz \
    --strategy edges
```

This selects ~7 motions:
- Standing (zero velocity)
- Maximum forward
- Maximum backward
- Maximum left strafe
- Maximum right strafe
- Maximum rotate left
- Maximum rotate right

**Pros:** Fast training, covers main directions
**Cons:** May not generalize well to intermediate velocities

### Method 2: Grid Sampling

Sample a regular grid across velocity space:

```bash
uv run python concatenate_motions.py ./beyondmimic_motions \
    --output holonomous_walk.npz \
    --strategy grid
```

This selects ~20-50 motions covering combinations of dx, dy, dtheta.

**Pros:** Better generalization to all velocities
**Cons:** Longer training time, larger motion file

### Method 3: All Motions

Use every available reference motion:

```bash
uv run python concatenate_motions.py ./beyondmimic_motions \
    --output holonomous_walk.npz \
    --strategy all
```

**Pros:** Most comprehensive coverage
**Cons:** Very long training time, may be redundant

### Method 4: Manual Selection

Select specific motions manually:

```bash
uv run python concatenate_motions.py ./beyondmimic_motions \
    --output my_custom_walk.npz \
    --motion-files 0.0_0.0_0.0.npz 0.2_0.0_0.0.npz -0.2_0.0_0.0.npz \
                   0.0_0.2_0.0.npz 0.0_-0.2_0.0.npz \
                   0.0_0.0_0.5.npz 0.0_0.0_-0.5.npz
```

### Training Tips for Holonomous Walking

1. **Start with edges**: Begin training with edge motions for faster iteration
2. **Monitor metrics**: Watch the adaptive sampling distribution to see which motions are hardest
3. **Increase training time**: More motions = more variation = longer convergence
4. **Adjust sampling**: Try `adaptive_uniform_ratio=0.2` for more exploration
5. **Test generalization**: After training, test intermediate velocities not in training set

### Example: Full Holonomous Training

```bash
# Convert all reference motions
uv run python convert_to_beyondmimic.py ./src/mjlab_microduck/data/reference_motions.pkl \
    --output-dir ./beyondmimic_motions \
    --fps 50 \
    --body-names trunk_base

# Create holonomous walk with edge strategy
uv run python concatenate_motions.py ./beyondmimic_motions \
    --output holonomous_walk.npz \
    --strategy edges

# Upload to wandb
uv run python upload_motion_artifact.py holonomous_walk.npz \
    --artifact-name microduck-holonomous-walk \
    --project mjlab_microduck \
    --entity <your-entity>  # Optional

# Train (use the artifact path from upload script output)
uv run train Mjlab-BeyondMimic-MicroDuck \
    --env.scene.num-envs 2048 \
    --registry-name <entity>/mjlab_microduck/microduck-holonomous-walk:latest \
    --agent.max_iterations 100000  # May need more iterations for multiple motions

# Test
uv run play Mjlab-BeyondMimic-MicroDuck \
    --wandb-run-path <your-run> \
    --registry-name <entity>/mjlab_microduck/microduck-holonomous-walk:latest
```

### Adaptive Sampling with Multiple Motions

BeyondMimic's adaptive sampling works great with concatenated motions:
- It tracks which **frames** are hard (not which motions)
- Hard frames from any motion get sampled more frequently
- Over time, it balances difficulty across all included motions
- You can see this in metrics: `sampling_entropy`, `sampling_top1_prob`

This means the agent naturally focuses on the hardest parts of your entire holonomous walking repertoire!

## Troubleshooting

**Error: "Must provide --registry-name for tracking tasks"**
- Tracking/BeyondMimic tasks require a `--registry-name` parameter pointing to a wandb artifact
- Format: `<entity>/<project>/<artifact-name>:version`
- Example: `pollen-robotics/mjlab_microduck/microduck-holonomous-walk:latest`
- You must upload your motion file to wandb first using `upload_motion_artifact.py`

**Error: "project 'uncategorized' not found" or artifact not found**
- The motion file hasn't been uploaded to wandb yet
- Use `uv run python upload_motion_artifact.py` to upload your .npz file first
- Make sure the artifact path matches exactly (entity/project/name:version)

**Error: "Body 'xxx' not found in model"**
- Check body names in `src/mjlab_microduck/robot/microduck/scene.xml`
- Use `mj_name2id` compatible names

**Training unstable / agent falls immediately**
- Check that motion file path is correct
- Verify .npz file has correct arrays (use `np.load()` to inspect)
- Try reducing velocity ranges in config
- Increase termination thresholds

**Agent doesn't track motion well**
- Adjust reward std parameters (make them larger for more forgiving rewards)
- Check adaptive sampling is enabled
- Verify forward kinematics during conversion is correct

**How to train on multiple motions?**
Use the `concatenate_motions.py` script to combine multiple motions:
```bash
uv run python concatenate_motions.py ./beyondmimic_motions --output holonomous_walk.npz --strategy edges
```
See the "Holonomous Walking" section above for details.

## Comparison with Current Imitation

**Advantages of BeyondMimic:**
- Adaptive curriculum (focuses on hard parts automatically)
- Full body tracking (not just joints)
- Proven implementation from research
- Can track complex multi-body systems

**Advantages of Current Imitation:**
- Simpler setup (no conversion needed)
- Continuous phase-based evaluation (smooth transitions)
- Matches paper's reward structure exactly
- Works well for periodic gaits

**Recommendation:** Start with current imitation for basic locomotion. Use BeyondMimic for:
- Complex non-periodic motions
- When you need to track multiple bodies
- When you want adaptive curriculum learning
