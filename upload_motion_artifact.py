#!/usr/bin/env python3
"""Upload a BeyondMimic motion file to wandb as an artifact."""

import argparse
import shutil
import sys
from pathlib import Path

try:
    import wandb
except ImportError:
    print("Error: wandb is not installed. Install it with: pip install wandb")
    sys.exit(1)


def upload_motion_artifact(
    motion_file: str,
    artifact_name: str,
    project: str = "mjlab_microduck",
    entity: str | None = None,
):
    """Upload a motion .npz file to wandb as an artifact.

    Args:
        motion_file: Path to .npz motion file
        artifact_name: Name for the artifact (e.g., "microduck-holonomous-walk")
        project: Wandb project name
        entity: Wandb entity/username (optional, uses default if not specified)
    """
    motion_path = Path(motion_file)
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    # Initialize wandb
    if entity:
        wandb.init(project=project, entity=entity, job_type="upload-motion")
    else:
        wandb.init(project=project, job_type="upload-motion")

    # Create artifact
    artifact = wandb.Artifact(
        name=artifact_name,
        type="motions",  # Must be "motions" (plural) to match mjlab's play script
        description=f"BeyondMimic motion file: {motion_path.name}",
        metadata={
            "filename": motion_path.name,
            "format": "beyondmimic_npz",
        }
    )

    # Add motion file (must be named "motion.npz" in the artifact)
    # Copy to temporary location with correct name
    temp_dir = Path("temp_motion_upload")
    temp_dir.mkdir(exist_ok=True)
    temp_motion = temp_dir / "motion.npz"
    shutil.copy(motion_path, temp_motion)

    artifact.add_file(str(temp_motion))

    # Log artifact
    print(f"Uploading {motion_file} as artifact '{artifact_name}'...")
    wandb.log_artifact(artifact)

    # Cleanup
    shutil.rmtree(temp_dir)

    # Get artifact path for training
    if entity:
        artifact_path = f"{entity}/{project}/{artifact_name}:latest"
    else:
        # Use wandb's default entity
        artifact_path = f"{wandb.run.entity}/{project}/{artifact_name}:latest"

    wandb.finish()

    print(f"\n✓ Success!")
    print(f"Artifact uploaded: {artifact_path}")
    print(f"\nTo train with this motion:")
    print(f"  uv run train Mjlab-BeyondMimic-MicroDuck \\")
    print(f"    --env.scene.num-envs 2048 \\")
    print(f"    --registry-name {artifact_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload BeyondMimic motion file to wandb as an artifact"
    )
    parser.add_argument(
        "motion_file",
        type=str,
        help="Path to .npz motion file"
    )
    parser.add_argument(
        "--artifact-name",
        type=str,
        required=True,
        help="Name for the artifact (e.g., 'microduck-holonomous-walk')"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="mjlab_microduck",
        help="Wandb project name (default: mjlab_microduck)"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Wandb entity/username (optional)"
    )
    args = parser.parse_args()

    upload_motion_artifact(
        args.motion_file,
        args.artifact_name,
        args.project,
        args.entity,
    )


if __name__ == "__main__":
    main()
