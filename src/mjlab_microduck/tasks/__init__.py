import os
import sys
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .microduck_imitation_env_cfg import make_microduck_imitation_env_cfg

# Standard velocity task (no imitation)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)

# Imitation learning task
# Set the MICRODUCK_REFERENCE_MOTION_PATH environment variable to your .pkl file
# Default: looks for reference_motions.pkl in the package's data directory
_default_reference_motion_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # mjlab_microduck package dir
    "data",
    "reference_motions.pkl"
)
_reference_motion_path = os.environ.get(
    "MICRODUCK_REFERENCE_MOTION_PATH",
    _default_reference_motion_path
)

if os.path.exists(_reference_motion_path):
    register_mjlab_task(
        task_id="Mjlab-Velocity-Flat-MicroDuck-Imitation",
        env_cfg=make_microduck_velocity_env_cfg(
            use_imitation=True,
            reference_motion_path=_reference_motion_path
        ),
        play_env_cfg=make_microduck_velocity_env_cfg(
            play=True,
            use_imitation=True,
            reference_motion_path=_reference_motion_path
        ),
        rl_cfg=MicroduckRlCfg,
        runner_cls=VelocityOnPolicyRunner,
    )
else:
    print(f"Warning: Reference motion file not found at {_reference_motion_path}")
    print("Imitation learning task 'Mjlab-Velocity-Flat-MicroDuck-Imitation' not registered.")
    print("To enable, set MICRODUCK_REFERENCE_MOTION_PATH environment variable or place file at default location.")

# Imitation motion tracking task
# Uses frame-based reference motions (reference_motion.pkl)
_imitation_motion_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # mjlab_microduck package dir
    "data",
    "reference_motion.pkl"
)

if os.path.exists(_imitation_motion_path):
    # Check if ghost visualization should be enabled (via --ghost flag)
    _enable_ghost_vis = "--ghost" in sys.argv

    register_mjlab_task(
        task_id="Mjlab-Imitation-Flat-MicroDuck",
        env_cfg=make_microduck_imitation_env_cfg(ghost_vis=False),  # Never show ghost during training
        play_env_cfg=make_microduck_imitation_env_cfg(play=True, ghost_vis=_enable_ghost_vis),
        rl_cfg=MicroduckRlCfg,  # Reuse the same RL config
        runner_cls=VelocityOnPolicyRunner,
    )
    ghost_status = "enabled" if _enable_ghost_vis else "disabled"
    print(f"✓ Imitation task registered: Mjlab-Imitation-Flat-MicroDuck (ghost vis: {ghost_status})")
else:
    print(f"Warning: Imitation motion file not found at {_imitation_motion_path}")
    print("Imitation task 'Mjlab-Imitation-Flat-MicroDuck' not registered.")
