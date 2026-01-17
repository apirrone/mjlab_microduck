import os
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)

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
# or modify the path below
_reference_motion_path = os.environ.get(
    "MICRODUCK_REFERENCE_MOTION_PATH",
    os.path.expanduser("~/MISC/mjlab_microduck/src/mjlab_microduck/data/reference_motions.pkl")
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
