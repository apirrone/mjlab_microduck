import os
import sys
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .microduck_imitation_env_cfg import (
    make_microduck_imitation_env_cfg,
    MicroduckImitationRlCfg,
)
from .microduck_standing_env_cfg import (
    make_microduck_standing_env_cfg,
    MicroduckStandingRlCfg,
)
from .microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
    MicroduckStandupRlCfg,
    StandupOnPolicyRunner,
)

# Standard velocity task (no imitation)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)

# Standing (balance) task - no walking, just robust standing
register_mjlab_task(
    task_id="Mjlab-Standing-Flat-MicroDuck",
    env_cfg=make_microduck_standing_env_cfg(),
    play_env_cfg=make_microduck_standing_env_cfg(play=True),
    rl_cfg=MicroduckStandingRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)
print("✓ Standing task registered: Mjlab-Standing-Flat-MicroDuck")

# Standup task - start fallen, learn to stand up
register_mjlab_task(
    task_id="Mjlab-Standup-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandupRlCfg,
    runner_cls=StandupOnPolicyRunner,
)
print("✓ Standup task registered: Mjlab-Standup-Flat-MicroDuck")

# Imitation motion tracking task
# Uses frame-based reference motions (reference_motion.pkl)
_imitation_motion_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # mjlab_microduck package dir
    "data",
    "reference_motion.pkl"
)

if os.path.exists(_imitation_motion_path):
    # Check if ghost visualization should be enabled (via GHOST env var or --ghost in argv)
    _enable_ghost_vis = (
        os.environ.get("GHOST", "0") == "1" or
        "--ghost" in sys.argv
    )

    register_mjlab_task(
        task_id="Mjlab-Imitation-Flat-MicroDuck",
        env_cfg=make_microduck_imitation_env_cfg(ghost_vis=False),  # Never show ghost during training
        play_env_cfg=make_microduck_imitation_env_cfg(play=True, ghost_vis=_enable_ghost_vis),
        rl_cfg=MicroduckImitationRlCfg,  # Use dedicated RL config with "imitation" prefix
        runner_cls=VelocityOnPolicyRunner,
    )
    ghost_status = "enabled" if _enable_ghost_vis else "disabled"
    print(f"✓ Imitation task registered: Mjlab-Imitation-Flat-MicroDuck (ghost vis: {ghost_status})")
else:
    print(f"Warning: Imitation motion file not found at {_imitation_motion_path}")
    print("Imitation task 'Mjlab-Imitation-Flat-MicroDuck' not registered.")
