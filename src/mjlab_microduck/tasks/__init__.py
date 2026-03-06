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
from .microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
    MicroduckGroundPickRlCfg,
)
from .microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
    MicroduckStandUpRlCfg,
)
from .microduck_roll_env_cfg import (
    make_microduck_roll_env_cfg,
    MicroduckRollRlCfg,
)

# Standard velocity task (no imitation)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)
print("✓ StandUp task registered: Mjlab-StandUp-Flat-MicroDuck")

# Roll task — episodic policy: stand → forward roll → stand
register_mjlab_task(
    task_id="Mjlab-Roll-Flat-MicroDuck",
    env_cfg=make_microduck_roll_env_cfg(),
    play_env_cfg=make_microduck_roll_env_cfg(play=True),
    rl_cfg=MicroduckRollRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)
print("✓ Roll task registered: Mjlab-Roll-Flat-MicroDuck")

# Ground pick task — episodic policy: crouch, touch ground with mouth, return to standing
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)
print("✓ Ground pick task registered: Mjlab-GroundPick-Flat-MicroDuck")

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
