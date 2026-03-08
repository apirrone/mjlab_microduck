import os
import sys
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    """Extends VelocityOnPolicyRunner to sync env.common_step_counter on resume.

    Without this, all step-based curricula reset to their initial values when
    training is resumed from a checkpoint, because common_step_counter always
    starts at 0 on env creation.
    """

    def load(self, path: str, load_optimizer: bool = True, map_location=None):
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        # Sync the env step counter so curricula resume at the correct stage.
        resumed_steps = self.current_learning_iteration * self.cfg["num_steps_per_env"]
        self.env.unwrapped.common_step_counter = resumed_steps
        print(f"[INFO] Resumed at iteration {self.current_learning_iteration} "
              f"→ common_step_counter set to {resumed_steps}")
        return infos

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
from .microduck_jump_env_cfg import (
    make_microduck_jump_env_cfg,
    MicroduckJumpRlCfg,
)

# Standard velocity task (no imitation)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ StandUp task registered: Mjlab-StandUp-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(rough=True),
    play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ StandUp task registered: Mjlab-StandUp-Rough-MicroDuck")

# Jump task — episodic policy: crouch to preload, jump, land, return to standing
register_mjlab_task(
    task_id="Mjlab-Jump-Flat-MicroDuck",
    env_cfg=make_microduck_jump_env_cfg(),
    play_env_cfg=make_microduck_jump_env_cfg(play=True),
    rl_cfg=MicroduckJumpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Jump task registered: Mjlab-Jump-Flat-MicroDuck")

# Ground pick task — episodic policy: crouch, touch ground with mouth, return to standing
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Ground pick task registered: Mjlab-GroundPick-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Ground pick task registered: Mjlab-GroundPick-Rough-MicroDuck")

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
        runner_cls=MicroduckOnPolicyRunner,
    )
    ghost_status = "enabled" if _enable_ghost_vis else "disabled"
    print(f"✓ Imitation task registered: Mjlab-Imitation-Flat-MicroDuck (ghost vis: {ghost_status})")
else:
    print(f"Warning: Imitation motion file not found at {_imitation_motion_path}")
    print("Imitation task 'Mjlab-Imitation-Flat-MicroDuck' not registered.")
