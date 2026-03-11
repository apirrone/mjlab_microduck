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
from .microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
    MicroduckRollersRlCfg,
)

def _make_roller_get_base_metadata():
    """Return a get_base_metadata replacement that skips joints with no actuator.

    Needed for the roller skate robot where passive wheel joints exist as DOFs
    but have no position actuators, causing a KeyError in the stock implementation.
    """
    import torch
    from mjlab.envs.mdp.actions.joint_actions import JointAction

    def roller_get_base_metadata(env, run_path):
        robot = env.scene["robot"]
        joint_action = env.action_manager.get_term("joint_pos")
        assert isinstance(joint_action, JointAction)

        joint_name_to_ctrl_id = {
            act.target.split("/")[-1]: act.id
            for act in robot.spec.actuators
        }

        # Filter to joints that have an actuator, preserving natural order
        all_names = list(robot.joint_names)
        actuated_idx = [i for i, n in enumerate(all_names) if n in joint_name_to_ctrl_id]
        joint_names = [all_names[i] for i in actuated_idx]
        ctrl_ids = [joint_name_to_ctrl_id[n] for n in joint_names]

        joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0]
        joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]

        return {
            "run_path": run_path,
            "joint_names": joint_names,
            "joint_stiffness": joint_stiffness.tolist(),
            "joint_damping": joint_damping.tolist(),
            "default_joint_pos": robot.data.default_joint_pos[0][actuated_idx].cpu().tolist(),
            "command_names": list(env.command_manager.active_terms),
            "observation_names": env.observation_manager.active_terms["policy"],
            "action_scale": joint_action._scale[0].cpu().tolist()
            if isinstance(joint_action._scale, torch.Tensor)
            else joint_action._scale,
        }

    return roller_get_base_metadata


class MicroduckRollersOnPolicyRunner(MicroduckOnPolicyRunner):
    """Runner for the roller skate task.

    Overrides save() to patch get_base_metadata in the velocity exporter module
    for the duration of the call, filtering out passive wheel joints that have no
    actuators and would cause a KeyError in the stock implementation.
    """

    def save(self, path, *args, **kwargs):
        import mjlab.tasks.velocity.rl.exporter as _vel_exporter
        orig = _vel_exporter.get_base_metadata
        _vel_exporter.get_base_metadata = _make_roller_get_base_metadata()
        try:
            super().save(path, *args, **kwargs)
        finally:
            _vel_exporter.get_base_metadata = orig

# Roller skate velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_velocity_rollers_env_cfg(),
    play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True),
    rl_cfg=MicroduckRollersRlCfg,
    runner_cls=MicroduckRollersOnPolicyRunner,
)
print("✓ Rollers task registered: Mjlab-Velocity-Flat-MicroDuck-Rollers")

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
