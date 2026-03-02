"""Microduck standing + body control environment.

Train from scratch to stand still, recover from pushes, and follow body_cmd
(z-height offset, pitch, roll). Same 54D obs / 14D action space as the velocity env.
"""

import math

from mjlab.managers.manager_term_config import CurriculumTermCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    BODY_CMD_MAX_ANGLE_DEG,
    BODY_CMD_MAX_Z,
    make_microduck_velocity_env_cfg,
)


def make_microduck_standing_env_cfg(play: bool = False):
    """Standing + body control environment.

    Inherits robot setup, sensors, 54D observations, action config, and domain
    randomization from the velocity env, then strips walking rewards and
    replaces the phase-2-gated body_cmd curriculum with one that starts from iter 0.

    Reward set:
      - pose (std_standing)       — joint regularization toward home position
      - upright (reduced)         — orientation bias when body_cmd is zero
      - stillness_at_zero_command — reward staying still in XY (compatible with height/tilt)
      - body_cmd_tracking         — main task signal, grows via curriculum
      - action_rate_l2            — smoothness
      - neck_action_rate_l2       — neck stability
      - joint_torques_l2          — efficiency

    Removed from velocity env:
      - track_linear/angular_velocity, air_time, foot_clearance, foot_swing_height,
        foot_slip, soft_landing, body_ang_vel, angular_momentum  (walking only)
      - com_height_target  (fixed [0.08–0.11] range blocks z body_cmd)
    """

    cfg = make_microduck_velocity_env_cfg(play=play)

    # -------------------------------------------------------------------------
    # Commands — always standing, vel_cmd obs stays [0, 0, 0]
    # -------------------------------------------------------------------------
    command = cfg.commands["twist"]
    command.rel_standing_envs = 1.0
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)

    # -------------------------------------------------------------------------
    # Rewards
    # -------------------------------------------------------------------------

    # Strip all walking-specific rewards.
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
        "body_ang_vel",
        "angular_momentum",
        # Fixed height range [0.08, 0.11] directly blocks z body_cmd > ±1 cm.
        "com_height_target",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # Reduce upright: body_cmd will request pitch/roll, upright must not dominate.
    cfg.rewards["upright"].weight = 0.5

    # body_cmd_tracking starts at weight=0; curriculum below grows it from iter 200.
    cfg.rewards["body_cmd_tracking"].weight = 0.0

    # -------------------------------------------------------------------------
    # Curriculum
    # -------------------------------------------------------------------------

    # Remove velocity-env and phase-2-gate curricula.
    for name in ["standing_envs", "velocity_command_ranges"]:
        if name in cfg.curriculum:
            del cfg.curriculum[name]

    _max_angle = math.radians(BODY_CMD_MAX_ANGLE_DEG)

    # Body_cmd weight: brief 200-iter standing warm-up, then ramp up.
    cfg.curriculum["body_cmd_weight"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
            "reward_name": "body_cmd_tracking",
            "weight_stages": [
                {"step": 0,         "weight": 0.0},
                {"step": 200 * 24,  "weight": 1.0},
                {"step": 500 * 24,  "weight": 2.0},
                {"step": 1000 * 24, "weight": 3.0},
                {"step": 2000 * 24, "weight": 4.0},
            ],
        },
    )

    # Body_cmd range: grows in sync with reward weight.
    cfg.curriculum["body_cmd_range"] = CurriculumTermCfg(
        func=microduck_mdp.body_cmd_range_curriculum,
        params={
            "event_name": "randomize_body_cmd",
            "range_stages": [
                {"step": 0,
                 "max_z": 0.0, "max_pitch": 0.0, "max_roll": 0.0},
                {"step": 200 * 24,
                 "max_z": 0.005, "max_pitch": math.radians(2.0),  "max_roll": math.radians(2.0)},
                {"step": 500 * 24,
                 "max_z": 0.010, "max_pitch": math.radians(5.0),  "max_roll": math.radians(5.0)},
                {"step": 1000 * 24,
                 "max_z": 0.020, "max_pitch": math.radians(10.0), "max_roll": math.radians(10.0)},
                {"step": 2000 * 24,
                 "max_z": BODY_CMD_MAX_Z, "max_pitch": _max_angle, "max_roll": _max_angle},
            ],
        },
    )

    return cfg


MicroduckStandingRlCfg = RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="standing",
    run_name="standing",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
