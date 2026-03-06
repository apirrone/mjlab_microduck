"""Microduck forward roll task.

Episodic policy that starts from a standing pose, performs a forward roll
(tucking, rolling over, recovering), and returns to a standing pose.
Identical obs/action space to the walking and ground-pick policies so all
three can be switched at runtime with a key-press.

Phase encoding (in the command slot, 3-D):
    command = [cos(2π·phase), sin(2π·phase), 0]
    phase ∈ [0, 0.5]  → roll phase   (reward forward pitch angular momentum)
    phase ∈ [0.5, 1]  → return phase (reward standing pose + upright orientation)

Phase is randomised per env on reset to decorrelate environments.
Period = 4 s (2 s rolling + 2 s recovering).
"""

from copy import deepcopy

# ── Domain randomisation ──────────────────────────────────────────────────────
ENABLE_COM_RANDOMIZATION          = True
ENABLE_KP_RANDOMIZATION           = True
ENABLE_KD_RANDOMIZATION           = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_VELOCITY_PUSHES            = False  # No pushes — roll dynamics are sensitive
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False
ENABLE_NECK_OFFSET_RANDOMIZATION  = False  # head is tucked as part of the roll

# ── Ranges (same as velocity env) ─────────────────────────────────────────────
COM_RANDOMIZATION_RANGE          = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
KP_RANDOMIZATION_RANGE           = (0.85, 1.15)
KD_RANDOMIZATION_RANGE           = (0.9, 1.1)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 1.0

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_ROLL_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp


def make_microduck_roll_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck forward roll environment configuration."""

    site_names = ["left_foot", "right_foot"]

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(foot_tpu_bottom|foot)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_ROLL_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg,)
    cfg.viewer.body_name = "trunk_base"

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: remove walking-specific terms ────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: roll objectives ───────────────────────────────────────────────

    # PRIMARY: phase-conditioned orientation.
    # upright × cos(2π·phase) — rewards upright at phase 0/1, inverted at phase 0.5.
    # Penalises "lean-and-recover": being upright at phase=0.5 gives reward = +1×(-1) = -1.
    # Only a genuine roll (inverted at 0.5) gets +1×(-1)... wait, inverted=-1, cos(π)=-1,
    # reward = (-1)×(-1) = +1.  Standing at 0.5: (+1)×(-1) = -1.
    cfg.rewards["roll_phase_orientation"] = RewardTermCfg(
        func=microduck_mdp.roll_phase_orientation,
        weight=6.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "command_name": "twist",
        },
    )

    # Roll phase: forward pitch angular momentum helps initiate and sustain the roll.
    # Lower weight than orientation — it's an auxiliary push signal.
    cfg.rewards["roll_pitch_momentum"] = RewardTermCfg(
        func=microduck_mdp.roll_pitch_angular_velocity,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "command_name": "twist",
            "max_ang_vel": 10.0,
        },
    )

    # Return phase: reward reaching standing height.
    cfg.rewards["roll_return_com_height"] = RewardTermCfg(
        func=microduck_mdp.roll_return_com_height,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "command_name": "twist",
            "target_height_min": 0.08,
            "target_height_max": 0.11,
        },
    )

    # Return phase: upward CoM velocity while still low.
    cfg.rewards["roll_return_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.roll_return_upward_velocity,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "command_name": "twist",
            "max_height": 0.08,
        },
    )

    # Return phase: return to standing joint pose.
    _LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
    cfg.rewards["roll_return_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_pose,
        weight=2.0,
        params={
            "std": 0.3,
            "command_name": "twist",
            "joint_indices": _LEG_JOINTS,
        },
    )

    _NECK_JOINTS = [5, 6, 7, 8]
    cfg.rewards["roll_return_pose_neck"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_pose,
        weight=2.0,
        params={
            "std": 0.15,
            "command_name": "twist",
            "joint_indices": _NECK_JOINTS,
        },
    )

    # ── Rewards: stability ────────────────────────────────────────────────────

    # Remove always-on upright: roll_phase_orientation already handles all phases.
    del cfg.rewards["upright"]

    # body_ang_vel removed: a forward roll requires very high angular velocity
    # (~5-10 rad/s); penalizing it would directly fight the roll motion.
    del cfg.rewards["body_ang_vel"]

    cfg.rewards["angular_momentum"].weight = -0.005  # minimal

    cfg.rewards["soft_landing"].weight = -1e-5

    # ── Rewards: regularisation ───────────────────────────────────────────────

    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=mdp.action_rate_l2, weight=-1.0
    )

    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-2e-3
    )

    # ── Observations (identical layout to walking policy) ─────────────────────
    del cfg.observations["policy"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    gravity_term_name = "projected_gravity"
    cfg.observations["policy"].terms[gravity_term_name] = deepcopy(
        cfg.observations["policy"].terms[gravity_term_name]
    )
    cfg.observations["policy"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["policy"].terms["base_ang_vel"]
    )

    cfg.observations["policy"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["policy"].terms["base_ang_vel"].delay_max_lag = 3
    cfg.observations["policy"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["policy"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["policy"].terms[gravity_term_name].delay_max_lag = 3
    cfg.observations["policy"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["policy"].terms["base_ang_vel"].noise   = Unoise(n_min=-0.024, n_max=0.024)
    cfg.observations["policy"].terms[gravity_term_name].noise = Unoise(n_min=-0.007, n_max=0.007)
    cfg.observations["policy"].terms["joint_pos"].noise      = Unoise(n_min=-0.0006, n_max=0.0006)
    cfg.observations["policy"].terms["joint_vel"].noise      = Unoise(n_min=-0.024, n_max=0.024)

    # ── Command: cyclic phase encoding ────────────────────────────────────────
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.class_type = microduck_mdp.RollPhaseCommand

    # ── Terminations ─────────────────────────────────────────────────────────
    # Remove fell_over: the robot WILL be inverted during the roll.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=mdp.randomize_field,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "field": "body_ipos",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=microduck_mdp.randomize_mass_and_inertia,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "scale_range": MASS_INERTIA_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        cfg.events["randomize_imu_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_imu_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE,
            },
        )

    # ── Terrain ───────────────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.2},
                {"step": 250 * 24,   "weight": -0.5},
                {"step": 500 * 24,   "weight": -1.0},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckRollRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="roll",
    run_name="roll",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=20_000,
)
