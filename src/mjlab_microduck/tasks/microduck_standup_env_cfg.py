"""Microduck stand-up environment configuration.

The robot is initialized lying on its back (body inverted 180° from upright)
and must learn to right itself and reach a stable standing posture.
"""

from copy import deepcopy

# Domain randomization toggles
ENABLE_COM_RANDOMIZATION = True
ENABLE_KP_RANDOMIZATION = True
ENABLE_KD_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True

# Domain randomization ranges
COM_RANDOMIZATION_RANGE = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
KP_RANDOMIZATION_RANGE = (0.85, 1.15)
KD_RANDOMIZATION_RANGE = (0.9, 1.1)
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
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp


def make_microduck_standup_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck stand-up environment configuration.

    The robot starts lying on its back (upside down) and must learn to
    right itself and reach a stable upright stance.
    """

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

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )

    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    if not rough:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    elif play and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = 20.0

    # Action configuration
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # === OBSERVATIONS ===
    del cfg.observations["policy"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=velocity_mdp.base_lin_vel,
        scale=1.0,
    )

    cfg.observations["policy"].terms["projected_gravity"] = deepcopy(
        cfg.observations["policy"].terms["projected_gravity"]
    )
    cfg.observations["policy"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["policy"].terms["base_ang_vel"]
    )

    cfg.observations["policy"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["policy"].terms["base_ang_vel"].delay_max_lag = 3
    cfg.observations["policy"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["policy"].terms["projected_gravity"].delay_min_lag = 0
    cfg.observations["policy"].terms["projected_gravity"].delay_max_lag = 3
    cfg.observations["policy"].terms["projected_gravity"].delay_update_period = 64

    cfg.observations["policy"].terms["base_ang_vel"].noise = Unoise(n_min=-0.024, n_max=0.024)
    cfg.observations["policy"].terms["projected_gravity"].noise = Unoise(n_min=-0.007, n_max=0.007)
    cfg.observations["policy"].terms["joint_pos"].noise = Unoise(n_min=-0.0006, n_max=0.0006)
    cfg.observations["policy"].terms["joint_vel"].noise = Unoise(n_min=-0.024, n_max=0.024)
    cfg.observations["policy"].enable_corruption = not play

    # === COMMANDS ===
    # Always zero velocity — task is to stand up, not walk
    command = cfg.commands["twist"]
    command.rel_standing_envs = 1.0
    command.rel_heading_envs = 0.0
    command.resampling_time_range = (1.0e9, 1.0e9)
    command.debug_vis = False
    command.class_type = microduck_mdp.VelocityCommandCommandOnly

    # === REWARDS ===
    cfg.rewards = {
        # Linear upright reward: +1 when vertical, 0 when horizontal, -1 when inverted.
        # Provides non-zero gradient at every tilt angle, unlike a narrow Gaussian
        # which is ~0 at the 90° prone starting position.
        "upright_linear": RewardTermCfg(
            func=microduck_mdp.body_upright_linear,
            weight=4.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
        ),
        # Reward upward CoM velocity: directly incentivizes the dynamic push needed
        # to go from prone to standing. Clamped to zero on the way down so the robot
        # isn't penalized for settling once upright.
        "com_upward_velocity": RewardTermCfg(
            func=microduck_mdp.com_upward_velocity,
            weight=3.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "max_height": 0.08,  # matches target_height_min — reward vanishes once standing
            },
        ),
        # Height reward: quadratic penalty below target, +1 when in standing range.
        "com_height_target": RewardTermCfg(
            func=microduck_mdp.com_height_target,
            weight=5.0,
            params={
                "target_height_min": 0.08,
                "target_height_max": 0.11,
            },
        ),
        # Pose reward only once standing (std is loose — don't over-constrain during
        # the dynamic standup phase, but reward the final upright pose).
        "pose": RewardTermCfg(
            func=velocity_mdp.variable_posture,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "command_name": "twist",
                "std_standing": {r".*": 0.5},
                "std_walking": {r".*": 0.5},
                "std_running": {r".*": 0.5},
                "walking_threshold": 0.01,
                "running_threshold": 1.5,
            },
        ),
        # Regularization — kept very light so motion penalties don't outweigh
        # the upward-velocity and upright rewards during the standup phase.
        "action_rate_l2": RewardTermCfg(
            func=velocity_mdp.action_rate_l2,
            weight=-0.01,
        ),
        "joint_torques_l2": RewardTermCfg(
            func=microduck_mdp.joint_torques_l2,
            weight=-1e-5,
        ),
        "dof_pos_limits": RewardTermCfg(
            func=velocity_mdp.joint_pos_limits,
            weight=-1.0,
        ),
        "self_collisions": RewardTermCfg(
            func=velocity_mdp.self_collision_cost,
            weight=-1.0,
            params={"sensor_name": self_collision_cfg.name},
        ),
    }

    # === TERMINATIONS ===
    # Remove fell_over — robot starts inverted, would terminate immediately
    del cfg.terminations["fell_over"]

    # === EVENTS ===
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    # Robot is pitched 90° forward (belly/front facing ground). Trunk CoM sits
    # at roughly the body's half-depth above the ground. Set z high enough that
    # the neck/head don't clip the floor given HOME_FRAME neck joint angles.
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.15)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names

    # Override orientation to face-down (runs after reset_base sets position)
    cfg.events["set_face_down"] = EventTermCfg(
        func=microduck_mdp.set_face_down_orientation,
        mode="reset",
    )

    # Domain randomization
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=velocity_mdp.randomize_field,
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

    # === CURRICULUM ===
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=velocity_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.01},
                {"step": 500 * 24,   "weight": -0.1},
                {"step": 1000 * 24,  "weight": -0.3},
                {"step": 2000 * 24,  "weight": -0.6},
                {"step": 2500 * 24,  "weight": -0.8},
                {"step": 3000 * 24,  "weight": -1.0},
            ],
        },
    )

    return cfg


MicroduckStandUpRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="standup",
    run_name="standup",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
