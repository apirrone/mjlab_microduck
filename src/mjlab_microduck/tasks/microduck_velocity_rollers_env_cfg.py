"""Microduck velocity environment — roller skate variant"""

import math
from copy import deepcopy

# Domain randomization toggles (same as velocity env)
ENABLE_COM_RANDOMIZATION = True
ENABLE_KP_RANDOMIZATION = True
ENABLE_KD_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = False
ENABLE_JOINT_DAMPING_RANDOMIZATION = False
ENABLE_VELOCITY_PUSHES = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False
ENABLE_NECK_OFFSET_RANDOMIZATION = True

NECK_OFFSET_MAX_ANGLE = 2.5
NECK_OFFSET_INTERVAL_S = (2.0, 5.0)

USE_PROJECTED_GRAVITY = True

COM_RANDOMIZATION_RANGE = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
KP_RANDOMIZATION_RANGE = (0.85, 1.15)
KD_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.98, 1.02)
JOINT_DAMPING_RANDOMIZATION_RANGE = (0.98, 1.02)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
VELOCITY_PUSH_RANGE = (-0.3, 0.3)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 1.0
BASE_ORIENTATION_MAX_PITCH_DEG = 10.0
BASE_ORIENTATION_MAX_ROLL_DEG = 5.0

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

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROLLERS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp


def make_microduck_velocity_rollers_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck roller skate velocity tracking environment configuration."""

    # Roller skates: feet stay near ground so pose stds are tighter on lower body.
    # Head stds are relaxed as with the walk env.
    std_standing = {
        r".*hip_yaw.*": 0.1,
        r".*hip_roll.*": 0.1,
        r".*hip_pitch.*": 0.1,
        r".*knee.*": 0.1,
        r".*ankle.*": 0.1,
        r".*neck.*": 0.05,
        r".*head.*": 0.05,
        r".*passive_.*": 999.0,  # passive wheel joints: effectively unconstrained
    }

    std_walking = {
        r".*hip_yaw.*": 0.3,
        r".*hip_roll.*": 0.1,
        r".*hip_pitch.*": 0.4,
        r".*knee.*": 0.4,
        r".*ankle.*": 0.25,
        r".*neck.*": 0.1,
        r".*head.*": 0.1,
        r".*passive_.*": 999.0,
    }

    std_running = {
        r".*hip_yaw.*": 0.5,
        r".*hip_roll.*": 0.2,
        r".*hip_pitch.*": 0.8,
        r".*knee.*": 0.8,
        r".*ankle.*": 0.5,
        r".*neck.*": 0.1,
        r".*head.*": 0.1,
        r".*passive_.*": 999.0,
    }

    # The roller XML defines left_foot and right_foot sites (same names as walk robot)
    site_names = ["left_foot", "right_foot"]

    # Contact sensor: roller_foot1 (left) and roller_foot2 (right) as subtree roots.
    # Each subtree contains both wheels of that side, so the slot is only "in contact"
    # when at least one wheel on that side touches the ground — and "airborne" only when
    # the whole skate lifts off. Using individual wheel bodies here would let the robot
    # cheat by leaning on the rear wheel while lifting the front one.
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(roller_foot1|roller_foot2)$",
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

    cfg = make_velocity_env_cfg()

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    # Robot setup
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROLLERS_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    # Action configuration — identical to the walk env (passive wheels have no actuators)
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    if ENABLE_NECK_OFFSET_RANDOMIZATION:
        joint_pos_action.class_type = microduck_mdp.NeckOffsetJointPositionAction

    # === REWARDS ===
    cfg.rewards["pose"].params["std_standing"] = std_standing
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_running
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].params["running_threshold"] = 0.5
    cfg.rewards["pose"].weight = 2.0

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 1.0

    # Foot-specific site names — foot_clearance and foot_swing_height stay on
    # (skating requires lifting feet for each stroke), foot_slip is removed
    # (rolling/sliding is inherent to skating)
    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)

    cfg.rewards["air_time"].weight = 5.0
    cfg.rewards["air_time"].params["command_threshold"] = 0.01
    del cfg.rewards["foot_slip"]  # skating involves rolling/sliding — don't penalise it

    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.01
    cfg.rewards["foot_clearance"].params["target_height"] = 0.02
    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.02

    cfg.rewards["soft_landing"].weight = -1e-05
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    cfg.rewards["track_linear_velocity"].weight = 5.0
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.08)  # tighter: steeper gradient away from 0
    cfg.rewards["track_angular_velocity"].weight = 3.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.25)

    cfg.rewards["action_rate_l2"].weight = -0.6


    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-0.1
    )

    cfg.rewards["com_height_target"] = RewardTermCfg(
        func=microduck_mdp.com_height_target,
        weight=1.2,
        params={
            "target_height_min": 0.0935,  # 0.08 + 0.0135 (roller height offset)
            "target_height_max": 0.1235,  # 0.11 + 0.0135
        },
    )

    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1e-3
    )

    # === EVENTS ===
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    if ENABLE_NECK_OFFSET_RANDOMIZATION:
        cfg.events["reset_neck_offset"] = EventTermCfg(
            func=microduck_mdp.reset_neck_offset,
            mode="reset",
        )
        cfg.events["randomize_neck_offset_target"] = EventTermCfg(
            func=microduck_mdp.randomize_neck_offset_target,
            mode="interval",
            interval_range_s=NECK_OFFSET_INTERVAL_S,
            params={"max_offset": NECK_OFFSET_MAX_ANGLE},
        )

    # foot_friction event uses named geoms (left_foot_collision / right_foot_collision)
    # which don't exist in the roller model — delete it
    del cfg.events["foot_friction"]

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.1335, 0.1435)  # +0.0135 roller height offset

    if ENABLE_VELOCITY_PUSHES:
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

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

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=mdp.randomize_field,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "field": "dof_frictionloss",
                "ranges": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_JOINT_DAMPING_RANDOMIZATION:
        cfg.events["randomize_joint_damping"] = EventTermCfg(
            func=mdp.randomize_field,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "field": "dof_damping",
                "ranges": JOINT_DAMPING_RANDOMIZATION_RANGE,
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

    if ENABLE_BASE_ORIENTATION_RANDOMIZATION:
        cfg.events["randomize_base_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_base_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_pitch_deg": BASE_ORIENTATION_MAX_PITCH_DEG,
                "max_roll_deg": BASE_ORIENTATION_MAX_ROLL_DEG,
            },
        )

    # === OBSERVATIONS ===
    del cfg.observations["policy"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )

    gravity_term_name = "projected_gravity" if USE_PROJECTED_GRAVITY else "raw_accelerometer"

    if not USE_PROJECTED_GRAVITY:
        del cfg.observations["policy"].terms["projected_gravity"]
        cfg.observations["policy"].terms["raw_accelerometer"] = ObservationTermCfg(
            func=microduck_mdp.raw_accelerometer,
            scale=1.0,
        )

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

    cfg.observations["policy"].terms["base_ang_vel"].noise = Unoise(n_min=-0.024, n_max=0.024)
    cfg.observations["policy"].terms[gravity_term_name].noise = Unoise(n_min=-0.007, n_max=0.007)
    cfg.observations["policy"].terms["joint_pos"].noise = Unoise(n_min=-0.0006, n_max=0.0006)
    cfg.observations["policy"].terms["joint_vel"].noise = Unoise(n_min=-0.024, n_max=0.024)

    # Exclude passive wheel joints from joint_pos and joint_vel observations.
    # Wheel angles accumulate unboundedly; exclude them to keep the obs space clean.
    # (Wheel velocity is also excluded for simplicity; add a custom obs term later if needed.)
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    cfg.observations["policy"].terms["joint_pos"].params["asset_cfg"] = passive_excluded
    cfg.observations["policy"].terms["joint_vel"].params["asset_cfg"] = passive_excluded
    cfg.observations["critic"].terms["joint_pos"].params["asset_cfg"] = deepcopy(passive_excluded)
    cfg.observations["critic"].terms["joint_vel"].params["asset_cfg"] = deepcopy(passive_excluded)

    # === COMMANDS ===
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = (-0.3, 0.3)
    command.ranges.lin_vel_y = (0.0, 0.0)  # lateral motion impossible on roller skates
    command.ranges.ang_vel_z = (-1.5, 1.5)
    command.viz.z_offset = 0.5
    command.class_type = microduck_mdp.VelocityCommandCommandOnly

    # Flat terrain only for now
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # === CURRICULUM ===
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.4},
                {"step": 250 * 24, "weight": -0.8},
                {"step": 500 * 24, "weight": -1.0},
            ],
        },
    )

    cfg.curriculum["standing_envs"] = CurriculumTermCfg(
        func=microduck_mdp.standing_envs_curriculum,
        params={
            "command_name": "twist",
            "standing_stages": [
                {"step": 0,           "rel_standing_envs": 0.0},   # no standing: force velocity commands
                {"step": 1000 * 24,   "rel_standing_envs": 0.05},
                {"step": 1500 * 24,   "rel_standing_envs": 0.1},
                {"step": 2000 * 24,   "rel_standing_envs": 0.15},
            ],
        },
    )

    cfg.curriculum["velocity_command_ranges"] = CurriculumTermCfg(
        func=microduck_mdp.velocity_command_ranges_curriculum,
        params={
            "command_name": "twist",
            "update_lin_vel_y": False,  # lateral motion impossible on roller skates
            "velocity_stages": [
                {"step": 0,          "lin_vel_range": 0.3,  "ang_vel_range": 1.5},
                {"step": 500 * 24,   "lin_vel_range": 0.35, "ang_vel_range": 1.6},
                {"step": 1000 * 24,  "lin_vel_range": 0.4,  "ang_vel_range": 1.7},
            ],
        },
    )

    if ENABLE_NECK_OFFSET_RANDOMIZATION:
        cfg.curriculum["neck_offset_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.neck_offset_curriculum,
            params={
                "event_name": "randomize_neck_offset_target",
                "offset_stages": [
                    {"step": 0,          "max_offset": 0.0},
                    {"step": 500 * 24,   "max_offset": 0.1},
                    {"step": 750 * 24,   "max_offset": 0.2},
                    {"step": 1000 * 24,  "max_offset": 0.3},
                    {"step": 1500 * 24,  "max_offset": 0.5},
                    {"step": 2000 * 24,  "max_offset": 0.7},
                    {"step": 2500 * 24,  "max_offset": 0.9},
                    {"step": 3000 * 24,  "max_offset": 1.1},
                    {"step": 3500 * 24,  "max_offset": 1.3},
                    {"step": 4000 * 24,  "max_offset": 1.5},
                    {"step": 4500 * 24,  "max_offset": 1.7},
                    {"step": 5000 * 24,  "max_offset": 1.9},
                    {"step": 5500 * 24,  "max_offset": 2.1},
                    {"step": 6000 * 24,  "max_offset": 2.3},
                    {"step": 6500 * 24,  "max_offset": NECK_OFFSET_MAX_ANGLE},
                ],
            },
        )

    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    return cfg


MicroduckRollersRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="velocity_rollers",
    run_name="velocity_rollers",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
