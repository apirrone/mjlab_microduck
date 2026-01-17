"""Microduck environment"""

import torch
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab_microduck.robot.microduck_constants import MICRODUCK_ROBOT_CFG
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.managers.manager_term_config import EventTermCfg, RewardTermCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab.tasks.velocity import mdp

from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.entity import Entity

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def make_microduck_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    std_walking = {
        # Lower body
        r".*hip_yaw.*": 0.3,
        r".*hip_roll.*": 0.2,
        r".*hip_pitch.*": 0.4,
        r".*knee.*": 0.4,
        r".*ankle.*": 0.15,
        # Head
        r".*neck.*": 0.1,
        r".*head.*": 0.1,
    }

    site_names = ["left_foot", "right_foot"]

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(foot|foot_2)$",
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

    cfg.viewer.body_name = "trunk_base"
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)

    cfg.scene.entities = {"robot": MICRODUCK_ROBOT_CFG}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    cfg.rewards["pose"].params["std_standing"] = {".*": 0.1}
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking

    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    cfg.rewards["foot_clearance"].params["target_height"] = 0.03
    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.01
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.03
    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = foot_frictions_geom_names

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["air_time"].weight = 0.5
    cfg.rewards["air_time"].params["command_threshold"] = 0.01

    cfg.rewards["track_linear_velocity"].weight = 4.0  # Was 2.0
    cfg.rewards["track_angular_velocity"].weight = 4.0  # Was 2.0

    # Removing base lin velocity observation
    del cfg.observations["policy"].terms["base_lin_vel"]

    #   def log_debug(env: ManagerBasedRlEnv, _):
    #     print("Sensor")
    #     print(env.scene.sensors["feet_ground_contact_left_foot_link_force"].data)
    #   cfg.events["log_debug"] = EventTermCfg(mode="interval", func=log_debug, interval_range_s=(0.0, 0.0))

    # cfg.actions["joint_pos"].actuator_names = (
    #     r".*(?<!head_yaw)(?<!head_pitch)(?<!head_roll)$",
    # )

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    cfg.commands["twist"].viz.z_offset = 1.0

    # Walking on plane only
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Disabling curriculum
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    #   cfg.sim.nconmax = 256
    #   cfg.sim.njmax = 512

    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.8, 0.8),
        "y": (-0.8, 0.8),
    }

    # Slightly increased L2 action rate penalty
    cfg.rewards["action_rate_l2"].weight = -0.3

    # Penalizing torque
    #   cfg.rewards["torque_l2"] = RewardTermCfg(func=mdp.joint_torques_l2, weight=-1e-4)

    # Penalizing velocities
    #   cfg.rewards["vel_l2"] = RewardTermCfg(func=joint_vel_l2, weight=-1e-3)

    # Disabling self-collision
    cfg.rewards["self_collisions"].weight = 0.0

    # More standing env, disabling heading envs
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.15  # Was 0.25
    command.rel_heading_envs = 0.0

    cfg.observations["policy"].terms["projected_gravity"] = deepcopy(
        cfg.observations["policy"].terms["projected_gravity"]
    )
    cfg.observations["policy"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["policy"].terms["base_ang_vel"]
    )

    cfg.observations["policy"].terms["base_ang_vel"].delay_min_lag = 1
    cfg.observations["policy"].terms["base_ang_vel"].delay_max_lag = 2
    cfg.observations["policy"].terms["base_ang_vel"].delay_update_period = 64

    cfg.observations["policy"].terms["projected_gravity"].delay_min_lag = 1
    cfg.observations["policy"].terms["projected_gravity"].delay_max_lag = 2
    cfg.observations["policy"].terms["projected_gravity"].delay_update_period = 64

    cfg.commands["twist"].ranges.ang_vel_z = (-1.0, 1.0)
    cfg.commands["twist"].ranges.lin_vel_y = (-1.0, 1.0)
    cfg.commands["twist"].ranges.lin_vel_x = (-1.0, 1.0)

    if play:
        #Disabling push
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
        }

        # Custom command
        # cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
        # cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
        # cfg.commands["twist"].ranges.lin_vel_x = (0.0, 0.0)
        # cfg.commands["twist"].rel_standing_envs = 0.0

    return cfg


MicroduckRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="microduck_velocity",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=50_000,
)
