"""Microduck environment"""

from copy import deepcopy

# Domain randomization toggles
ENABLE_COM_RANDOMIZATION = True
ENABLE_KP_RANDOMIZATION = True

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
)
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp


def make_microduck_velocity_env_cfg(
    play: bool = False, use_imitation: bool = False, reference_motion_path: str = ""
) -> ManagerBasedRlEnvCfg:
    """Create Microduck velocity tracking environment configuration."""

    std_walking = {
        # Lower body
        r".*hip_yaw.*": 0.3,
        r".*hip_roll.*": 0.2,
        r".*hip_pitch.*": 0.4,
        r".*knee.*": 0.4,
        r".*ankle.*": 0.25, # was 0.15
        # Head
        r".*neck.*": 0.1,
        r".*head.*": 0.1,
    }

    site_names = ["left_foot", "right_foot"]

    # Contact sensor for feet - LEFT, RIGHT order
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(foot_tpu_bottom|foot)$",  # LEFT foot first, RIGHT foot second
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

    # Base configuration
    cfg = make_velocity_env_cfg()

    # for to_remove in [
    #     # "foot_clearance",
    #     # "foot_swing_height",
    #     # "angular_momentum",
    #     # "body_ang_vel",
    # ]:
    #     del cfg.rewards[to_remove]

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    # Robot setup
    cfg.scene.entities = {"robot": MICRODUCK_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    # Action configuration
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # === REWARDS ===
    # Pose reward configuration
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.1}
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].weight = 1.0  # was 2.0

    # Body-specific reward configurations
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 1.0  # was 1.0

    # Foot-specific configurations
    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    # Body-specific configurations
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)

    cfg.rewards["foot_slip"].weight = -0.1  # was -1.0
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.01

    # Body dynamics rewards
    cfg.rewards["soft_landing"].weight = -1e-05

    # Air time reward
    cfg.rewards["air_time"].weight = 5.0  # Increased to encourage walking
    cfg.rewards["air_time"].params["command_threshold"] = 0.01
    cfg.rewards["air_time"].params["threshold_min"] = 0.035  # was 0.025 (worked with fast gait)
    cfg.rewards["air_time"].params["threshold_max"] = 0.15

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    # Velocity tracking rewards (will be disabled when using imitation)
    cfg.rewards["track_linear_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].weight = 2.0

    # Action smoothness
    cfg.rewards["action_rate_l2"].weight = -0.4 # -0.5 doesn't work somehow, stiff left leg

    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.01
    cfg.rewards["foot_clearance"].params["target_height"] = 0.01  # Reduced for small robot (was 0.03)

    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.01  # Reduced for small robot (was 0.03)

    # cfg.rewards["leg_action_rate_l2"] = RewardTermCfg(
    # func=microduck_mdp.leg_action_rate_l2, weight=-0.5
    # )

    # Leg joint velocity penalty (encourage slower, smoother motion)
    # cfg.rewards["leg_joint_vel_l2"] = RewardTermCfg(
    # func=microduck_mdp.leg_joint_vel_l2, weight=-0.02
    # )

    # Neck stability
    # cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
    # func=microduck_mdp.neck_action_rate_l2, weight=-5.0 # was -0.1
    # )
    # cfg.rewards["neck_joint_vel_l2"] = RewardTermCfg(
    # func=microduck_mdp.neck_joint_vel_l2, weight=-0.1
    # )

    # CoM height target
    cfg.rewards["com_height_target"] = RewardTermCfg(
        func=microduck_mdp.com_height_target,
        weight=1.0,
        params={
            "target_height_min": 0.09,
            "target_height_max": 0.13,
        },
    )

    # === SURVIVAL REWARD (applies to all tasks) ===
    # Critical baseline reward for staying alive
    # cfg.rewards["survival"] = RewardTermCfg(
    #     func=microduck_mdp.is_alive, weight=2.0
    # )

    # === REGULARIZATION REWARDS (applies to all tasks) ===
    # Joint torques penalty
    # cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    # func=microduck_mdp.joint_torques_l2, weight=-1e-3
    # )

    # Joint accelerations penalty
    # cfg.rewards["joint_accelerations_l2"] = RewardTermCfg(
    # func=microduck_mdp.joint_accelerations_l2, weight=-2.5e-6
    # )

    # Leg action acceleration penalty
    # cfg.rewards["leg_action_acceleration_l2"] = RewardTermCfg(
    # func=microduck_mdp.leg_action_acceleration_l2, weight=-0.45
    # )

    # Neck action acceleration penalty
    # cfg.rewards["neck_action_acceleration_l2"] = RewardTermCfg(
    # func=microduck_mdp.neck_action_acceleration_l2, weight=-5.0
    # )

    # Imitation learning setup (optional, lightweight guidance)
    imitation_state = None
    if use_imitation and reference_motion_path:
        from mjlab_microduck.reference_motion import ReferenceMotionLoader
        import os

        if not os.path.exists(reference_motion_path):
            raise FileNotFoundError(
                f"Reference motion file not found: {reference_motion_path}"
            )

        # Create reference motion loader and imitation state
        ref_motion_loader = ReferenceMotionLoader(reference_motion_path)
        imitation_state = microduck_mdp.ImitationRewardState(ref_motion_loader)

        # Keep velocity tracking rewards active (helps with command following)
        # Reduced weight to not compete too much with imitation
        cfg.rewards["track_linear_velocity"].weight = 2.0
        cfg.rewards["track_angular_velocity"].weight = 1.0

        # Disable rewards not in the paper's imitation table
        cfg.rewards["air_time"].weight = 0.0
        cfg.rewards["soft_landing"].weight = 0.0

        # Add imitation reward (rebalanced for better command following)
        cfg.rewards["imitation"] = RewardTermCfg(
            func=microduck_mdp.imitation_reward,
            weight=1.0,
            params={
                "imitation_state": imitation_state,
                "command_threshold": 0.01,
                "weight_torso_pos_xy": 1.0,
                "weight_torso_orient": 1.0,
                "weight_lin_vel_xy": 1.0,
                "weight_lin_vel_z": 1.0,
                "weight_ang_vel_xy": 0.5,
                "weight_ang_vel_z": 0.5,
                "weight_leg_joint_pos": 10.0,
                "weight_neck_joint_pos": 30.0,
                "weight_leg_joint_vel": 1e-3,
                "weight_neck_joint_vel": 0.5,
                "weight_contact": 5.0,
            },
        )

    # Events
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
        params={"imitation_state": imitation_state}
        if imitation_state is not None
        else {},
    )

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = foot_frictions_geom_names
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.3, 0.3),
        "y": (-0.3, 0.3),
    }

    # Domain randomization - sampled once per episode at reset
    if ENABLE_COM_RANDOMIZATION:
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        # Randomize CoM position (±0.1cm on xyz)
        cfg.events["randomize_com"] = EventTermCfg(
            func=mdp.randomize_field,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "field": "body_ipos",  # Body inertial position (CoM)
                "ranges": (-0.001, 0.001),  # ±0.1cm
            },
        )

    if ENABLE_KP_RANDOMIZATION:
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        # Randomize motor kp gains (±2%)
        # Uses custom function that handles DelayedActuator
        cfg.events["randomize_motor_kp"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": (0.98, 1.02),  # ±2%
                "kd_range": (1.0, 1.0),  # Keep kd unchanged
            },
        )

    # Observations
    del cfg.observations["policy"].terms["base_lin_vel"]

    cfg.observations["policy"].terms["projected_gravity"] = deepcopy(
        cfg.observations["policy"].terms["projected_gravity"]
    )
    cfg.observations["policy"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["policy"].terms["base_ang_vel"]
    )

    cfg.observations["policy"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["policy"].terms["base_ang_vel"].delay_max_lag = 2
    cfg.observations["policy"].terms["base_ang_vel"].delay_update_period = 64

    cfg.observations["policy"].terms["projected_gravity"].delay_min_lag = 0
    cfg.observations["policy"].terms["projected_gravity"].delay_max_lag = 2
    cfg.observations["policy"].terms["projected_gravity"].delay_update_period = 64

    # Add imitation observations if using imitation
    if use_imitation and reference_motion_path and imitation_state is not None:
        from mjlab.managers.manager_term_config import ObservationTermCfg

        # Add phase observation to policy (for both training and play)
        cfg.observations["policy"].terms["imitation_phase"] = ObservationTermCfg(
            func=microduck_mdp.imitation_phase_observation,
            params={"imitation_state": imitation_state},
        )

        # Add phase observation to critic as well
        cfg.observations["critic"].terms["imitation_phase"] = ObservationTermCfg(
            func=microduck_mdp.imitation_phase_observation,
            params={"imitation_state": imitation_state},
        )

        # Add reference motion to critic privileged observations
        # Include in both training and play to keep model architecture consistent
        cfg.observations["critic"].terms["reference_motion"] = ObservationTermCfg(
            func=microduck_mdp.reference_motion_observation,
            params={"imitation_state": imitation_state},
        )

    # Commands - matched to reference motion coverage!
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.02  # Small amount for balance/stability training
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = (-0.4, 0.4)
    command.ranges.lin_vel_y = (-0.4, 0.4)
    command.ranges.ang_vel_z = (-1.0, 1.0)
    command.viz.z_offset = 1.0

    # Terrain
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # # Add action rate curriculum
    # cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
    #     func=mdp.reward_weight,
    #     params={
    #         "reward_name": "action_rate_l2",
    #         "weight_stages": [
    #             {"step": 0, "weight": -0.5},
    #             {"step": 10000, "weight": -0.7},
    #             {"step": 20000, "weight": -1.0},
    #         ],
    #     },
    # )

    # Disable default curriculum
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

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
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
