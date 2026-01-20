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
from mjlab.managers.manager_term_config import EventTermCfg, RewardTermCfg, ObservationTermCfg, CurriculumTermCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab.tasks.velocity import mdp

from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.entity import Entity

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.reference_motion import ReferenceMotionLoader
import torch

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def debug_joint_properties(env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG):
    """Debug function to print joint properties on first call."""
    if not hasattr(debug_joint_properties, "_printed"):
        print("\n" + "="*70)
        print("MJLAB JOINT PROPERTIES DEBUG")
        print("="*70)

        # Get model from the sim
        model = env.sim.mj_model

        print(f"\nModel timestep: {model.opt.timestep:.6f}s")
        print(f"Number of actuators: {model.nu}")

        print("\nACTUATOR PROPERTIES:")
        for i in range(model.nu):
            import mujoco
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            joint_id = model.actuator_trnid[i, 0]
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)

            # Get actuator properties
            gain = model.actuator_gainprm[i, 0]  # kp
            bias = model.actuator_biasprm[i, 1]  # kv
            forcerange = model.actuator_forcerange[i]

            # Get DOF address for this joint (joints can have multiple DOFs)
            dof_adr = model.jnt_dofadr[joint_id]
            damping = model.dof_damping[dof_adr]
            armature = model.dof_armature[dof_adr]
            frictionloss = model.dof_frictionloss[dof_adr]

            print(f"  {i:2d}. {actuator_name:20s} -> {joint_name:20s}")
            print(f"      Actuator: kp={gain:6.2f}, kv={bias:6.2f}, force=[{forcerange[0]:6.2f}, {forcerange[1]:6.2f}]")
            print(f"      Joint:    damping={damping:6.3f}, armature={armature:6.4f}, friction={frictionloss:6.3f}")

        print("="*70 + "\n")
        debug_joint_properties._printed = True


def joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def make_microduck_velocity_env_cfg(
    play: bool = False,
    use_imitation: bool = False,
    reference_motion_path: str = None,
) -> ManagerBasedRlEnvCfg:
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

    # CRITICAL: Body names in robot.xml are:
    # - Left foot: "foot_tpu_bottom" (child of left_roll_to_pitch)
    # - Right foot: "foot" (child of right_roll_to_pitch)
    # Pattern must match in LEFT, RIGHT order to match reference motion
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

    cfg.rewards["foot_clearance"].params["target_height"] = 0.01  # 1cm clearance for 23cm tall robot
    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.01
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.01  # 1cm swing height for 23cm tall robot
    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01

    # CRITICAL: Strong penalty for foot sliding to prevent "split and slide" behavior
    cfg.rewards["foot_slip"].weight = -2.0  # Increased from default -0.1
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.01

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Add debug event to print joint properties at startup
    cfg.events["debug_joint_properties"] = EventTermCfg(
        func=debug_joint_properties,
        mode="startup",
    )

    # Add reset event to clear action history caches
    # This is critical for action rate and acceleration penalties
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = foot_frictions_geom_names

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    # STRONGLY encourage proper gait with foot lifting (not sliding!)
    # Limit air time to prevent jumping - small robot should take quick steps
    cfg.rewards["air_time"].weight = 3.0  # Increased from 0.5 to force foot lifting
    cfg.rewards["air_time"].params["command_threshold"] = 0.01
    cfg.rewards["air_time"].params["threshold_min"] = 0.02  # Minimum air time for valid step
    cfg.rewards["air_time"].params["threshold_max"] = 0.15  # Maximum air time - prevents jumping!

    # Only set velocity tracking weights if NOT using imitation
    # (imitation curriculum will control these weights)
    if not use_imitation:
        cfg.rewards["track_linear_velocity"].weight = 4.0  # Was 2.0
        cfg.rewards["track_angular_velocity"].weight = 4.0  # Was 2.0
    else:
        cfg.rewards["track_linear_velocity"].weight = 0.0  # Controlled by curriculum
        cfg.rewards["track_angular_velocity"].weight = 0.0  # Controlled by curriculum

    # Removing base lin velocity observation
    del cfg.observations["policy"].terms["base_lin_vel"]

    #   def log_debug(env: ManagerBasedRlEnv, _):
    #     print("Sensor")
    #     print(env.scene.sensors["feet_ground_contact_left_foot_link_force"].data)
    #   cfg.events["log_debug"] = EventTermCfg(mode="interval", func=log_debug, interval_range_s=(0.0, 0.0))

    # cfg.actions["joint_pos"].actuator_names = (
    #     r".*(?<!head_yaw)(?<!head_pitch)(?<!head_roll)$",
    # )

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.12)

    cfg.commands["twist"].viz.z_offset = 1.0

    # Walking on plane only
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Disable default curriculum (terrain levels, command velocity)
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    #   cfg.sim.nconmax = 256
    #   cfg.sim.njmax = 512

    # Push range for robustness and recovery behavior training
    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.8, 0.8),
        "y": (-0.8, 0.8),
    }

    # Reduce action rate penalty to allow dynamic movement (was -0.5, too restrictive)
    cfg.rewards["action_rate_l2"].weight = -0.5

    # Add specific neck penalties to keep head stable
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2,
        weight=-2.0  # Stronger penalty for neck action changes
    )
    cfg.rewards["neck_joint_vel_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_joint_vel_l2,
        weight=-0.5  # Penalty for neck joint velocities
    )

    # Penalizing torque
    #   cfg.rewards["torque_l2"] = RewardTermCfg(func=mdp.joint_torques_l2, weight=-1e-4)

    # Penalizing velocities
    #   cfg.rewards["vel_l2"] = RewardTermCfg(func=joint_vel_l2, weight=-1e-3)

    # Disabling self-collision
    cfg.rewards["self_collisions"].weight = 0.0
    # Enable soft landing for better push recovery
    cfg.rewards["soft_landing"].weight = 0.5

    # CoM height target: keep center of mass between 0.1 and 0.15 meters
    cfg.rewards["com_height_target"] = RewardTermCfg(
        func=microduck_mdp.com_height_target,
        weight=1.0,
        params={
            "target_height_min": 0.07,
            "target_height_max": 0.13,
        }
    )

    # More standing env, disabling heading envs
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.1
    command.rel_heading_envs = 0.0

    cfg.observations["policy"].terms["projected_gravity"] = deepcopy(
        cfg.observations["policy"].terms["projected_gravity"]
    )
    cfg.observations["policy"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["policy"].terms["base_ang_vel"]
    )

    # Disable observation delays in play mode for cleaner testing
    # (During training, delays help with sim2real transfer)
    if not play:
        cfg.observations["policy"].terms["base_ang_vel"].delay_min_lag = 1
        cfg.observations["policy"].terms["base_ang_vel"].delay_max_lag = 2
        cfg.observations["policy"].terms["base_ang_vel"].delay_update_period = 64

        cfg.observations["policy"].terms["projected_gravity"].delay_min_lag = 1
        cfg.observations["policy"].terms["projected_gravity"].delay_max_lag = 2
        cfg.observations["policy"].terms["projected_gravity"].delay_update_period = 64

    cfg.commands["twist"].ranges.ang_vel_z = (-1.0, 1.0)
    cfg.commands["twist"].ranges.lin_vel_y = (-0.5, 0.5)
    cfg.commands["twist"].ranges.lin_vel_x = (-0.5, 0.5)

    # if play:
        #Disabling push
        # cfg.events["push_robot"].params["velocity_range"] = {
            # "x": (0.0, 0.0),
            # "y": (0.0, 0.0),
        # }

        # Custom command
        # cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
        # cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
        # cfg.commands["twist"].ranges.lin_vel_x = (0.0, 0.0)
        # cfg.commands["twist"].rel_standing_envs = 0.0

    # Imitation learning setup
    if use_imitation and reference_motion_path:
        import os
        if not os.path.exists(reference_motion_path):
            raise FileNotFoundError(f"Reference motion file not found: {reference_motion_path}")

        # Create reference motion loader and imitation state
        ref_motion_loader = ReferenceMotionLoader(reference_motion_path)
        imitation_state = microduck_mdp.ImitationRewardState(ref_motion_loader)

        if not play:
            # BD-X paper reward structure
            # Disable most rewards and keep only essential ones
            rewards_to_disable = [
                "upright", "body_ang_vel", "angular_momentum", "air_time", "foot_clearance", "foot_swing_height", "foot_slip",
                "self_collisions", "feet_stumble", "feet_slide", "dof_acc", "action_rate_l2"
            ]
            for reward_name in rewards_to_disable:
                if reward_name in cfg.rewards:
                    cfg.rewards[reward_name].weight = 0.0

            # Velocity tracking rewards - controlled by curriculum
            # Start at 0.0, curriculum will gradually increase them
            cfg.rewards["track_linear_velocity"].weight = 0.0  # Curriculum controls this
            cfg.rewards["track_angular_velocity"].weight = 0.0  # Curriculum controls this

            # Regularization rewards (BD-X paper Table I)
            # Joint torques: -||τ||², weight 1.0·10⁻³
            if "joint_torques_l2" not in cfg.rewards:
                cfg.rewards["joint_torques_l2"] = RewardTermCfg(
                    func=mdp.joint_torques_l2,
                    weight=-1.0e-3
                )
            else:
                cfg.rewards["joint_torques_l2"].weight = -1.0e-3

            # Joint accelerations: -||q̈||², weight 2.5·10⁻⁶
            cfg.rewards["joint_accelerations_l2"] = RewardTermCfg(
                func=microduck_mdp.joint_accelerations_l2,
                weight=-2.5e-6
            )

            # Leg action rate: -||a_l - a_{t-1,l}||², weight 1.5
            # INCREASED by 5x for smooth real robot behavior
            cfg.rewards["leg_action_rate_l2"] = RewardTermCfg(
                func=microduck_mdp.leg_action_rate_l2,
                weight=-2.5  # Was -0.5, increased for smoothness
            )

            # Neck action rate: -||a_n - a_{t-1,n}||², weight 5.0
            # INCREASED by 5x for smooth real robot behavior
            cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
                func=microduck_mdp.neck_action_rate_l2,
                weight=-5.0  # Was -1.0, increased for smoothness
            )

            # Leg action acceleration: -||a_l - 2a_{t-1,l} + a_{t-2,l}||², weight 0.45
            # INCREASED by 5x for smooth real robot behavior
            cfg.rewards["leg_action_acceleration_l2"] = RewardTermCfg(
                func=microduck_mdp.leg_action_acceleration_l2,
                weight=-0.5  # Was -0.1, increased for smoothness
            )

            # Neck action acceleration: -||a_n - 2a_{t-1,n} + a_{t-2,n}||², weight 5.0
            # INCREASED by 5x for smooth real robot behavior
            cfg.rewards["neck_action_acceleration_l2"] = RewardTermCfg(
                func=microduck_mdp.neck_action_acceleration_l2,
                weight=-5.0  # Was -1.0, increased for smoothness
            )

            # Survival reward: 1.0, weight 20.0 (REBALANCED for robustness)
            # Increased to encourage recovery from pushes and prioritize staying upright
            cfg.rewards["alive"] = RewardTermCfg(
                func=microduck_mdp.is_alive,
                weight=20.0  # Increased to balance with imitation reward
            )

            # Imitation reward (BD-X paper Table I)
            # Base weight matches curriculum start, will be controlled by curriculum
            cfg.rewards["imitation"] = RewardTermCfg(
                func=microduck_mdp.imitation_reward,
                weight=1.0, 
                params={
                    "imitation_state": imitation_state,
                    "command_threshold": 0.01,
                    # Torso tracking
                    "weight_torso_pos_xy": 1.0,  # exp(-200.0 * ||p_xy - p̂_xy||²)
                    "weight_torso_orient": 1.0,  # exp(-20.0 * ||θ ⊟ θ̂||²)
                    # Velocity tracking
                    "weight_lin_vel_xy": 1.0,  # exp(-8.0 * ||v_xy - v̂_xy||²)
                    "weight_lin_vel_z": 1.0,   # exp(-8.0 * (v_z - v̂_z)²)
                    "weight_ang_vel_xy": 0.5,  # exp(-2.0 * ||ω_xy - ω̂_xy||²)
                    "weight_ang_vel_z": 0.5,   # exp(-2.0 * (ω_z - ω̂_z)²)
                    # Joint tracking (separated leg vs neck)
                    "weight_leg_joint_pos": 15.0,   # -||q_l - q̂_l||²
                    "weight_neck_joint_pos": 30.0,  # -||q_n - q̂_n||² (reduced from 100.0 for robustness)
                    "weight_leg_joint_vel": 1.0e-3,  # -||q̇_l - q̇̂_l||²
                    "weight_neck_joint_vel": 1.0,    # -||q̇_n - q̇̂_n||²
                    # Contact tracking (INCREASED to encourage foot lifting during swing)
                    "weight_contact": 5.0,  # Was 10
                }
            )

            # Curriculum for imitation learning (TRAINING ONLY)
            # Gradually reduce imitation weight and increase velocity tracking requirements
            # Steps are in environment steps (training_iterations * steps_per_iteration)
            # Using multiplier of 24 * 2048 = 49152 steps per training iteration
            # cfg.curriculum["imitation_weight"] = CurriculumTermCfg(
                # func=mdp.reward_weight,
                # params={
                    # "reward_name": "imitation",
                    # "weight_stages": [
                        # {"step": 0,                "weight": 1.0},   # 0-5k iterations: Learn basic gait
                        # {"step": 5000 * 49152,     "weight": 0.8},   # 5-10k: Start reducing imitation dominance
                        # {"step": 10000 * 49152,    "weight": 0.6},   # 10-15k: Further reduce
                        # {"step": 15000 * 49152,    "weight": 0.5},   # 15-20k: Prioritize robustness
                    # ],
                # },
            # )
# 
            # cfg.curriculum["track_linear_velocity_weight"] = CurriculumTermCfg(
                # func=mdp.reward_weight,
                # params={
                    # "reward_name": "track_linear_velocity",
                    # "weight_stages": [
                        # {"step": 0,                "weight": 0.0},   # 0-5k iterations: Don't track velocity, focus on gait
                        # {"step": 5000 * 49152,     "weight": 1.0},   # 5-10k: Start velocity tracking
                        # {"step": 10000 * 49152,    "weight": 2.0},   # 10-15k: Increase requirement
                        # {"step": 15000 * 49152,    "weight": 4.0},   # 15-20k: Full velocity tracking
                    # ],
                # },
            # )
# 
            # cfg.curriculum["track_angular_velocity_weight"] = CurriculumTermCfg(
                # func=mdp.reward_weight,
                # params={
                    # "reward_name": "track_angular_velocity",
                    # "weight_stages": [
                        # {"step": 0,                "weight": 0.0},   # 0-5k iterations: Don't track velocity, focus on gait
                        # {"step": 5000 * 49152,     "weight": 1.0},   # 5-10k: Start velocity tracking
                        # {"step": 10000 * 49152,    "weight": 2.0},   # 10-15k: Increase requirement
                        # {"step": 15000 * 49152,    "weight": 4.0},   # 15-20k: Full velocity tracking
                    # ],
                # },
            # )

        # Add reset event to reset imitation phase
        cfg.events["reset_imitation_phase"] = EventTermCfg(
            func=lambda env, env_ids: imitation_state.reset_phases(env_ids),
            mode="reset",
        )

        # Add phase observation to policy (for both training and play)
        cfg.observations["policy"].terms["imitation_phase"] = ObservationTermCfg(
            func=microduck_mdp.imitation_phase_observation,
            params={"imitation_state": imitation_state}
        )

        # Add reference motion to critic privileged observations
        # Include in both training and play to keep model architecture consistent
        # (critic is not used during play, but needs same size to load checkpoint)
        cfg.observations["critic"].terms["reference_motion"] = ObservationTermCfg(
            func=microduck_mdp.reference_motion_observation,
            params={"imitation_state": imitation_state}
        )

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
