"""BeyondMimic tracking task for Microduck

This task uses the BeyondMimic implementation from mjlab to track reference motions.
Unlike the imitation task which uses polynomial-fitted motions, this uses full body
tracking with sampled trajectories.

To use this task:
1. Convert your polynomial reference motions to BeyondMimic format:
   python convert_to_beyondmimic.py ./src/mjlab_microduck/data/reference_motions.pkl \
       --output-dir ./beyondmimic_motions

2. Train with any of the generated motion files:
   uv run train Mjlab-BeyondMimic-MicroDuck --env.commands.motion.motion_file ./beyondmimic_motions/0.01_0.0_-0.4.npz
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_ROBOT_CFG


def make_microduck_beyondmimic_env_cfg(
    play: bool = False,
    motion_file: str = "",
) -> ManagerBasedRlEnvCfg:
    """Create Microduck BeyondMimic tracking environment configuration.

    Args:
        play: Whether this is for inference/play mode
        motion_file: Path to .npz motion file. Can be empty string - the train script
                    will download it from wandb artifacts based on --registry-name

    Returns:
        Environment configuration
    """
    # motion_file can be empty - it will be set by the train script from wandb artifacts

    # Start with base tracking config
    cfg = make_tracking_env_cfg()

    # Robot setup
    cfg.scene.entities = {"robot": MICRODUCK_ROBOT_CFG}
    cfg.viewer.body_name = "trunk_base"

    # Contact sensor for feet
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

    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

    # Configure motion command
    # Note: motion_file will be set by train script from wandb artifact if empty
    if motion_file:
        cfg.commands["motion"].motion_file = motion_file
    cfg.commands["motion"].anchor_body_name = "trunk_base"
    cfg.commands["motion"].body_names = ("trunk_base",)  # Can add more bodies for tracking
    cfg.commands["motion"].asset_name = "robot"

    # Adjust motion command randomization ranges for microduck
    cfg.commands["motion"].pose_range = {
        "x": (-0.02, 0.02),
        "y": (-0.02, 0.02),
        "z": (-0.01, 0.01),
        "roll": (-0.05, 0.05),
        "pitch": (-0.05, 0.05),
        "yaw": (-0.1, 0.1),
    }

    cfg.commands["motion"].velocity_range = {
        "x": (-0.3, 0.3),
        "y": (-0.3, 0.3),
        "z": (-0.1, 0.1),
        "roll": (-0.3, 0.3),
        "pitch": (-0.3, 0.3),
        "yaw": (-0.5, 0.5),
    }

    cfg.commands["motion"].joint_position_range = (-0.05, 0.05)

    # BeyondMimic sampling settings
    cfg.commands["motion"].sampling_mode = "adaptive"  # Can be: adaptive, uniform, start
    cfg.commands["motion"].adaptive_kernel_size = 5
    cfg.commands["motion"].adaptive_lambda = 0.8
    cfg.commands["motion"].adaptive_uniform_ratio = 0.1
    cfg.commands["motion"].adaptive_alpha = 0.001

    # Visualization
    cfg.commands["motion"].viz.mode = "ghost"  # Can be: ghost, frames
    cfg.commands["motion"].viz.ghost_color = (0.5, 0.7, 0.5, 0.5)

    # Action configuration
    cfg.actions["joint_pos"].scale = 0.5
    cfg.actions["joint_pos"].use_default_offset = True

    # === REWARDS ===
    # BeyondMimic rewards (exponential for smooth gradients)
    cfg.rewards["motion_global_root_pos"].weight = 0.5
    cfg.rewards["motion_global_root_pos"].params["std"] = 0.3

    cfg.rewards["motion_global_root_ori"].weight = 0.5
    cfg.rewards["motion_global_root_ori"].params["std"] = 0.4

    cfg.rewards["motion_body_pos"].weight = 1.0
    cfg.rewards["motion_body_pos"].params["std"] = 0.3

    cfg.rewards["motion_body_ori"].weight = 1.0
    cfg.rewards["motion_body_ori"].params["std"] = 0.4

    cfg.rewards["motion_body_lin_vel"].weight = 1.0
    cfg.rewards["motion_body_lin_vel"].params["std"] = 1.0

    cfg.rewards["motion_body_ang_vel"].weight = 1.0
    cfg.rewards["motion_body_ang_vel"].params["std"] = 3.14

    # Regularization
    cfg.rewards["action_rate_l2"].weight = -0.1

    cfg.rewards["joint_limit"].weight = -10.0

    cfg.rewards["self_collisions"].weight = -10.0
    cfg.rewards["self_collisions"].params["sensor_name"] = "self_collision"

    # === TERMINATIONS ===
    cfg.terminations["anchor_pos"].params["threshold"] = 0.25
    cfg.terminations["anchor_ori"].params["threshold"] = 0.8
    cfg.terminations["ee_body_pos"].params["threshold"] = 0.25
    cfg.terminations["ee_body_pos"].params["body_names"] = ()  # No end effector termination for now

    # === EVENTS ===
    # Foot friction randomization
    foot_frictions_geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names

    # Base COM randomization
    cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk_base",)

    # Push robot disturbances
    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.1, 0.1),
        "roll": (-0.3, 0.3),
        "pitch": (-0.3, 0.3),
        "yaw": (-0.5, 0.5),
    }

    # === OBSERVATIONS ===
    # The tracking task already has appropriate observations configured

    # Terrain
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    return cfg


MicroduckBeyondMimicRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="microduck_beyondmimic",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
