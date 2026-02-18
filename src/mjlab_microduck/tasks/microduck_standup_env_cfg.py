"""Microduck standup environment configuration.

The robot is initialised lying on its front or back at z=0.3m, then dropped.
For the first SETTLE_TIME seconds the physics run freely; after that, rewards
activate to train the robot to reach and hold the default standing pose.
"""

import math
from copy import deepcopy

import torch
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
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_collisions_constants import MICRODUCK_COLLISIONS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp


class SettleActionsWrapper:
    """Zeros policy actions for the first ``settle_steps`` env-steps after each reset.

    Prevents the policy from fighting the physics during the initial fall so the
    robot settles into a stable ground pose before training begins.
    """

    def __init__(self, env, settle_steps: int) -> None:
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "settle_steps", settle_steps)

    def step(self, actions: torch.Tensor):
        # episode_length_buf is 0 at the very first step after a reset.
        mask = (self._env.episode_length_buf < self.settle_steps).float().unsqueeze(-1)
        actions = actions * (1.0 - mask)
        return self._env.step(actions)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_env"), name)

    def __setattr__(self, name: str, value) -> None:
        if name in ("_env", "settle_steps"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_env"), name, value)


class StandupOnPolicyRunner(VelocityOnPolicyRunner):
    """VelocityOnPolicyRunner that freezes policy actions for the first second after each reset."""

    SETTLE_TIME_S: float = 1.0

    def __init__(self, env, train_cfg, log_dir=None, device="cpu") -> None:
        super().__init__(env, train_cfg, log_dir, device)
        settle_steps = int(
            self.env.max_episode_length / self.env.cfg.episode_length_s * self.SETTLE_TIME_S
        )
        self.env = SettleActionsWrapper(self.env, settle_steps)


# How long to let the robot fall and settle before rewards activate (seconds).
SETTLE_TIME = 2.0


def make_microduck_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck standup environment configuration."""

    cfg = make_velocity_env_cfg()

    # ── Robot ──────────────────────────────────────────────────────────────────
    cfg.scene.entities = {"robot": MICRODUCK_COLLISIONS_ROBOT_CFG}
    cfg.viewer.body_name = "trunk_base"

    # Flat terrain only
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Sensors: we only need the feet contact sensor (for air-time etc.)
    cfg.scene.sensors = ()

    # More collision geoms → more contacts per world; give headroom
    cfg.sim.nconmax = 200
    cfg.sim.njmax = 600

    # ── Observations ──────────────────────────────────────────────────────────
    # Simple proprioceptive state; no velocity commands.
    base_obs = cfg.observations["policy"].terms
    policy_terms = {
        "projected_gravity": ObservationTermCfg(
            func=velocity_mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
            noise=Unoise(n_min=-0.007, n_max=0.007),
        ),
        "base_ang_vel": deepcopy(base_obs["base_ang_vel"]),
        "joint_pos": deepcopy(base_obs["joint_pos"]),
        "joint_vel": deepcopy(base_obs["joint_vel"]),
        "actions": deepcopy(base_obs["actions"]),
    }
    cfg.observations["policy"] = ObservationGroupCfg(
        terms=policy_terms,
        enable_corruption=not play,
        concatenate_terms=True,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
        terms=deepcopy(policy_terms),
        enable_corruption=False,
        concatenate_terms=True,
    )

    # ── Commands ──────────────────────────────────────────────────────────────
    # No velocity commands for standup.
    cfg.commands = {}

    # ── Actions ───────────────────────────────────────────────────────────────
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards ───────────────────────────────────────────────────────────────
    cfg.rewards = {
        # Stand upright (active after SETTLE_TIME)
        "upright": RewardTermCfg(
            func=microduck_mdp.standup_upright,
            weight=10.0,
            params={"settle_time": SETTLE_TIME, "std": 0.4},
        ),
        # Reach default joint positions (active after SETTLE_TIME)
        "joint_pos": RewardTermCfg(
            func=microduck_mdp.standup_joint_pos,
            weight=5.0,
            params={
                "settle_time": SETTLE_TIME,
                "std": 0.5,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
            },
        ),
        # Stand at the right height (active after SETTLE_TIME)
        "height": RewardTermCfg(
            func=microduck_mdp.standup_height,
            weight=3.0,
            params={
                "settle_time": SETTLE_TIME,
                "target_height_min": 0.08,
                "target_height_max": 0.11,
            },
        ),
        # Regularisation (always active)
        "action_rate_l2": RewardTermCfg(
            func=velocity_mdp.action_rate_l2,
            weight=-0.3,
        ),
        "joint_torques_l2": RewardTermCfg(
            func=microduck_mdp.joint_torques_l2,
            weight=-1e-3,
        ),
        # Terminal penalty
        "termination": RewardTermCfg(
            func=velocity_mdp.is_terminated,
            weight=-10.0,
        ),
    }

    # ── Events ────────────────────────────────────────────────────────────────
    # Drop the robot from a higher height so it has room to fall and settle.
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.3, 0.3)

    # Randomise orientation to lying on front or back.
    cfg.events["randomize_fallen_orientation"] = EventTermCfg(
        func=microduck_mdp.randomize_fallen_orientation,
        mode="reset",
    )

    # Reset action history (keeps action-rate penalty consistent across episodes).
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    # Remove events that reference velocity commands or foot-friction geoms.
    for key in list(cfg.events.keys()):
        if key not in ("reset_base", "randomize_fallen_orientation", "reset_action_history"):
            del cfg.events[key]

    # ── Terminations ──────────────────────────────────────────────────────────
    # No fell_over: the robot STARTS fallen; only terminate on time-out.
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=velocity_mdp.time_out, time_out=True),
    }

    # ── Curriculum ────────────────────────────────────────────────────────────
    cfg.curriculum = {}

    # ── Episode length ────────────────────────────────────────────────────────
    cfg.episode_length_s = 20.0

    return cfg


MicroduckStandupRlCfg = RslRlOnPolicyRunnerCfg(
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
    max_iterations=20_000,
)
