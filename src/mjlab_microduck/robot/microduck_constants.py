import os
from pathlib import Path

import mujoco
from mjlab.actuator import DelayedActuatorCfg, XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

_ROBOT_DIR: Path = Path(os.path.dirname(__file__)) / "microduck"

MICRODUCK_XML: Path = _ROBOT_DIR / "robot.xml"
MICRODUCK_WALK_XML: Path = _ROBOT_DIR / "robot_walk.xml"
MICRODUCK_STANDUP_XML: Path = _ROBOT_DIR / "robot_standup.xml"
MICRODUCK_GROUND_PICK_XML: Path = _ROBOT_DIR / "robot_ground_pick.xml"

assert MICRODUCK_XML.exists(), f"XML not found: {MICRODUCK_XML}"
assert MICRODUCK_WALK_XML.exists(), f"XML not found: {MICRODUCK_WALK_XML}"
assert MICRODUCK_STANDUP_XML.exists(), f"XML not found: {MICRODUCK_STANDUP_XML}"
# robot_ground_pick.xml is optional — falls back to robot_standup.xml if absent
_GROUND_PICK_XML = MICRODUCK_GROUND_PICK_XML if MICRODUCK_GROUND_PICK_XML.exists() else MICRODUCK_STANDUP_XML


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_XML))


def get_walk_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_WALK_XML))


def get_standup_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_STANDUP_XML))


def get_ground_pick_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_GROUND_PICK_XML))


HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        # Lower body
        r".*hip_yaw.*": 0.0,
        r".*hip_roll.*": 0.0,
        r".*left_hip_pitch.*": 0.6,
        r".*right_hip_pitch.*": -0.6,
        r".*left_knee.*": -1.2,
        r".*right_knee.*": 1.2,
        r".*left_ankle.*": 0.6,
        r".*right_ankle.*": -0.6,
        # Head
        r".*neck_pitch.*": -0.5,
        r".*head_pitch.*": 0.5,
        r".*head_yaw.*": 0.0,
        r".*head_roll.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (0.6,)},
)

actuators = DelayedActuatorCfg(
    delay_min_lag=0,  # Increased from 0 - real actuators have consistent delay
    delay_max_lag=3,  # Increased from 3 - force lower-gain control
    base_cfg=XmlPositionActuatorCfg(joint_names_expr=(r".*",)),
)

# actuators=XmlPositionActuatorCfg(joint_names_expr=(r".*",))

MICRODUCK_ROBOT_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_STANDUP_ROBOT_CFG = EntityCfg(
    spec_fn=get_standup_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_GROUND_PICK_ROBOT_CFG = EntityCfg(
    spec_fn=get_ground_pick_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainImporterCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": MICRODUCK_ROBOT_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")
    viewer.launch(scene.compile())
