import os
from pathlib import Path

import mujoco
from mjlab.actuator import DelayedActuatorCfg, XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from mjlab_microduck.robot.microduck_constants import HOME_FRAME, actuators

MICRODUCK_COLLISIONS_XML: Path = Path(os.path.dirname(__file__)) / "microduck_collisions" / "robot.xml"
assert MICRODUCK_COLLISIONS_XML.exists(), f"XML not found: {MICRODUCK_COLLISIONS_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_COLLISIONS_XML))


# Same foot properties as FULL_COLLISION but disable_other_geoms=False so the
# trunk (uc) and head collision meshes defined in the XML remain active.
COLLISIONS_COLLISION_CFG = CollisionCfg(
    geom_names_expr=[r"^(left|right)_foot_collision$"],
    condim={r"^(left|right)_foot_collision$": 3},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (0.6,)},
    disable_other_geoms=False,
)

MICRODUCK_COLLISIONS_ROBOT_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=HOME_FRAME,
    collisions=(COLLISIONS_COLLISION_CFG,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)
