"""TRON2 Pro (WF_TRON1A) wheel-biped constants for mjlab."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF 与资源加载
##

_HERE = Path(__file__).parent

TRON_PRO_XML = _HERE / "xmls" / "tron_pro.xml"
WF_TRON1A_MESH_DIR = _HERE / "xmls" / "assets"
assert TRON_PRO_XML.exists(), f"Missing MJCF file: {TRON_PRO_XML}"
assert WF_TRON1A_MESH_DIR.exists(), f"Missing mesh directory: {WF_TRON1A_MESH_DIR}"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, WF_TRON1A_MESH_DIR, meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(TRON_PRO_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


LEG_EFFORT_LIMIT = 80.0
WHEEL_EFFORT_LIMIT = 40.0

ARMATURE = 0.01

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 目标固有频率 10 Hz（角频率）
DAMPING_RATIO = 2.0  # 过阻尼：>1 避免抖动

STIFFNESS = ARMATURE * NATURAL_FREQ**2
DAMPING = 2 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ

TRON_PRO_LEG_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "abad_.*_Joint",
    "hip_.*_Joint",
    "knee_.*_Joint",
  ),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=LEG_EFFORT_LIMIT,
  armature=ARMATURE,
)

TRON_PRO_WHEEL_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("wheel_.*_Joint",),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=WHEEL_EFFORT_LIMIT,
  armature=ARMATURE,
)

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.85),
  joint_pos={
    "abad_L_Joint": 0.0,
    "abad_R_Joint": 0.0,
    # Left/right hip & knee joint axes are opposite in WF_TRON1A.
    # Set mirrored initial values explicitly to keep stance symmetric.
    "hip_L_Joint": 0.3,
    "hip_R_Joint": -0.3,
    "knee_L_Joint": 0.6,
    "knee_R_Joint": -0.6,
    "wheel_L_Joint": 0.0,
    "wheel_R_Joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# 碰撞配置
##
#
# 所有名字以 "_collision" 结尾的 geom + 脚底球 (L_foot / R_foot)
# 共同构成碰撞集合；脚底单独给一个柔一点的 solimp，模拟缓冲脚垫。
##

_foot_regex = r"^[LR]_foot$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision", _foot_regex),
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp={_foot_regex: (0.015, 1, 0.03)},
)

##
# 最终组装
##

TRON_PRO_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(TRON_PRO_LEG_ACTUATOR_CFG, TRON_PRO_WHEEL_ACTUATOR_CFG),
  soft_joint_pos_limit_factor=0.9,
)


def get_tron_pro_robot_cfg() -> EntityCfg:
  """构造一份新的 TRON2 Pro 机器人配置。

  每次调用都会返回新实例，避免配置被意外共享/修改。
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=TRON_PRO_ARTICULATION,
  )


##
# 动作尺度（action scale）
##
#
# 推导公式：scale = 0.25 * effort_limit / stiffness
# 物理含义：策略网络输出 ±1 时，对应关节位置偏移不超过 0.25·力矩上限除以刚度。
# 这样能让 PPO 输出的高斯分布保持合理幅度。
##

TRON_PRO_ACTION_SCALE: dict[str, float] = {}
for _a in TRON_PRO_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  _d = _a.damping
  _names = _a.target_names_expr
  assert _e is not None
  for _n in _names:
    TRON_PRO_ACTION_SCALE[_n] = 0.25 * _e / _s


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_tron_pro_robot_cfg())
  viewer.launch(robot.spec.compile())
