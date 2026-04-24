"""TRON2 Pro (WF_TRON1A) wheel-biped velocity task environment config."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from tron_pro_velocity.tron_pro.tron_pro_constants import (
  TRON_PRO_ACTION_SCALE,
  get_tron_pro_robot_cfg,
)


def tron_pro_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """TRON2 Pro 崎岖地形速度跟踪环境配置。

  参数：
    play: 是否处于回放/评测模式。True 时会关闭随机化、拉长 episode 等。
  """
  # Start from mjlab built-in velocity template, then override robot-specific terms.
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  # Wheel-biped on rough hfield creates many simultaneous contacts.
  # Increase contact buffers to avoid "height field collision overflow".
  cfg.sim.contact_sensor_maxmatch = 2000
  cfg.sim.nconmax = 400

  # Bind robot entity for this task.
  cfg.scene.entities = {"robot": get_tron_pro_robot_cfg()}

  # Ensure terrain scan is evaluated in robot base frame.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base"

  site_names = ("L", "R")
  geom_names = ("L_foot", "R_foot")

  # Foot-ground contact is consumed by air-time / slip / landing rewards.
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # Any non-foot geometry touching terrain is considered illegal contact.
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision(\d*)?$",
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # Convert normalized policy actions into joint position deltas.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = TRON_PRO_ACTION_SCALE

  # Track the base body defined in tron_pro.xml.
  cfg.viewer.body_name = "base"
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  # Domain randomization targets.
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)

  # Pose prior bandwidths (smaller value -> stronger pull to nominal pose).
  cfg.rewards["pose"].params["std_standing"] = {
    "abad_.*_Joint": 0.08,
    "hip_.*_Joint": 0.10,
    "knee_.*_Joint": 0.15,
    "wheel_.*_Joint": 1.0,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    "abad_.*_Joint": 0.25,
    "hip_.*_Joint": 0.30,
    "knee_.*_Joint": 0.45,
    "wheel_.*_Joint": 2.0,
  }
  cfg.rewards["pose"].params["std_running"] = {
    "abad_.*_Joint": 0.25,
    "hip_.*_Joint": 0.30,
    "knee_.*_Joint": 0.45,
    "wheel_.*_Joint": 2.0,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # Keep these disabled initially; enable when stability tuning is needed.
  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0

  # Early terminate when body parts other than wheels/feet hit the ground.
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

  cmd = cfg.commands["twist"]
  assert isinstance(cmd, UniformVelocityCommandCfg)
  cmd.viz.z_offset = 0.5

  if play:
    # Deterministic long-horizon evaluation mode.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def tron_pro_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """TRON2 Pro 平地速度任务配置（仅保留 flat 任务时使用）。"""
  # Reuse rough config, then simplify terrain and sensor stack.
  cfg = tron_pro_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.nconmax = 128

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # No terrain height scan needed on flat ground.
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)

  # Remove terrain curriculum in flat setup.
  cfg.curriculum.pop("terrain_levels", None)

  return cfg