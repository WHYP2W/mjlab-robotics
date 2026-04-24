"""TRON2 Pro 双足机器人速度跟踪任务注册入口。

本文件被 mjlab 在启动时自动发现（通过 pyproject.toml 中的 entry point），
负责把我们定义好的环境配置和 RL 配置打包成可调用的 task_id。

注册后，你可以用命令行调用：
  uv run train Mjlab-Velocity-Flat-Tron-Pro
  uv run train Mjlab-Velocity-Rough-Tron-Pro
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  tron_pro_rough_env_cfg,
)
from .rl_cfg import tron_pro_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Tron-Pro",
  env_cfg=tron_pro_rough_env_cfg(),
  play_env_cfg=tron_pro_rough_env_cfg(play=True),
  rl_cfg=tron_pro_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)