"""TRON2 Pro 速度跟踪任务的强化学习超参数配置。

使用 PPO（Proximal Policy Optimization）算法。
网络结构、学习率等超参数先沿用 ANYmal C 的默认值，
等模型能跑起来后你再根据训练曲线调优。
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def tron_pro_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """构造 TRON2 Pro 速度任务的 RL runner 配置。

  返回：
    一个 RslRlOnPolicyRunnerCfg 对象，包含 actor/critic 网络结构、
    PPO 算法超参数、实验名、最大迭代步数等。
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      stochastic=True,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      entropy_coef=0.01,
    ),
    experiment_name="tron_pro_velocity",
    max_iterations=10_000,
  )
