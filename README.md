# mjlab 示例：anymal_c_velocity（ANYmal C 速度跟踪任务）

![ANYmal C locomotion](teaser.gif)

本示例展示了如何将一个**自定义机器人**集成到现有的 mjlab 任务中。这里我们训练一个 ANYmal C 四足机器人行走，并让它跟踪指令下达的机身速度。

## 项目结构

```
src/anymal_c_velocity/
  __init__.py                        # 任务注册（入口点）
  env_cfgs.py                        # 环境配置（传感器、奖励、终止条件）
  rl_cfg.py                          # 强化学习超参数（PPO 算法）
  anymal_c/
    anymal_c_constants.py            # 机器人定义（执行器、碰撞、初始状态）
    xmls/
      anymal_c.xml                   # MuJoCo MJCF 模型文件
      assets/                        # 网格模型与贴图资源
```

## 工作原理

### 1. 依赖 mjlab

在 `pyproject.toml` 文件中声明对 `mjlab` 的依赖，并配置一个 entry point（入口点），这样 mjlab 在导入时就能自动发现你的任务：

```toml
[build-system]
requires = ["uv_build>=0.8.18,<0.9.0"]
build-backend = "uv_build"

[project]
dependencies = ["mjlab>=1.1.0"]

[project.entry-points."mjlab.tasks"]
anymal_c_velocity = "anymal_c_velocity"
```

`[build-system]` 表是必需的。如果没有配置构建后端（build backend），该包不会被安装到环境中，entry point 也就不会被注册。

### 2. 定义你的机器人

`anymal_c_constants.py` 提供了机器人的 `EntityCfg`：包含一个 MuJoCo spec（从 XML 加载）、执行器参数、碰撞属性以及关节初始状态。这是整个项目中**唯一**与你机器人硬件相关的部分。

关键组成部分：
- **`get_spec()`**：加载 MJCF XML 文件及其网格模型资源。
- **`BuiltinPositionActuatorCfg`**：PD 控制增益、力矩限制、电机 armature（转子惯量）。
- **`EntityCfg`**：将 spec、执行器、碰撞和初始状态整合在一起。

### 3. 配置环境

`env_cfgs.py` 以 `make_velocity_env_cfg()`（内置速度任务的默认配置）为起点，并针对你的机器人进行定制：

- 将 `cfg.scene.entities` 设置为你的机器人。
- 配置接触传感器（哪些 geom 是脚部、什么算作非法接触）。
- 调整奖励权重、终止条件和可视化（viewer）设置。
- 可选：设置地形课程学习（terrain curriculum）。

还提供了一个 `play=True` 的变体，用于在评估时关闭随机化。

### 4. 配置强化学习

`rl_cfg.py` 返回一个带有 PPO 超参数的 `RslRlOnPolicyRunnerCfg`（包括网络架构、学习率、clip 参数等）。建议先使用默认值，再据此进行调优。

### 5. 注册任务

`__init__.py` 为每个变体（rough/flat，即崎岖地形/平坦地形）调用 `register_mjlab_task()`，并传入环境配置、强化学习配置和 runner 类：

```python
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Anymal-C",
  env_cfg=anymal_c_flat_env_cfg(),
  play_env_cfg=anymal_c_flat_env_cfg(play=True),
  rl_cfg=anymal_c_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
```

## 使用方法

```sh
# 健全性检查：观察机器人在零动作下站立然后倒下的过程
uv run play Mjlab-Velocity-Flat-Anymal-C --agent zero

# 训练
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Flat-Anymal-C \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3_000

# 回放训练好的模型检查点
uv run play Mjlab-Velocity-Flat-Anymal-C --wandb-run-path <wandb-run-path>
```
