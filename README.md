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

# 新增任务：TRON2 Pro 双足机器人速度跟踪

本仓库在 ANYmal C 示例基础上扩展，增加了一个 **双足机器人 TRON2 Pro** 的速度跟踪任务，用于演示如何把 mjlab 的通用任务框架复用到不同形态的机器人上。

## 项目结构

```
src/tron_pro_velocity/
  __init__.py                        # 任务注册入口（当前仅注册 flat task）
  env_cfgs.py                        # 双足环境配置（2 只脚、新关节 regex）
  rl_cfg.py                          # 强化学习超参数（PPO）
  tron_pro/
    tron_pro_constants.py            # 双足机器人定义（执行器、碰撞、初始姿态）
    xmls/
      tron_pro.xml                   # 双足 MuJoCo MJCF 模板（骨架版，占位盒子）
      assets/                        # 真实 mesh/texture 放这里（当前为空）
```

## 当前状态：已接入真实 WF_TRON1A 模型

`tron_pro.xml` 已直接改为 `src/WF_TRON1A/xml/robot.xml` 的适配版本，不再是占位盒子。Mesh 文件通过相对路径从 `src/WF_TRON1A/meshes/` 加载。

当前模型是轮足双足（wheel-biped）结构：
- 每腿 4 个关节：`abad`、`hip`、`knee`、`wheel`
- 左右轮碰撞体命名为 `L_foot` / `R_foot`（用于接触传感器）
- 机身 body 名称统一为 `base`（用于 viewer / reward / sensor）

## 关节命名约定

每条腿 4 个关节，共 8 个驱动 DOF。命名规则如下（必须在 XML 和 Python 里保持一致）：

| 缩写 | 含义 | 物理意义 |
|---|---|---|
| `abad` | 外展/内收关节 | 腿左右摆 |
| `hip` | 髋关节 | 腿前后摆 |
| `knee` | 膝关节 | 小腿屈伸 |
| `wheel` | 轮关节 | 车轮滚动 |

完整关节名示例：`abad_L_Joint`、`hip_R_Joint`、`wheel_L_Joint`。

## 用 URDF 替换骨架的步骤

拿到 TRON2 Pro 的 URDF 后，按顺序完成：

1. **安装转换工具**：`pip install urdf2mjcf`（或使用 MuJoCo 自带的 `compile` 命令）。
2. **执行转换**：把 URDF 转成 MJCF，得到 `.xml` 和 mesh 文件。
3. **对齐命名**：保证 joint 名称仍匹配 `abad/hip/knee/wheel` 这四类 regex。
4. **拷贝资源**：把 STL 放到 `src/WF_TRON1A/meshes/`，并在 XML `meshdir` 对齐路径。
5. **替换 XML**：更新 `src/tron_pro_velocity/tron_pro/xmls/tron_pro.xml` 内容。
6. **调整 constants**：更新 `tron_pro_constants.py` 里的 `LEG_EFFORT_LIMIT` / `WHEEL_EFFORT_LIMIT` / `ARMATURE`。
7. **视觉检查**：运行 `uv run python -c "from tron_pro_velocity.tron_pro.tron_pro_constants import get_spec; print(get_spec().compile().nq)"`。

## 使用方法

```powershell
# 健全性检查：让骨架版机器人在零动作下倒下（验证流程通畅）
uv run play Mjlab-Velocity-Flat-Tron-Pro --agent zero

# 训练（数据无意义，只验证 pipeline）
$env:CUDA_VISIBLE_DEVICES="0"
uv run train Mjlab-Velocity-Flat-Tron-Pro `
  --env.scene.num-envs 2048 `
  --agent.max-iterations 500

# 当前项目仅保留 flat 任务
# 如果你后续需要 rough，请先在 __init__.py 重新注册 rough task_id
```

## 两个任务的差异对照

| 维度 | ANYmal C | TRON2 Pro |
|---|---|---|
| 形态 | 四足 | 双足 |
| 脚数 | 4（LF/RF/LH/RH） | 2（L/R） |
| 每条腿 DOF | 3（HAA/HFE/KFE） | 4（abad/hip/knee/wheel） |
| 总驱动 DOF | 12 | 8 |
| 初始高度 | 0.54 m | 0.85 m（WF_TRON1A 实模） |
| 平衡难度 | 静稳定（支撑面大） | 动稳定（一直需要主动平衡） |

## 下一步计划

- [ ] 用短训练验证轮足任务稳定性（flat）
- [ ] 根据训练曲线微调轮关节与腿关节的动作尺度
- [ ] 如果需要，把 wheel 关节从 position action 改为 velocity action

