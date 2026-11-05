# 从 Isaac Sim/Lab 迁移 RL 策略到 MuJoCo —— 资源获取清单

> 目标：把在 Isaac Lab（rsl_rl）中训练好的 Go2 行走策略，迁移到 MuJoCo 中运行（sim-to-sim），最终在 MuJoCo 查看器里看到狗行走。
>
> 原理：策略网络只是一个数学函数，权重原封不动，换的只是执行它的物理世界。但**接口必须与训练时完全一致**。

## 一、需要获取的文件

训练产物位于训练机的：

```
logs/rsl_rl/<实验名>/<日期_时间>/
```

### 必需（3 个）

| # | 文件 | 作用 |
|---|---|---|
| 1 | `model_xxxxx.pt` | 策略权重（actor_critic state_dict）。拿**步数最大**的 checkpoint（如 `model_2999.pt`）或 `model_final.pt` |
| 2 | `params/env.yaml` | **最关键**。包含：观测项与顺序、观测缩放、`action_scale`、默认关节角、PD 增益（stiffness/damping）、力矩限幅、`decimation`、`sim.dt`、动作裁剪范围 |
| 3 | `params/agent.yaml` | 网络结构：隐藏层尺寸（如 [512, 256]）、激活函数（ELU/ReLU）、策略类别（ActorCritic 或 **ActorCriticRecurrent**，后者迁移方式不同） |

### 如存在，一并获取（2 个）

| # | 文件 | 说明 |
|---|---|---|
| 4 | `params/wrap.yaml` | 新版 Isaac Lab 才有，观测归一化包装配置 |
| 5 | `empirical_normalization.pt` | 训练若启用观测归一化则**必须**配套加载 |

### 锦上添花（可无）

| # | 文件 | 说明 |
|---|---|---|
| 6 | `policy.pt`（JIT）/ `.onnx` | 已导出的部署格式，与 #1 二选一 |
| 7 | 任务名 + Isaac Lab 版本号 | 如 `Isaac-Velocity-Rough-Unitree-Go2-v0`；用于确认观测拼接顺序的代码行为 |

### ⚠️ 特殊情况

自定义环境（非官方任务）时，还需提供**自定义任务的源码目录**（env 定义、观测/奖励函数 .py）。

### 不需要

tensorboard 事件、wandb 日志、`gitdiff.pkl`、训练录像 mp4。

## 二、文件 → 迁移接口的对应关系

MuJoCo 侧（`10_go2_sim2sim.py`）需要恢复训练时的 6 个接口，全部来自上述文件：

| 接口 | 来源 |
|---|---|
| 观测布局与缩放 | `env.yaml`（观测项定义）+ 任务代码 |
| 动作语义（默认角 + action_scale） | `env.yaml` |
| PD 增益 + 力矩限幅 | `env.yaml`（stiffness/damping/effort_limit） |
| 控制频率（decimation） | `env.yaml` |
| Actor 网络结构 | `agent.yaml` |
| 观测归一化 | `empirical_normalization.pt` + `wrap.yaml` |

## 三、建议的存放结构

```
examples/weights/
├── model_2999.pt
├── env.yaml
├── agent.yaml
├── wrap.yaml                  （如有）
└── empirical_normalization.pt （如有）
```

## 四、迁移时的已知注意点

1. **关节顺序不一致**：Isaac（legged_gym 体系）张量顺序通常 FR 开头；Menagerie MJCF 执行器顺序 FL 开头 → 必须按**关节名**建立映射表。
2. **默认站姿不一致**：Isaac Go2 默认角（thigh≈0.4, calf≈−0.9）≠ Menagerie `home` 关键帧（0.9, −1.8）→ 观测与动作偏移一律以 **Isaac 配置为准**。
3. **PD 在 MuJoCo 侧手动实现**：Menagerie 执行器是纯力矩电机，`τ = kp·(q_des − q) − kd·q̇` 后 clip 再写入 `data.ctrl`。
4. **物理差异**：PhysX ≠ MuJoCo（接触求解、摩擦），多数策略可直接迁移；个别需微调摩擦或步频。

## 五、验收标准

给速度指令 `(0.6 m/s, 0, 0)`：狗以小跑姿态直线前进，被拖拽后能恢复路线。

## 六、调试症状速查

| 现象 | 最可能原因 |
|---|---|
| 瘫软不动 | 权重未正确加载 / PD 增益错 |
| 疯狂抽搐 | 观测布局或关节顺序错 |
| 站稳但不走 | action_scale / 默认角错，指令未进 obs |
| 走几步摔倒 | 物理差异，微调摩擦 / 步频 |
