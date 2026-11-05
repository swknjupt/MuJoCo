# MuJoCo 使用指南

> 本文档针对当前目录（`~/Document/kevin_workSpace/MUJOCO`）的安装环境编写，
> 所有代码与命令均已在本地实际验证通过。

---

## 目录

1. [MuJoCo 简介](#1-mujoco-简介)
2. [本机环境说明](#2-本机环境说明)
3. [快速开始](#3-快速开始)
4. [核心概念](#4-核心概念)
5. [MJCF 模型编写](#5-mjcf-模型编写)
6. [可视化与渲染](#6-可视化与渲染)
7. [控制与传感](#7-控制与传感)
8. [常用 API 速查表](#8-常用-api-速查表)
9. [常见问题 FAQ](#9-常见问题-faq)
10. [官方资源](#10-官方资源)

---

## 1. MuJoCo 简介

**MuJoCo**（Multi-Joint dynamics with Contact）是 Google DeepMind 开源的物理仿真引擎，
专注于接触动力学的高精度、高速仿真，广泛用于：

- 机器人学与运动规划研究
- 强化学习（RL）训练环境（Gymnasium、dm_control 的底层引擎）
- 生物力学、计算机动画等

当前安装版本：**3.15.0**（2026 年最新正式版）。

---

## 2. 本机环境说明

### 2.1 环境构成

| 项目 | 值 |
|---|---|
| MuJoCo 版本 | 3.15.0 |
| Python | 3.13.9（uv 管理的独立版本，与系统 Python 3.8 完全隔离） |
| 虚拟环境 | `./.venv/`（当前目录下） |
| 系统环境 | Ubuntu 20.04，glibc 2.31 |

### 2.2 目录结构

```
MUJOCO/
├── .venv/                  # Python 虚拟环境（mujoco 安装于此）
├── docs/
│   └── mujoco-guide.md     # 本文档
├── examples/
│   ├── 01_hello_mujoco.py      # 最小仿真示例
│   ├── 02_load_model.py        # 加载 XML 模型文件
│   ├── 03_actuator_control.py  # 执行器控制与传感器
│   ├── 04_render_offscreen.py  # 离屏渲染保存 PNG
│   ├── models/
│   │   ├── ball.xml            # 平面 + 小球 + 方块场景
│   │   └── pendulum.xml        # 单摆（铰链关节 + 电机 + 传感器）
│   └── output/                 # 渲染输出目录
├── requirements.txt        # 依赖固化
└── verify_mujoco.py        # 安装验证脚本
```

### 2.3 使用虚拟环境

```bash
# 方式一：激活后使用（推荐日常开发）
source .venv/bin/activate
python your_script.py        # 此时的 python 即 3.13.9
deactivate                   # 退出虚拟环境

# 方式二：不激活，直接使用完整路径
.venv/bin/python your_script.py

# 安装新的包（两种方式等价）
uv pip install --python .venv/bin/python <包名>   # 需要已安装 uv
# 或激活后: pip install <包名>
```

### 2.4 环境重建（如删除或迁移后）

```bash
uv venv --python 3.13 --seed .venv      # 创建虚拟环境（Python 自动下载管理）
uv pip install --python .venv/bin/python -r requirements.txt
```

---

## 3. 快速开始

最小可运行示例（对应 `examples/01_hello_mujoco.py`）：

```python
import mujoco

# 1. 从字符串构建模型（也可以从 .xml 文件构建）
xml = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="sphere" type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# 2. 仿真循环：每步调用 mj_step
for i in range(500):
    mujoco.mj_step(model, data)

# 3. 读取状态
print(data.time)      # 仿真时间 1.0s
print(data.qpos)      # 广义坐标位置（小球 x y z + 四元数姿态）
print(data.qvel)      # 广义速度
```

实测输出（小球从 1m 高度落下，2 倍半径高度即静止在地面）：

```
MuJoCo 版本: 3.15.0
自由度: nq=7, nv=6, timestep=0.002s
t=0.200s  高度 z=+0.8018m  垂直速度 vz=-1.9620m/s
...
仿真结束: 小球停在高度 z=0.0996m (球半径 0.1m)
```

---

## 4. 核心概念

### 4.1 MjModel（静态模型）与 MjData（动态状态）

| 对象 | 含义 | 生命周期 |
|---|---|---|
| `MjModel` | 模型的**静态描述**：自由度、质量、摩擦、步长等编译结果 | 加载后不变 |
| `MjData` | 仿真的**动态状态**：时间、位置、速度、控制、传感器读数 | 每步更新 |

```python
model = mujoco.MjModel.from_xml_path("model.xml")
data = mujoco.MjData(model)   # data 必须绑定到对应 model
```

### 4.2 关键函数

- **`mj_step(model, data)`**：推进一个完整仿真步（前向动力学 + 积分）。
- **`mj_forward(model, data)`**：只计算前向动力学（不积分），
  用于在修改状态/控制后刷新派生量（如 `data.xpos`、传感器），常在渲染前调用。
- **`mj_resetData(model, data)`**：状态归零，重新开始仿真。

### 4.3 状态向量

自由浮动刚体（`freejoint`）有 7 个 qpos（3 位置 + 4 四元数）和 6 个 qvel：

```
qpos = [x, y, z, qw, qx, qy, qz]
```

时间步长默认 0.002s，由模型中 `<option timestep="0.002"/>` 控制，读取用
`model.opt.timestep`（注意：不是 `model.timestep`）。

---

## 5. MJCF 模型编写

MJCF 是 MuJoCo 原生 XML 格式。完整参考见官方文档
[XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)。

### 5.1 基本骨架

```xml
<mujoco model="my_model">
  <option timestep="0.002" gravity="0 0 -9.81"/>   <!-- 全局选项 -->

  <worldbody>                       <!-- 世界，所有物体在这里 -->
    <light name="top" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1"/>  <!-- 静态几何体，无 body 包裹 -->

    <body name="pole" pos="0 0 1.2">      <!-- 刚体 -->
      <joint name="hinge" type="hinge" axis="0 1 0"/>  <!-- 关节（连接到父体） -->
      <geom name="rod" type="capsule" fromto="0 0 0  0 0 -0.8" size="0.03" mass="1"/>
      <site name="tip" pos="0 0 -0.8" size="0.02"/>    <!-- 标记点，用于传感/观测 -->
    </body>
  </worldbody>

  <actuator>                        <!-- 执行器 -->
    <motor name="motor" joint="hinge" gear="1" ctrlrange="-5 5"/>
  </actuator>

  <sensor>                          <!-- 传感器 -->
    <jointpos name="angle" joint="hinge"/>
  </sensor>
</mujoco>
```

### 5.2 重要约定

- **角度单位默认是"度"**（`<compiler angle="degree"/>` 是默认值）。
  `range="-90 90"` 表示 ±90°（≈ ±1.57 rad）。想用弧度需显式写
  `<compiler angle="radian"/>`。这是新手最常见的坑！
- 不放在任何 `<body>` 内的 `<geom>`（如地面）是**静态**的，固定在世界上。
- `<body>` 之间必须通过 `<joint>` 连接，否则两个 body 会焊死在一起；
  自由刚体用 `<freejoint/>`。
- 常用几何体：`plane` / `sphere` / `capsule` / `box` / `cylinder` / `mesh`。

### 5.3 从文件加载

```python
model = mujoco.MjModel.from_xml_path("examples/models/ball.xml")
```

### 5.4 现成模型库：MuJoCo Menagerie

[Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
提供数十种高质量机器人模型（Franka 机械臂、宇树四足/人形、Shadow Hand、
ANYmal、Cassie 等），clone 后直接 `from_xml_path` 加载即可。

---

## 6. 可视化与渲染

MuJoCo 的渲染后端由环境变量 `MUJOCO_GL` 决定，**必须在 `import mujoco` 之前设置**：

```python
import os
os.environ["MUJOCO_GL"] = "egl"   # 可选: egl / osmesa / glfw
import mujoco                     # 一定要在这之后 import
```

| 后端 | 场景 | 依赖 |
|---|---|---|
| `egl` | 无窗口离屏渲染（服务器/训练时保存视频） | libEGL（本机已具备 ✓） |
| `osmesa` | 无 GPU 的纯 CPU 离屏渲染 | `sudo apt install libosmesa6` |
| `glfw` | 有显示器的交互窗口（viewer 自动使用） | libGL + 显示器（本机已具备 ✓） |

### 6.1 交互式查看器（本机可用）

本机存在图形桌面（`DISPLAY=:0`），可直接弹出交互窗口：

```python
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("examples/models/pendulum.xml")
data = mujoco.MjData(model)

# 阻塞式：打开独立窗口
mujoco.viewer.launch(model, data)
```

更常用的是**被动模式**——窗口与你的仿真循环并行，代码可实时修改控制：

```python
import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("examples/models/pendulum.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[0] = 3.0 * np.sin(2 * np.pi * 0.5 * data.time)
        mujoco.mj_step(model, data)
        viewer.sync()   # 把最新状态推给窗口
```

命令行快速查看模型文件：`.venv/bin/python -m mujoco.viewer --mjcf=examples/models/ball.xml`

### 6.2 离屏渲染保存图片（对应 `examples/04_render_offscreen.py`）

```python
import os
os.environ["MUJOCO_GL"] = "egl"

import mujoco

model = mujoco.MjModel.from_xml_path("examples/models/ball.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data)          # 可传 camera=相机名或id
img = renderer.render()              # numpy 数组 (480, 640, 3) RGB
renderer.close()

from PIL import Image
Image.fromarray(img).save("frame.png")
```

已实测生成 `examples/output/frame_*.png`，场景渲染正常（含光照与阴影）。

### 6.3 合成视频（思路）

循环中每 N 步渲染一帧存入列表，最后用 `imageio`/`opencv` 写成 mp4：

```python
# pip install imageio[ffmpeg] 后
import imageio
with imageio.get_writer("sim.mp4", fps=30) as w:
    for frame in frames:
        w.append_data(frame)
```

---

## 7. 控制与传感

对应 `examples/03_actuator_control.py` 与 `examples/models/pendulum.xml`。

### 7.1 施加控制

执行器在 MJCF 的 `<actuator>` 中定义，仿真循环里直接写 `data.ctrl`：

```python
data.ctrl[0] = torque   # 数值范围受 ctrlrange 限制
mujoco.mj_step(model, data)
```

### 7.2 读取传感器

传感器在 `<sensor>` 中定义，`mj_step`/`mj_forward` 后从 `data.sensordata` 按定义顺序读取：

```python
angle  = data.sensordata[0]   # jointpos
angvel = data.sensordata[1]   # jointvel
tip    = data.sensordata[2:5] # framepos (x, y, z)
```

实测输出（正弦力矩驱动单摆，幅值 3 N·m）：

```
 t(s)   ctrl     角度(rad)   角速度(rad/s)  末端位置(x, z)
 0.00  +0.000   +0.0000    +0.0000      (+0.000, +0.400)
 0.50  +3.000   +0.6074    +2.7057      (-0.457, +0.543)
 ...
```

### 7.3 名称与 ID 互查

```python
jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
```

`mjtObj` 枚举包含 `mjOBJ_BODY / mjOBJ_JOINT / mjOBJ_GEOM / mjOBJ_SITE /
mjOBJ_ACTUATOR / mjOBJ_SENSOR` 等。

### 7.4 便捷访问器

`data` 支持按名称索引（返回命名切片视图）：

```python
data.body("red_ball").xpos   # 该刚体在世界系下的位置
data.joint("hinge").qpos     # 该关节的 qpos 切片
```

---

## 8. 常用 API 速查表

### 模型构建与查询

| API | 说明 |
|---|---|
| `mujoco.MjModel.from_xml_string(xml)` | 从字符串构建模型 |
| `mujoco.MjModel.from_xml_path(path)` | 从 XML 文件构建模型 |
| `model.nq / nv / nu / nbody / ngeom` | 位置维数 / 速度维数 / 执行器数 / 刚体数 / 几何体数 |
| `model.opt.timestep` | 仿真步长 |
| `model.opt.gravity` | 重力向量 |
| `model.jnt_range` | 关节限位（弧度存储） |

### 仿真循环

| API | 说明 |
|---|---|
| `mujoco.mj_step(model, data)` | 推进一个仿真步 |
| `mujoco.mj_forward(model, data)` | 仅前向动力学，刷新派生量 |
| `mujoco.mj_resetData(model, data)` | 状态清零 |
| `data.time / qpos / qvel / ctrl` | 时间 / 位置 / 速度 / 控制 |
| `data.sensordata` | 传感器读数 |
| `data.xpos` | 所有刚体世界坐标 (nbody × 3) |

### 名称工具

| API | 说明 |
|---|---|
| `mujoco.mj_name2id(model, type, name)` | 名称 → ID |
| `mujoco.mj_id2name(model, type, id)` | ID → 名称 |
| `data.body("名") / joint("名") / geom("名")` | 按名称访问 |

### 渲染与查看

| API | 说明 |
|---|---|
| `mujoco.Renderer(model, h, w)` | 离屏渲染器 |
| `renderer.update_scene(data)` | 更新场景（可加 `camera=`） |
| `renderer.render()` | 返回 RGB numpy 数组 |
| `mujoco.viewer.launch(model, data)` | 阻塞式交互窗口 |
| `mujoco.viewer.launch_passive(model, data)` | 非阻塞窗口 + `viewer.sync()` |

---

## 9. 常见问题 FAQ

### Q1: `MUJOCO_GL` 设置了但不生效？

`MUJOCO_GL` 在 `import mujoco` 时读取一次。若在 import 之后才设置（或
import 后再改环境变量），不会生效。**务必在 import mujoco 之前设置**。

### Q2: 渲染报错 / 黑屏 / 无后端可用？

- 服务器无显示器：用 `MUJOCO_GL=egl`（优先）或 `osmesa`
  （需 `sudo apt install libosmesa6`）。
- 本机已验证：EGL 与 GLFW 均可用。EGL 走 Mesa 软件渲染时会出现
  `libEGL warning: failed to open /dev/dri/renderD128: Permission denied`
  警告——**无害**，只是提示 GPU 加速不可用、退回 CPU 软渲染。
- 想启用 GPU 渲染加速：`sudo usermod -aG render,video $USER` 后重新登录。

### Q3: 关节限位和预想差很远？

MJCF 的角度默认单位是**度**。`range="-1.57 1.57"` 实际是 ±1.57°。
需要弧度请写 `<compiler angle="radian"/>`。

### Q4: `model.timestep` 报 AttributeError？

步长在 `model.opt.timestep`，`MjModel` 没有顶层 `timestep` 属性。

### Q5: 怎么安装其他包（如 gymnasium、imageio）？

```bash
uv pip install --python .venv/bin/python "gymnasium[mujoco]"
# 或激活 venv 后: pip install <包名>
```

### Q6: 怎么确认环境是否完好？

```bash
.venv/bin/python verify_mujoco.py   # 建模 + 仿真 + 渲染自检
```

---

## 10. 官方资源

| 资源 | 链接 |
|---|---|
| 官方文档（API / XML Reference） | https://mujoco.readthedocs.io/en/stable/ |
| GitHub 仓库 | https://github.com/google-deepmind/mujoco |
| 官方 Python 入门教程（Colab 笔记本） | https://github.com/google-deepmind/mujoco/blob/main/python/tutorial.ipynb |
| Menagerie 机器人模型库 | https://github.com/google-deepmind/mujoco_menagerie |
| dm_control（DeepMind 控制任务库） | https://github.com/google-deepmind/dm_control |
| Gymnasium（RL 环境标准接口） | https://gymnasium.farama.org/ |
