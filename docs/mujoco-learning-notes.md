# MuJoCo 学习笔记

> 记录本机 MuJoCo 环境的实测验证过程：环境自检 → 图形界面上手 → RL 训练链路。
> 所有结论均为 2026-10-14 在本机实际操作验证所得，命令可直接复制使用。
> 配套阅读：[mujoco-guide.md](mujoco-guide.md)（MuJoCo 使用手册）。

---

## 目录

1. [环境验证](#1-环境验证)
2. [打开图形化界面的方式](#2-打开图形化界面的方式)
3. [viewer 窗口操作指南](#3-viewer-窗口操作指南)
4. [RL 训练环境搭建](#4-rl-训练环境搭建)
5. [常用命令速查](#5-常用命令速查)

---

## 1. 环境验证

### 1.1 环境构成

| 项目 | 值 |
|---|---|
| MuJoCo | 3.15.0（装在 `.venv` 内） |
| Python | 3.13.9（uv 0.9.5 管理的解释器） |
| 虚拟环境 | `./.venv/`（`include-system-site-packages = false`，与系统完全隔离） |
| 系统环境 | WSL2 (Ubuntu 20.04) + WSLg 图形支持 |
| GPU | NVIDIA T1200 Laptop GPU（WSL 驱动 596.52） |

### 1.2 为什么 Linux 图形窗口能弹在 Windows 桌面上

WSL2 自带 **WSLg** 图形子系统，环境变量已自动配好：

```
DISPLAY=:0                # X11 协议入口
WAYLAND_DISPLAY=wayland-0 # Wayland 协议入口
```

MuJoCo 的交互窗口基于 GLFW/X11，直接通过 WSLg 显示在 Windows 桌面，
无需 VcXsrv 等第三方 X Server，也无需任何额外配置。

### 1.3 验证结果汇总

| 验证项 | 方法 | 结果 |
|---|---|---|
| 仿真步进 | `verify_mujoco.py`（建模 + mj_step 1000 步 + 物理断言） | ✅ 通过 |
| EGL 离屏渲染 | `MUJOCO_GL=egl` + `mujoco.Renderer` | ✅ 通过 |
| GLFW GL 上下文 | `MUJOCO_GL=glfw` + `mujoco.Renderer` | ✅ 通过 |
| 交互式 viewer | `mujoco.viewer.launch(model, data)` | ✅ 窗口正常弹出 |

自检脚本随时可跑：

```bash
.venv/bin/python verify_mujoco.py
```

### 1.4 已知警告（可忽略）

EGL 渲染时输出：

```
libEGL warning: failed to open /dev/dri/renderD128: Permission denied
```

这是 WSL 下直渲染设备节点的权限提示，**渲染实际正常完成**，不影响任何功能。
介意的话可把用户加入 `render` 组（`sudo usermod -aG render $USER`），通常没必要。

---

## 2. 打开图形化界面的方式

> 补充一个底层事实：Python 包里的 `mujoco.viewer` 就是**官方 simulate 应用**
> 的封装（源码 `from mujoco import _simulate`），界面、面板、快捷键与官方文档完全一致。

### 2.1 命令行直接打开（最快捷）

```bash
# 打开指定模型（参数名是 --mjcf，已从 viewer.py 源码确认）
python3 -m mujoco.viewer --mjcf examples/models/pendulum.xml

# 不带参数：打开空场景，之后可 File → Open 或拖拽 XML 进窗口
python3 -m mujoco.viewer
```

### 2.2 脚本内调用

```python
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("examples/models/pendulum.xml")
data = mujoco.MjData(model)

# 方式一：阻塞式，窗口关闭前脚本停在这一行
mujoco.viewer.launch(model, data)
# mujoco.viewer.launch_from_path("xxx.xml")   # 等价于命令行方式
```

```python
# 方式二：非阻塞式，适合自己写控制循环、边跑边看
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)   # 自己的仿真/控制逻辑
        viewer.sync()                 # 同步状态到窗口（也会把 GUI 输入传回 data）
```

### 2.3 Gymnasium 训练环境可视化

```python
import gymnasium as gym

env = gym.make("InvertedPendulum-v5", render_mode="human")  # 弹窗实时观看
obs, _ = env.reset()
while True:
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    if terminated or truncated:
        break
```

---

## 3. viewer 窗口操作指南

### 3.1 鼠标交互

| 操作 | 效果 |
|---|---|
| 左键拖动 | 旋转视角 |
| 右键拖动 | 平移视角 |
| 滚轮 | 缩放 |
| 双击物体 | 选中（施力/查看信息的前提） |

**对选中物体施加扰动**（交互的灵魂功能）：

| 键位 | 效果 |
|---|---|
| `Ctrl` + 左键拖动 | 施加**力矩**（旋转物体） |
| `Ctrl` + 右键拖动 | 竖直平面 (x,z) 施加**平移力** |
| `Ctrl` + `Shift` + 右键拖动 | 水平面 (x,y) 施加平移力 |

### 3.2 左侧面板详解（9 个）

各面板可用 `Alt + 首字母` 快速展开/折叠（F=File, O=Option, S=Simulation,
W=Watch, P=Physics, R=Rendering, V=Visualization, G=Group enable, L=Logging）。

| 面板 | 作用 | 常用项 |
|---|---|---|
| **File** | 模型文件操作 | Open 加载模型（支持拖拽 XML 进窗口）、Save xml / Save mjb、Screenshot（Ctrl+P） |
| **Option** | ⭐ 运行时改物理参数，立即生效，不用改 XML | Timestep 步长、**Gravity 重力**（改成 0 看物体飘起、1.62 模拟月球）、Integrator 积分器、Solver/Iterations 求解器 |
| **Simulation** | 仿真控制 | Pause（空格）、Reset（Backspace）、Reload 重载、Sync 实时同步开关、**Slowdown 慢动作**（100%→0.1%）、History 历史回放滑条 |
| **Watch** | 实时监视任意变量 | 输入 `time`、`qpos`、`energy`、`sensor(传感器名)` 看实时数值 |
| **Physics** | ⭐ 物理量可视化叠加（调试模型神器） | Contact point 接触点、Contact force 接触力箭头、Inertia 惯性盒、Convex hull 凸包、Center of Mass（M） |
| **Rendering** | OpenGL 画面特效 | Wireframe 线框（W）、Shadow 阴影（S）、Reflection（R）、Skybox（K）、Fog（G） |
| **Visualization** | 显示元素与相机 | Label 标签（F7 循环）、Frame 坐标架（F6 循环）、各元素显示尺寸、相机参数（FOV 等） |
| **Group enable** | 按组显隐物体 | geom 组 0-5（**数字键 0~5 直接切**）、site 组（Shift+0~5）；典型用法：隐藏装饰壳只看碰撞体 |
| **Logging** | 日志系统配置 | 错误/警告输出到控制台还是文件（默认 `MUJOCO_LOG.TXT`）、info 主题开关；排查仿真警告用 |

> 注意：Option 里改的参数在 Reset 后会恢复 XML 设定值。

### 3.3 右侧面板（模型相关）

- **Joint**：关节状态滑条
- **Control**：⭐ 电机控制滑条——每个 actuator 一根（如摆锤模型的 motor，范围由 XML 的
  `ctrlrange` 决定），拖动即施加控制
- **Equality / Tendon**：等式约束、腱绳相关

`Tab` / `Shift+Tab` 切换左右面板显示。

### 3.4 功能键与快捷键

| 键 | 功能 |
|---|---|
| F1 | 帮助浮层（快捷键速查） |
| F2 | 信息浮层（模型统计） |
| F3 | 性能分析器 |
| **F4** | ⭐ **传感器实时曲线图**（模型里定义了 sensor 的直接画曲线） |
| F5 | 全屏 |
| `Ctrl+P` | 截图 |
| `Ctrl+M` / `Ctrl+D` | 导出完整模型/数据结构到文本（MJMODEL.TXT / MJDATA.TXT，学 MJCF 结构神器） |
| 空格 / Backspace | 暂停 / 重置；暂停后 `→` 单步 |
| `[` `]` | 循环切换模型内定义的固定相机 |
| 数字键 0~5 | 开关 geom group 0~5 |

小技巧：**长按右键悬停在面板控件上会显示它自己的快捷键**；双击面板标题展开/折叠全部面板。

### 3.5 新手上手路线（以 examples/models/pendulum.xml 为例）

1. 鼠标转视角、缩放，找好机位
2. 空格暂停，`→` 单步，感受仿真节奏
3. 双击绿杆选中 → `Ctrl+左键拖动` 施加力矩让摆锤晃起来
4. 右侧 Control 面板拖 `motor` 滑条（-5~+5），用电机驱动摆锤
5. Watch 面板输入 `sensor(angle_sensor)` 看关节角；按 F4 看角度曲线
6. Option 里把 Gravity 改成 0，再看 Physics 里勾选的 Contact point
7. File → Open 换个模型继续玩（推荐 `examples/models/ball.xml` 拖红球）

---

## 4. RL 训练环境搭建

### 4.1 已安装组件

```bash
pip install "gymnasium[mujoco]" stable-baselines3 tensorboard
```

> 注意：stable-baselines3 会自动把 PyTorch 作为依赖装上（含全套 CUDA 组件，
> 约 2-3 GB，网络慢时耐心等待）。

| 组件 | 版本 | 说明 |
|---|---|---|
| gymnasium | 1.4.0 | RL 环境接口，含 MuJoCo 官方环境（HalfCheetah、Ant、Humanoid 等） |
| torch | 2.14.1+cu130 | PyTorch，CUDA 13.0 |
| stable-baselines3 | 2.9.0 | PPO/SAC/TD3 等现成算法 |
| tensorboard | 2.21.0 | 训练曲线可视化 |

GPU 验证（实测输出）：

```
CUDA 可用 : True
GPU       : NVIDIA T1200 Laptop GPU
CUDA 版本 : 13.0
```

### 4.2 包都装在哪？

**全部装进 `.venv`，不碰系统目录，无需 sudo。** 判别方法：

```bash
pip -V          # 显示 pip 属于哪个 site-packages
which python3   # venv 激活时指向 .venv/bin/python3
```

- `.venv/pyvenv.cfg` 中 `include-system-site-packages = false` → 与系统 Python 完全隔离
- venv 由 uv 创建；安装可用普通 `pip install`，也可用更快的 `uv pip install`
- ⚠️ `pip list` 里看到的 ROS 2 (Galactic) 包来自环境变量
  `PYTHONPATH=/opt/ros/galactic/lib/python3.8/site-packages`，并没有装进 venv，
  且是 Python 3.8 编译的，在 3.13 下无法导入。做 MuJoCo + ROS 2 联合仿真
  需要单独建 Python 3.8 环境

### 4.3 训练实测（examples/05_train_rl_ppo.py）

PPO 在 MuJoCo 倒立摆（InvertedPendulum-v5）上训练 5 万步：

```
训练完成: 50,000 步用时 112.7s (444 步/秒)
评估 10 局平均回报: 1000.0 (满分！倒立摆全程稳稳立住)
模型已保存: examples/output/ppo_inverted_pendulum.zip
```

运行方式：

```bash
.venv/bin/python examples/05_train_rl_ppo.py
```

### 4.4 经验与坑

1. **MlpPolicy 小网络用 CPU 往往更快**：SB3 会发警告提示 GPU 利用率低。
   小型全连接网络的瓶颈在数据搬运而非计算；GPU 优势要在 CNN 策略
   （图像输入）或大网络上才体现。切换方法：

   ```python
   model = PPO("MlpPolicy", env, device="cpu")
   ```

2. **长时间训练用 TensorBoard 看曲线**：

   ```bash
   tensorboard --logdir examples/output/   # 浏览器打开 http://localhost:6006
   ```

3. **装完新包记得同步 `requirements.txt`**（`pip freeze > requirements.txt`），
   注意ROS 包会混进来，建议用
   `pip freeze --path .venv/lib/python3.13/site-packages > requirements.txt`
   只导出 venv 内的包。

---

## 5. 常用命令速查

```bash
# ── 环境自检 ────────────────────────────────────────────
.venv/bin/python verify_mujoco.py              # 仿真+渲染自检

# ── 图形界面 ────────────────────────────────────────────
python3 -m mujoco.viewer --mjcf examples/models/pendulum.xml   # 打开摆锤
python3 -m mujoco.viewer --mjcf examples/models/ball.xml       # 打开红球场景
python3 -m mujoco.viewer                                        # 空场景

# ── RL 训练 ─────────────────────────────────────────────
.venv/bin/python examples/05_train_rl_ppo.py   # PPO 训练倒立摆（冒烟测试）
tensorboard --logdir examples/output/          # 看训练曲线

# ── 导出的模型怎么用 ────────────────────────────────────
# examples/output/ppo_inverted_pendulum.zip
# 用 SB3 加载: PPO.load("examples/output/ppo_inverted_pendulum")
```

值得一看的内置模型（Gymnasium 自带，可用 viewer 直接打开）：

```
.venv/lib/python3.13/site-packages/gymnasium/envs/mujoco/assets/humanoid.xml   # 人形机器人
.venv/lib/python3.13/site-packages/gymnasium/envs/mujoco/assets/ant.xml        # 四足蚂蚁（8 个电机滑条可玩）
.venv/lib/python3.13/site-packages/gymnasium/envs/mujoco/assets/half_cheetah.xml
```
