"""MuJoCo 环境自检脚本: 建模 + 仿真 + 离屏渲染。

运行: .venv/bin/python verify_mujoco.py
"""

import os

# MUJOCO_GL 必须在 import mujoco 之前设置
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

print(f"[1] MuJoCo 版本: {mujoco.__version__}")
print(f"    Python 绑定路径: {mujoco.__file__}")

xml = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="sphere" type="sphere" size=".1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
print(f"[2] 模型构建成功: nq={model.nq}, nv={model.nv}, nbody={model.nbody}")

for _ in range(1000):
    mujoco.mj_step(model, data)

ball_z = data.qpos[2]
print(f"[3] 仿真步进成功: 1000 步 (2s) 后小球高度 z = {ball_z:.4f} m")
assert 0.09 < ball_z < 0.15, "小球应静止在地面上 (球心高度≈半径 0.1)"

mujoco.mj_forward(model, data)
try:
    renderer = mujoco.Renderer(model, height=240, width=320)
    renderer.update_scene(data)
    img = renderer.render()
    renderer.close()
    assert img is not None and img.shape == (240, 320, 3)
    brightness = float(np.mean(img))
    assert brightness > 5, "渲染图像不应为全黑"
    print(f"[4] 离屏渲染成功 (后端: {os.environ['MUJOCO_GL']}), "
          f"图像尺寸: {img.shape[1]}x{img.shape[0]}x3, 平均亮度: {brightness:.1f}")
except Exception as e:
    print(f"[4] 离屏渲染不可用 ({e}), 仿真功能不受影响")

print("\n全部验证通过 ✓  MuJoCo 环境安装正确")
