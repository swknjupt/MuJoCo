"""最小示例：用字符串 XML 建模并仿真一个自由落体小球。

运行: .venv/bin/python examples/01_hello_mujoco.py
"""

import mujoco

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

print(f"MuJoCo 版本: {mujoco.__version__}")
print(f"自由度: nq={model.nq}, nv={model.nv}, timestep={model.opt.timestep}s")
print(f"初始位置 qpos = {data.qpos}")

for i in range(500):
    mujoco.mj_step(model, data)
    if i in (99, 199, 299, 399, 499):
        t, z, vz = data.time, data.qpos[2], data.qvel[2]
        print(f"t={t:.3f}s  高度 z={z:+.4f}m  垂直速度 vz={vz:+.4f}m/s")

print(f"仿真结束: 小球停在高度 z={data.qpos[2]:.4f}m (球半径 0.1m)")
