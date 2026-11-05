"""从本地 XML 文件加载模型，并查看模型结构与常用属性。

运行: .venv/bin/python examples/02_load_model.py
"""

from pathlib import Path

import mujoco

xml_path = Path(__file__).parent / "models" / "ball.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))

print(f"模型文件: {xml_path.name}")
print(f"仿真步长 timestep = {model.opt.timestep}s")
print(f"重力 = {model.opt.gravity}")
print(f"自由度: nq={model.nq}, nv={model.nv}, 刚体数 nbody={model.nbody}, "
      f"几何体数 ngeom={model.ngeom}")

print("\n--- 关节 (joints) ---")
for jid in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    jtype = model.jnt_type[jid]
    print(f"  [{jid}] {name:<12} type={jtype}  qpos地址={model.jnt_qposadr[jid]}")

print("\n--- 刚体 (bodies) ---")
for bid in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    mass = model.body_mass[bid]
    print(f"  [{bid}] {name:<12} 质量={mass:.3f}kg")

print("\n--- 几何体 (geoms) ---")
for gid in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    gtype = model.geom_type[gid]
    print(f"  [{gid}] {name:<12} type={gtype}")

data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
print(f"\nmj_forward 后红球初始位置 = {data.body('red_ball').xpos}")
