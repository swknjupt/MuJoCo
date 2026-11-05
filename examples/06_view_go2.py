"""查看 Unitree Go2 机械狗：加载 Menagerie 官方模型并启动交互查看器。

模型来源: https://github.com/google-deepmind/mujoco_menagerie (unitree_go2)
查看操作: 左键旋转 / 滚轮缩放 / 双击选中刚体 / Backspace 重置 / Space 暂停
站立姿态: 左面板 Simulation -> Key 滑块选 0 -> Load key (home 关键帧)
注意: macOS 上需用 mjpython 运行; Linux 上直接 python 运行
运行: .venv/bin/python examples/06_view_go2.py
"""

from pathlib import Path

import mujoco

# 模型在仓库根目录的 mujoco_menagerie/unitree_go2/ 下
xml_path = Path(__file__).parent.parent / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print(f"模型文件: {xml_path}")
print(f"自由度: nq={model.nq}, nv={model.nv}, 执行器 nu={model.nu}, 刚体 nbody={model.nbody}")

print("\n--- 关节 (joints) ---")
for jid in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    print(f"  [{jid}] {name:<10} 类型={model.jnt_type[jid]}")

print("\n--- 执行器 (actuators) ---")
for aid in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
    print(f"  [{aid}] {name:<10} ctrl范围={model.actuator_ctrlrange[aid]}")

# 加载 "home" 站立关键帧, 让 Go2 以标准站姿出现
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    key_name = model.key(0).name
    base_z = data.body("base").xpos[2]
    print(f"\n已加载关键帧 '{key_name}': 躯干离地高度 z = {base_z:.3f} m")

print("\n启动交互查看器 (关闭窗口后脚本结束)...")
mujoco.viewer.launch(model, data)
print("查看器已关闭")
