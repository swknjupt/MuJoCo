"""带执行器的模型：对单摆施加正弦力矩控制，并读取传感器。

运行: .venv/bin/python examples/03_actuator_control.py
"""

from pathlib import Path

import mujoco
import numpy as np

xml_path = Path(__file__).parent / "models" / "pendulum.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

nu = model.nu
print(f"执行器数量 nu={nu}, 传感器数量 nsensor={model.nsensor}")
print(f"关节限位 range = {model.jnt_range[0]} rad")

mujoco.mj_resetData(model, data)
print("\n t(s)   ctrl     角度(rad)   角速度(rad/s)  末端位置(x, z)")
for step in range(1500):
    target = 3.0 * np.sin(2.0 * np.pi * 0.5 * data.time)
    data.ctrl[0] = target
    mujoco.mj_step(model, data)
    if step % 250 == 0:
        angle = data.sensordata[0]
        angvel = data.sensordata[1]
        tip_x, _, tip_z = data.sensordata[2:5]
        print(f"{data.time:5.2f}  {data.ctrl[0]:+.3f}   {angle:+.4f}    "
              f"{angvel:+.4f}      ({tip_x:+.3f}, {tip_z:+.3f})")

print(f"\n最终关节角度 = {data.sensordata[0]:+.4f} rad")
print(f"提示: ctrl 通过 data.ctrl 施加, 传感器通过 data.sensordata 读取")
