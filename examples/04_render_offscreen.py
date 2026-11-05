"""离屏渲染：MUJOCO_GL=egl 渲染 ball.xml 场景并保存 PNG 图片。

注意: MUJOCO_GL 必须在 import mujoco 之前设置!
运行: .venv/bin/python examples/04_render_offscreen.py
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # 可选后端: egl / osmesa / glfw

from pathlib import Path

import mujoco
import numpy as np


def save_png(img: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(img).save(path)


xml_path = Path(__file__).parent / "models" / "ball.xml"
out_dir = Path(__file__).parent / "output"
out_dir.mkdir(exist_ok=True)

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, height=480, width=640)
print(f"渲染后端: {os.environ['MUJOCO_GL']}, 输出目录: {out_dir}/")

for step in range(1000):
    mujoco.mj_step(model, data)
    if step % 250 == 0:
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        img = renderer.render()
        png_path = out_dir / f"frame_{step:04d}.png"
        save_png(img, png_path)
        print(f"  t={data.time:.2f}s  -> {png_path.name}  尺寸={img.shape}")

renderer.close()
print("离屏渲染完成")
