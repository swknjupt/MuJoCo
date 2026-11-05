# AGENTS.md

## Workspace

Personal MuJoCo learning workspace — no git repo, no CI, no lint/test config. Verification means running the scripts. Docs, docstrings, and comments are in Chinese; keep new content consistent.

Reference material (do not duplicate): `docs/mujoco-learning-notes.md` (verified env setup, 2026-10), `docs/mujoco-guide.md` (usage manual), `docs/embodied-robot-learning-roadmap.md` (learning plan).

## Environment

- Python 3.13 venv at `.venv/`, interpreter managed by **uv** (no `pyproject.toml`). Always run scripts as `.venv/bin/python <script>` from the repo root. Install deps with `uv pip install -r requirements.txt` (fully pinned: mujoco 3.15.0, torch 2.x CUDA stack, stable-baselines3, gymnasium).
- Host is **WSL2 (Ubuntu) + WSLg** with NVIDIA GPU; `torch.cuda.is_available()` is True. Interactive `mujoco.viewer.launch()` windows appear on the Windows desktop via WSLg — no extra setup needed (macOS would require `mjpython`, irrelevant here).
- After any dependency or environment change, run the sanity check: `.venv/bin/python verify_mujoco.py` (covers model build → simulate → offscreen render).

## MuJoCo gotchas

- `MUJOCO_GL=egl` must be set **before** `import mujoco` for offscreen rendering — see `verify_mujoco.py` / `examples/04_render_offscreen.py` for the `os.environ.setdefault` pattern. Backends: `egl` (headless), `glfw` (interactive viewer, used automatically).
- Run from the repo root: `examples/05_train_rl_ppo.py` saves its checkpoint to the CWD-relative path `examples/output/…` (scripts 02–04 and 06 use `Path(__file__)`-relative paths and are CWD-independent).
- RL stack is Gymnasium env → SB3 (e.g. `PPO("MlpPolicy", …, device="auto")`) → torch; `InvertedPendulum-v5` is the current smoke-test env.

## Layout

- `examples/01..06_*.py` — numbered, self-contained learning scripts; each docstring states its run command (and viewer controls for 06).
- `examples/models/` — hand-written MJCF XMLs (ball, pendulum); `examples/output/` — generated artifacts only (PNG frames, `.zip` checkpoints).
- `mujoco_menagerie/unitree_go2/` — vendored subset of google-deepmind/mujoco_menagerie (Go2 only, has its own LICENSE/CHANGELOG). Load the robot via `mujoco_menagerie/unitree_go2/scene.xml`; `scene_mjx.xml`/`go2_mjx.xml` are the MJX (GPU) variants with sphere-only collision geoms. Don't modify vendored assets.
