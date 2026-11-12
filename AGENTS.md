# AGENTS.md

## Workspace

Personal MuJoCo learning workspace — no git repo, no CI, no lint/test config. Verification means running the scripts. Docs, docstrings, and comments are in Chinese; keep new content consistent.

Reference material (do not duplicate): `docs/mujoco-learning-notes.md` (verified env setup, 2026-10), `docs/mujoco-guide.md` (usage manual), `docs/embodied-robot-learning-roadmap.md` (learning plan).

The current engineering track is the robotics MBD platform under `docs/mbd/`. Treat:

- `docs/mbd/robotics-mbd-tasks.md` as the task/status source of truth;
- `docs/mbd/robotics-mbd-taskD.md` as the execution-level source of truth for D-series work;
- `docs/mbd/robotics-mbd-platform-design.md`, `robotics-mbd-demo-plan.md`, and `robotics-mbd-schedule.md` as upstream design/scheduling references.

Do not mark a task complete until its stated acceptance criteria are met. Record blockers and acceptance evidence in the MBD task documents.

## Environment

- Python venv at `.venv/`, interpreter managed by **uv** (no `pyproject.toml`). The current venv is Python 3.14.0, not 3.13; inspect `.venv/bin/python --version` instead of hard-coding the version. Always run scripts as `.venv/bin/python <script>` from the repo root. Install deps with `uv pip install -r requirements.txt` (fully pinned: mujoco 3.15.0, torch 2.x CUDA stack, stable-baselines3, gymnasium, onnx/onnxruntime).
- Host is **WSL2 (Ubuntu) + WSLg**. A CUDA-enabled torch wheel is installed, but `torch.cuda.is_available()` is currently **False** in this workspace; verify it before assuming GPU acceleration instead of relying on stale documentation. Interactive `mujoco.viewer.launch()` windows appear on the Windows desktop via WSLg — no extra setup needed (macOS would require `mjpython`, irrelevant here).
- After any dependency or environment change, run the sanity check: `.venv/bin/python verify_mujoco.py` (covers model build → simulate → offscreen render).
- Do not run real-time work in WSL2. PREEMPT_RT, kexec, cyclictest, and target-machine acceptance belong on the T14 RT box (`kevin@192.168.36.142`, hostname `kevin-tp`); IP/login details may change and must be verified before use.
- Never persist the T14 password in repository files, scripts, shell history, or logs. Use an interactive SSH prompt or a user-managed SSH key.

## MuJoCo gotchas

- `MUJOCO_GL=egl` must be set **before** `import mujoco` for offscreen rendering — see `verify_mujoco.py` / `examples/04_render_offscreen.py` for the `os.environ.setdefault` pattern. Backends: `egl` (headless), `glfw` (interactive viewer, used automatically).
- Run from the repo root: `examples/05_train_rl_ppo.py` saves its checkpoint to the CWD-relative path `examples/output/ppo_inverted_pendulum.zip` (scripts 02–04 and 06 use `Path(__file__)`-relative paths and are CWD-independent). Prefer `Path(__file__)`-relative paths for new scripts.
- RL stack is Gymnasium env → SB3 (e.g. `PPO("MlpPolicy", …, device="auto")`) → torch. `InvertedPendulum-v5` remains the minimal SB3 smoke-test environment; the MBD task uses `demo.envs.PendulumEnv`.
- The MBD environment deliberately mirrors deployment: policy actions are target angles at 100Hz, while the environment executes 1kHz PD with `decimation=10`. Preserve this interface when changing training or deployment code.

## MBD and RT safety

- When hardware arrives, run **Stage 1 D-Lite** before the full Demo: D2.2 bus check → DL1 release fixture/guard → DL2 free-swing real/sim capture → DL3 parameter identification → DL4 model-driven A/B brake → DL5 ten repeats and comparison report. Do not load the RL policy until DL1–DL5 pass.
- Stage 1 uses `demo/configs/stage1_pendulum_model.json` and `demo/configs/stage1_brake_governor.json`; they are demo-only model/safety configurations, not the Stage 2 production threshold set.
- The T14 GRUB default is **generic**. Enter RT only through the documented `kexec-rt.sh` flow; do not select RT directly from GRUB.
- For unattended RT jobs, use the batch mode, keep a harmless short smoke test first, and prefer the headless configuration that blacklists desktop GPU/audio modules and starts `rt-batch.service` directly.
- `hwlatdetect-on-RT` is permanently abandoned. Do not reintroduce it.
- If T14 becomes unreachable or hangs, perform a full power-off and power-on. Do not use hot reboot, which can leave CNVi/ME state dirty.
- `rt-batch` should normally remain disabled after acceptance work; enable it only for a prepared queue, then disable it after collecting results.
- Target-machine reports must be copied back from T14 and distinguished from host-only timings. Host timing is functional evidence, not RT acceptance evidence.

## Layout

- `examples/01..06_*.py` — numbered, self-contained learning scripts; each docstring states its run command (and viewer controls for 06).
- `examples/models/` — hand-written MJCF XMLs for the learning examples; `demo/models/` — MJCF used by the MBD demo.
- `demo/envs/` — Gymnasium pendulum environment, evaluation, and acceptance tests.
- `demo/rt/` — RT loop, governor, timeboxed inference, fault injection, simulated plant, RT-SIL runner, job scripts, and T14 kexec/batch helpers.
- `demo/host/` — hardware-independent fast fault simulation, governor pre-tuning, timeline replay, and single-file HTML reporting.
- `contracts/` — M0 FPGA/RTOS contracts, timing budget, and versioned signal-dictionary schema.
- `examples/output/` — generated artifacts only (PNG frames, checkpoints, ONNX, JSON reports, CSV timelines); do not hand-edit generated outputs.
- `docs/mbd/` — MBD design, demo, schedule, task status, and D-series execution specifications.
- `mujoco_menagerie/unitree_go2/` — vendored subset of google-deepmind/mujoco_menagerie (Go2 only, has its own LICENSE/CHANGELOG). Load the robot via `mujoco_menagerie/unitree_go2/scene.xml`; `scene_mjx.xml`/`go2_mjx.xml` are the MJX (GPU) variants with sphere-only collision geoms. Don't modify vendored assets.

## Verification commands

Run these from the repo root as applicable:

```bash
.venv/bin/python verify_mujoco.py
.venv/bin/python -m demo.envs.test_pendulum_env
.venv/bin/python -m demo.envs.pendulum_eval
.venv/bin/python -m demo.rt.test_governor
.venv/bin/python -m demo.rt.test_inference
.venv/bin/python -m demo.rt.test_inject
.venv/bin/python -m demo.rt.test_motor_io
.venv/bin/python -m demo.rt.test_run_rt_sil_config
.venv/bin/python -m demo.host.test_report
.venv/bin/python -m demo.host.test_fast_tools
.venv/bin/python contracts/test_signal_dict.py
```

Host-only RT-SIL integration (not an RT acceptance result):

```bash
.venv/bin/python demo/rt/run_rt_sil.py --allow-unprivileged --allow-non-rt
```

The training entrypoint is `demo/train_pendulum_ppo.py`; it saves
`examples/output/ppo_pendulum_demo.zip`. ONNX export is
`demo/export_policy_onnx.py`; the lightweight target-machine benchmark is
`demo/benchmark_onnx.py`.

Host-side D3 tools:

```bash
.venv/bin/python demo/host/scan_faults.py
.venv/bin/python demo/host/tune_governor.py --mode all
.venv/bin/python demo/host/replay.py --run-dir examples/output/t14-rt-20261110 --act 4
.venv/bin/python demo/host/report.py --run-dir examples/output/t14-rt-20261110
```
