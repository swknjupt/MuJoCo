#!/usr/bin/env bash
set -euo pipefail

cd /home/kevin/mbd-rt
.venv/bin/python demo/rt/run_rt_sil.py --control-cpu 1 --inference-cpu 2
