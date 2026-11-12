#!/usr/bin/env bash
set -euo pipefail

cd /home/kevin/mbd-rt
.venv/bin/python demo/rt/loop.py --duration 1800 --cpu 2 \
  --report examples/output/rt_loop_target_isolated.json
