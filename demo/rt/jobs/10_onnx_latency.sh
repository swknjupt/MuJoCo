#!/usr/bin/env bash
set -euo pipefail

cd /home/kevin/mbd-rt
.venv/bin/python demo/benchmark_onnx.py
