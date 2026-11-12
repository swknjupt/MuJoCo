#!/usr/bin/env bash
set -euo pipefail

test -e /sys/kernel/realtime
echo "smoke kernel: $(uname -r)"
cyclictest -m -p99 -i 1000 -D 60 -a 1 -t 1
echo "smoke complete"
