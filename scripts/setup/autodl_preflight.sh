#!/usr/bin/env bash
set -u

failures=0
check_command() {
  if command -v "$1" >/dev/null 2>&1; then
    printf '[OK] %s: %s\n' "$1" "$(command -v "$1")"
  else
    printf '[WARN] missing command: %s\n' "$1"
    failures=$((failures + 1))
  fi
}

for tool in git python ffmpeg ffprobe; do check_command "$tool"; done
if command -v nvidia-smi >/dev/null 2>&1; then
  printf '[OK] GPU telemetry available\n'
else
  printf '[WARN] GPU telemetry unavailable (preflight continues)\n'
fi
for path in /root/autodl-tmp/AIC-VideoHighlight-run /root/autodl-tmp/datasets /root/autodl-tmp/models /root/autodl-tmp/hf-cache; do
  if [ -e "$path" ]; then printf '[OK] %s\n' "$path"; else printf '[WARN] missing path: %s\n' "$path"; failures=$((failures + 1)); fi
done
printf '[CHECKPOINT] preflight warnings: %s\n' "$failures"
exit 0
