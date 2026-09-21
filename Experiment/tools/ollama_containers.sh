#!/usr/bin/env bash
# Point 16 helper: keep the GPU server's per-GPU Ollama containers in shape.
#
# Run this ON THE GPU SERVER (172.26.94.12), not on the laptop: docker lives
# there. It is idempotent -- "status" only looks, "start" creates what is
# missing and leaves running containers alone.
#
# Layout documented in Experiment/.env and Experiment/README.md. Confirm the
# port <-> container <-> GPU mapping against the live server before trusting it:
#
#   ./ollama_containers.sh status
#
# Usage:
#   ./ollama_containers.sh status            # containers, devices, HTTP probe
#   ./ollama_containers.sh start             # start every missing container
#   ./ollama_containers.sh start ollama021 ollama021-2-2

set -uo pipefail

IMAGE="${OLLAMA_IMAGE:-ollama/ollama:0.21}"
MODELS_DIR="${OLLAMA_MODELS_DIR:-/data/ollama_models}"
CONTEXT="${OLLAMA_CONTEXT_LENGTH:-32768}"

# name:gpu:port
UNITS=(
  "ollama021-3-1:2:41137"
  "ollama021-3-2:3:41138"
  "ollama021-2-1:4:41133"
  "ollama021-2-2:5:41134"
  "ollama021-1:6:41135"
  "ollama021:7:41136"
)

die() { echo "error: $*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker not found (run this on the GPU server)"

running() { docker ps --format '{{.Names}}' | grep -Fxq "$1"; }
exists() { docker ps -a --format '{{.Names}}' | grep -Fxq "$1"; }

probe() {
  local port="$1"
  local body
  body="$(curl -sS -m 6 "http://127.0.0.1:${port}/api/tags" 2>/dev/null)" || {
    echo "no answer on :${port}"
    return 1
  }
  echo "HTTP ok on :${port} ($(printf '%s' "$body" | grep -o '"name"' | wc -l | tr -d ' ') models)"
}

status() {
  echo "== docker containers (ollama*) =="
  docker ps -a --filter "name=ollama" \
    --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true
  echo
  echo "== device requests =="
  for entry in "${UNITS[@]}"; do
    IFS=: read -r name gpu port <<<"$entry"
    if exists "$name"; then
      devices="$(docker inspect "$name" --format '{{json .HostConfig.DeviceRequests}}' 2>/dev/null)"
      echo "${name} (expected device ${gpu}, port ${port}): ${devices}"
    else
      echo "${name} (expected device ${gpu}, port ${port}): MISSING"
    fi
  done
  echo
  echo "== HTTP probe (from this host) =="
  for entry in "${UNITS[@]}"; do
    IFS=: read -r name gpu port <<<"$entry"
    printf '%-18s %s\n' "$name" "$(probe "$port" || true)"
  done
}

start_one() {
  local name="$1" gpu="$2" port="$3"
  if running "$name"; then
    echo "[skip] ${name} already running (port ${port})"
    return 0
  fi
  if exists "$name"; then
    echo "[warn] ${name} exists but is stopped; starting it as-is (its old port/device binding is kept)"
    docker start "$name" >/dev/null && echo "[ok  ] ${name} started"
    return $?
  fi
  echo "[run ] ${name} -> device ${gpu}, host port ${port}"
  docker run -d --name "$name" --gpus "\"device=${gpu}\"" -p "0.0.0.0:${port}:11434" \
    -v "${MODELS_DIR}:/root/.ollama/models" -e "OLLAMA_CONTEXT_LENGTH=${CONTEXT}" \
    --restart unless-stopped "$IMAGE" >/dev/null \
    && echo "[ok  ] ${name} started"
}

start() {
  local wanted=("$@")
  for entry in "${UNITS[@]}"; do
    IFS=: read -r name gpu port <<<"$entry"
    if (( ${#wanted[@]} > 0 )); then
      local match=0
      for want in "${wanted[@]}"; do [[ "$want" == "$name" ]] && match=1; done
      (( match )) || continue
    fi
    start_one "$name" "$gpu" "$port"
  done
  echo
  echo "Now verify the model lanes from the laptop:"
  echo "  python Experiment/tools/check_ollama_units.py --units http://172.26.94.12:41133,http://172.26.94.12:41134,http://172.26.94.12:41135,http://172.26.94.12:41136,http://172.26.94.12:41137,http://172.26.94.12:41138"
}

action="${1:-status}"
shift || true
case "$action" in
  status) status ;;
  start) start "$@" ;;
  *) die "unknown action '${action}' (use: status | start [name ...])" ;;
esac
