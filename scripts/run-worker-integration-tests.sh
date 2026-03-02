#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[integration-test] %s\n' "$*"
}

trim() {
  local s="$1"
  # shellcheck disable=SC2001
  s="$(echo "$s" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  printf '%s' "$s"
}

as_unix_url() {
  local value="$1"
  if [[ "$value" == unix://* ]]; then
    printf '%s' "$value"
  else
    printf 'unix://%s' "$value"
  fi
}

is_usable_unix_socket() {
  local host="$1"
  [[ "$host" == unix://* ]] || return 1
  local path="${host#unix://}"
  [[ -S "$path" ]]
}

check_host() {
  local host="$1"
  if [[ "$host" == unix://* ]]; then
    is_usable_unix_socket "$host"
  else
    if command -v docker >/dev/null 2>&1; then
      docker --host "$host" version >/dev/null 2>&1
    else
      return 1
    fi
  fi
}

discover_hosts() {
  local out=()

  if [[ -n "${DOCKER_HOST:-}" ]]; then
    out+=("$(trim "$DOCKER_HOST")")
  fi

  if command -v docker >/dev/null 2>&1; then
    local ctx cur host
    cur="$(docker context show 2>/dev/null || true)"
    cur="$(trim "$cur")"
    if [[ -n "$cur" ]]; then
      host="$(docker context inspect "$cur" --format '{{(index .Endpoints "docker").Host}}' 2>/dev/null || true)"
      host="$(trim "$host")"
      [[ -n "$host" ]] && out+=("$host")
    fi

    while IFS= read -r ctx; do
      [[ -z "$ctx" ]] && continue
      host="$(docker context inspect "$ctx" --format '{{(index .Endpoints "docker").Host}}' 2>/dev/null || true)"
      host="$(trim "$host")"
      [[ -n "$host" ]] && out+=("$host")
    done < <(docker context ls --format '{{.Name}}' 2>/dev/null || true)
  fi

  if command -v podman >/dev/null 2>&1; then
    local podman_machine_socket podman_info_socket
    podman_machine_socket="$(podman machine inspect --format '{{.ConnectionInfo.PodmanSocket.Path}}' 2>/dev/null || true)"
    podman_machine_socket="$(trim "$podman_machine_socket")"
    if [[ -n "$podman_machine_socket" ]]; then
      out+=("$(as_unix_url "$podman_machine_socket")")
    fi

    podman_info_socket="$(podman info --format '{{.Host.RemoteSocket.Path}}' 2>/dev/null || true)"
    podman_info_socket="$(trim "$podman_info_socket")"
    if [[ -n "$podman_info_socket" ]]; then
      out+=("$(as_unix_url "$podman_info_socket")")
    fi
  fi

  out+=("unix:///var/run/docker.sock")
  out+=("unix://$HOME/.docker/run/docker.sock")

  # Print unique, non-empty candidates in discovered order.
  awk 'NF && !seen[$0]++ { print }' <<<"$(printf '%s\n' "${out[@]}")"
}

pick_host() {
  local host
  while IFS= read -r host; do
    [[ -z "$host" ]] && continue
    if check_host "$host"; then
      printf '%s' "$host"
      return 0
    fi
  done < <(discover_hosts)
  return 1
}

main() {
  local host
  if ! host="$(pick_host)"; then
    log "No usable Docker/Podman endpoint found."
    log "Checked DOCKER_HOST, docker contexts, podman machine/info sockets, and common defaults."
    exit 1
  fi

  export DOCKER_HOST="$host"
  export TESTCONTAINERS_RYUK_DISABLED="${TESTCONTAINERS_RYUK_DISABLED:-true}"
  log "Using DOCKER_HOST=$DOCKER_HOST"
  log "TESTCONTAINERS_RYUK_DISABLED=$TESTCONTAINERS_RYUK_DISABLED"

  cargo test --test integration -p worker "$@"
}

main "$@"
