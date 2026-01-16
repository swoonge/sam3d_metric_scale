#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_env="bufferx"

forward_args=("$@")
if [[ " $* " != *" --env "* ]]; then
  forward_args=(--env "${default_env}" "${forward_args[@]}")
fi

exec "${script_dir}/setup_bufferx_new_env.sh" "${forward_args[@]}"
