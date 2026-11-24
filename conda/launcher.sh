#!/usr/bin/env bash
set -euo pipefail
prefix="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime="$prefix/share/eukcontigminer/runtime/0.91"
if [[ ! -x "$runtime/bin/eukcontigminer" ]]; then
    echo 'ECM runtime installation is incomplete; repeat the Conda installation.' >&2
    exit 1
fi
exec "$runtime/bin/eukcontigminer" "$@"
