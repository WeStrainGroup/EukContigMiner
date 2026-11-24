#!/usr/bin/env bash
set -euo pipefail
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'The ECM Conda package requires Linux x86-64; use the release wheel on other supported platforms.' >&2
    exit 1
fi
runtime="$PREFIX/share/eukcontigminer/runtime/0.91"
"$PREFIX/bin/python" -m venv "$runtime"
"$runtime/bin/python" -m pip install "$PREFIX/share/eukcontigminer/eukcontigminer-0.91-py3-none-any.whl"
"$runtime/bin/python" -m pip check
"$runtime/bin/python" -c 'from eukcontigminer.runtime.encoders import check_runtime_dependencies; check_runtime_dependencies()'
