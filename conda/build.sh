#!/usr/bin/env bash
set -euo pipefail
destination="$PREFIX/share/eukcontigminer"
mkdir -p "$destination" "$PREFIX/bin"
cp "$SRC_DIR/dist/eukcontigminer-0.91-py3-none-any.whl" "$destination/"
cp "$SRC_DIR/README.md" "$SRC_DIR/LICENSE" "$SRC_DIR/THIRD_PARTY_NOTICES.md" "$destination/"
cp "$SRC_DIR/conda/launcher.sh" "$PREFIX/bin/eukcontigminer"
chmod 755 "$PREFIX/bin/eukcontigminer"
