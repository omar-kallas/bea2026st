#!/usr/bin/env bash
set -euo pipefail

# Usage: ./export_environment.sh [output_file] [env_name]
# Generates a less strict conda environment file from history and appends
# current pip installs under a pip: subsection.

OUT_FILE="${1:-environment.yml}"
ENV_NAME="${2:-${CONDA_DEFAULT_ENV:-}}"
TMP_HIST="$(mktemp)"
TMP_FULL="$(mktemp)"
TMP_PIP="$(mktemp)"
cleanup() {
  rm -f "$TMP_HIST" "$TMP_FULL" "$TMP_PIP"
}
trap cleanup EXIT

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not available in PATH." >&2
  exit 1
fi
if ! command -v pip >/dev/null 2>&1; then
  echo "Error: pip is not available in PATH." >&2
  exit 1
fi

if [[ -n "$ENV_NAME" ]]; then
  conda env export --from-history -n "$ENV_NAME" > "$TMP_HIST"
  conda env export -n "$ENV_NAME" > "$TMP_FULL"
else
  conda env export --from-history > "$TMP_HIST"
  conda env export > "$TMP_FULL"
fi

awk '
  /^  - pip:/ { pip = 1; next }
  /^prefix:[[:space:]]/ { pip = 0 }
  pip && /^      - / {
    sub(/^      - /, "")
    print
  }
' "$TMP_FULL" > "$TMP_PIP"

if grep -Eq '^[[:space:]]*-[[:space:]]*pip([[:space:]]*$|=)' "$TMP_HIST"; then
  HAS_PIP_DEP=1
else
  HAS_PIP_DEP=0
fi

awk -v pip_file="$TMP_PIP" -v has_pip_dep="$HAS_PIP_DEP" '
  function emit_pip_block(    line) {
    if (emitted) return
    if ((getline line < pip_file) > 0) {
      if (!has_pip_dep) print "  - pip"
      print "  - pip:"
      print "      - " line
      while ((getline line < pip_file) > 0) {
        print "      - " line
      }
      close(pip_file)
    }
    emitted = 1
  }

  /^prefix:[[:space:]]/ {
    emit_pip_block()
    next
  }

  {
    print
  }

  END {
    emit_pip_block()
  }
' "$TMP_HIST" > "$OUT_FILE"

if [[ -n "$ENV_NAME" ]]; then
  echo "Wrote $OUT_FILE (env: $ENV_NAME)"
else
  echo "Wrote $OUT_FILE (env: current shell)"
fi
