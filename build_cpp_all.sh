#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[build-all] Python not found. Set PYTHON_BIN explicitly." >&2
  exit 1
fi

echo "[build-all] Root: ${ROOT_DIR}"
echo "[build-all] Python: ${PYTHON_BIN}"

PROJECTS=(
  "cpp_backend/native"
)

for project in "${PROJECTS[@]}"; do
  echo "[build-all] Building ${project}"
  PYTHON_BIN="${PYTHON_BIN}" bash "${ROOT_DIR}/${project}/build.sh"
done

echo "[build-all] Done."
