#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOW_MEM="${LOW_MEM:-0}"

if [[ "${LOW_MEM}" == "1" ]]; then
  BUILD_DIR="${SCRIPT_DIR}/build_lowmem"
  CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Debug}"
  BUILD_JOBS="${BUILD_JOBS:-1}"
  CXX_FLAGS_LOW_MEM="-O0 -g0 -fno-var-tracking -fno-var-tracking-assignments -fno-ipa-cp-clone"
  CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS:-${CXX_FLAGS_LOW_MEM}}"
else
  BUILD_DIR="${SCRIPT_DIR}/build"
  CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
  BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
  CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS:-}"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[native] Python not found. Set PYTHON_BIN explicitly." >&2
  exit 1
fi

echo "[native] Using python: ${PYTHON_BIN}"
echo "[native] LOW_MEM: ${LOW_MEM}"
echo "[native] Build dir: ${BUILD_DIR}"
echo "[native] Build type: ${CMAKE_BUILD_TYPE}"
echo "[native] Build jobs: ${BUILD_JOBS}"
PYBIND11_DIR="$(${PYTHON_BIN} -m pybind11 --cmakedir)"
PYTHON_INCLUDE_DIR="$(${PYTHON_BIN} -c "import sysconfig; print(sysconfig.get_paths()['include'])")"
PYTHON_LIB_DIR="$(${PYTHON_BIN} -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")"
if [[ -f "${PYTHON_LIB_DIR}/libpython3.10.so" ]]; then
  PYTHON_LIBRARY="${PYTHON_LIB_DIR}/libpython3.10.so"
else
  PYTHON_LIBRARY="${PYTHON_LIB_DIR}/libpython3.so"
fi
echo "[native] pybind11_DIR: ${PYBIND11_DIR}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
  -DPython3_EXECUTABLE="${PYTHON_BIN}" \
  -DPYTHON_EXECUTABLE="${PYTHON_BIN}" \
  -DPYTHON_INCLUDE_DIR="${PYTHON_INCLUDE_DIR}" \
  -DPYTHON_LIBRARY="${PYTHON_LIBRARY}" \
  -Dpybind11_DIR="${PYBIND11_DIR}"

cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"

MODULE_PATH="$(find "${BUILD_DIR}" -maxdepth 2 -type f -name 'nmpc_native*.so' | head -n 1)"
if [[ -z "${MODULE_PATH}" ]]; then
  echo "[native] build succeeded but nmpc_native*.so not found" >&2
  exit 1
fi

cp -f "${MODULE_PATH}" "${ROOT_DIR}/cpp_backend/"
echo "[native] Copied module to ${ROOT_DIR}/cpp_backend/"
