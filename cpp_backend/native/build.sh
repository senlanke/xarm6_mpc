#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOW_MEM="${LOW_MEM:-0}"
USE_CCACHE="${USE_CCACHE:-1}"
USE_NINJA="${USE_NINJA:-1}"
LOCAL_TOOLS_DIR="${LOCAL_TOOLS_DIR:-${ROOT_DIR}/.local-tools/bin}"

if [[ -d "${LOCAL_TOOLS_DIR}" ]]; then
  export PATH="${LOCAL_TOOLS_DIR}:${PATH}"
fi

if [[ -z "${CCACHE_DIR:-}" ]]; then
  export CCACHE_DIR="${ROOT_DIR}/.ccache"
fi
mkdir -p "${CCACHE_DIR}"

detect_build_jobs() {
  local cores mem_kb
  cores="$(nproc 2>/dev/null || echo 1)"
  if [[ "${cores}" -lt 1 ]]; then
    cores=1
  fi

  if [[ -r /proc/meminfo ]]; then
    mem_kb="$(awk '/MemAvailable:/ {print $2; exit}' /proc/meminfo)"
  else
    mem_kb=""
  fi

  if [[ -n "${mem_kb}" ]]; then
    # Keep conservative defaults on low-memory hosts to avoid OOM kill.
    if (( mem_kb < 6 * 1024 * 1024 )); then
      echo 1
      return
    fi
    if (( mem_kb < 12 * 1024 * 1024 )); then
      if (( cores > 2 )); then
        echo 2
      else
        echo "${cores}"
      fi
      return
    fi
  fi

  if (( cores > 4 )); then
    echo 4
  else
    echo "${cores}"
  fi
}

if [[ "${LOW_MEM}" == "1" ]]; then
  BUILD_DIR="${SCRIPT_DIR}/build_lowmem"
  CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Debug}"
  BUILD_JOBS="${BUILD_JOBS:-1}"
  CXX_FLAGS_LOW_MEM="-O0 -g0 -fno-var-tracking -fno-var-tracking-assignments -fno-ipa-cp-clone -fno-inline-functions -fno-inline-small-functions -fno-tree-vectorize"
  CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS:-${CXX_FLAGS_LOW_MEM}}"
  CMAKE_CXX_FLAGS_DEBUG="${CMAKE_CXX_FLAGS_DEBUG:--O0 -g0 -DNDEBUG}"
  LOW_MEM_CMAKE_ARGS=(-DCMAKE_CXX_FLAGS_DEBUG="${CMAKE_CXX_FLAGS_DEBUG}")
else
  BUILD_DIR="${SCRIPT_DIR}/build"
  CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
  BUILD_JOBS="${BUILD_JOBS:-$(detect_build_jobs)}"
  CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS:-}"
  LOW_MEM_CMAKE_ARGS=()
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

GENERATOR_ARGS=()
if [[ "${USE_NINJA}" == "1" ]]; then
  if [[ -n "${CMAKE_GENERATOR:-}" ]]; then
    GENERATOR_ARGS=(-G "${CMAKE_GENERATOR}")
  elif command -v ninja >/dev/null 2>&1; then
    GENERATOR_ARGS=(-G Ninja)
  fi
fi
if [[ ${#GENERATOR_ARGS[@]} -gt 0 ]]; then
  echo "[native] Generator: ${GENERATOR_ARGS[1]}"
else
  echo "[native] Generator: default"
fi

CMAKE_EXTRA_ARGS=()
if [[ "${USE_CCACHE}" == "1" ]] && command -v ccache >/dev/null 2>&1; then
  CMAKE_EXTRA_ARGS+=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
  echo "[native] ccache: enabled"
else
  echo "[native] ccache: disabled"
fi

PYBIND11_DIR="$(${PYTHON_BIN} -m pybind11 --cmakedir)"
PYTHON_INCLUDE_DIR="$(${PYTHON_BIN} -c "import sysconfig; print(sysconfig.get_paths()['include'])")"
PYTHON_LIB_DIR="$(${PYTHON_BIN} -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")"
if [[ -f "${PYTHON_LIB_DIR}/libpython3.10.so" ]]; then
  PYTHON_LIBRARY="${PYTHON_LIB_DIR}/libpython3.10.so"
else
  PYTHON_LIBRARY="${PYTHON_LIB_DIR}/libpython3.so"
fi
echo "[native] pybind11_DIR: ${PYBIND11_DIR}"

requested_generator=""
if [[ ${#GENERATOR_ARGS[@]} -gt 0 ]]; then
  requested_generator="${GENERATOR_ARGS[1]}"
fi

if [[ -f "${BUILD_DIR}/CMakeCache.txt" && -n "${requested_generator}" ]]; then
  cached_generator="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1)"
  if [[ -n "${cached_generator}" && "${cached_generator}" != "${requested_generator}" ]]; then
    echo "[native] Generator changed (${cached_generator} -> ${requested_generator}), cleaning ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
  fi
fi

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  "${GENERATOR_ARGS[@]}" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
  -DPython3_EXECUTABLE="${PYTHON_BIN}" \
  -DPYTHON_EXECUTABLE="${PYTHON_BIN}" \
  -DPYTHON_INCLUDE_DIR="${PYTHON_INCLUDE_DIR}" \
  -DPYTHON_LIBRARY="${PYTHON_LIBRARY}" \
  -Dpybind11_DIR="${PYBIND11_DIR}" \
  "${LOW_MEM_CMAKE_ARGS[@]}" \
  "${CMAKE_EXTRA_ARGS[@]}"

cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"

MODULE_PATH="$(find "${BUILD_DIR}" -maxdepth 2 -type f -name 'nmpc_native*.so' | head -n 1)"
if [[ -z "${MODULE_PATH}" ]]; then
  echo "[native] build succeeded but nmpc_native*.so not found" >&2
  exit 1
fi

cp -f "${MODULE_PATH}" "${ROOT_DIR}/cpp_backend/"
echo "[native] Copied module to ${ROOT_DIR}/cpp_backend/"
