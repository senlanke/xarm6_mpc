# cpp_backend

This folder keeps separate high-performance backends while leaving original scripts intact.

## Native C++ module (pybind11)
- `native/src/nmpc_native.cpp`
  - `run_nmpc`: full native NMPC loop.
  - `RenderStepController`: C++ control stepping for render mode.
  - `RenderTools`: C++ render marker helpers and batch trajectory FK.

## Build
- `bash build_cpp_all.sh`
