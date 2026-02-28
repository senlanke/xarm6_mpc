# cpp_backend

This folder keeps separate high-performance backends while leaving original scripts intact.

## Native C++ module (pybind11)
- `native/src/ddp_reach_solver.cpp`: DDP reach solver wrapper (C++).
- `native/src/render_step_controller.cpp`: render-step torque control wrapper.
- `native/src/render_tools.cpp`: marker helpers and batch FK.
- `native/src/run_nmpc.cpp`: full native NMPC loop.
- `native/src/module_bindings.cpp`: pybind11 module bindings.

## Build
- `bash build_cpp_all.sh`
- Low-memory build (recommended on RAM-limited machines):
  - `LOW_MEM=1 PYTHON_BIN=/home/kesl/miniconda3/envs/ke/bin/python bash build_cpp_all.sh`
