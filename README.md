# xarm_sim

Simulation and example controllers for xArm in MuJoCo.

## Overview

This repository contains simulation and example controllers for xArm robots in MuJoCo physics simulator. It includes MPC (Model Predictive Control) implementations and various control strategies for xArm6 robotic manipulator.

## Features

- Full MuJoCo simulation environment for xArm6 robot
- MPC (Model Predictive Control) implementations
- C++ backend for performance optimization
- Integration with Crocoddyl optimal control library
- Support for both torque and position control modes

## Recent changes

### 1) Torque actuator setup
- Switched actuator definitions to torque motors (`<motor ...>`) in:
  - `ufactory_xarm6/xarm6_nohand_motor.xml`
  - `xarm_description/xarm_mj/xarm7_nohand.xml`
- This makes `mj_data.ctrl` interpreted as torque command directly.

### 2) Crocoddyl C++ dynamics backend (Python loop unchanged)
- Added `cpp_backend/reach_ddp_xarm6_cpp.py`:
  - Uses Crocoddyl built-in C++ dynamics action model
    `DifferentialActionModelFreeFwdDynamics`.
- Added `example/reach_mpc_xarm6_nmpc_cpp.py`:
  - Same control flow as original script, but solver backend switched to the above.

### 3) Native C++ NMPC main loop
- Added native module sources:
  - `cpp_backend/native/src/run_nmpc.cpp` (main NMPC loop)
  - `cpp_backend/native/src/ddp_reach_solver.cpp` (DDP wrapper)
  - `cpp_backend/native/src/module_bindings.cpp` (pybind11 bindings)
- Added `example/reach_mpc_xarm6_nmpc_native.py`:
  - Python entry for calling native module `nmpc_native`.

### 4) Render-mode control stepping moved to C++ extension
- Added render-side native sources:
  - `cpp_backend/native/src/render_step_controller.cpp`
  - `cpp_backend/native/src/render_tools.cpp`
- `RenderStepController` (pybind11):
  - Handles per-step phase index (`i/K`), desired state picking from `xs`,
    and PD+gravity torque computation in C++.
- Added `RenderTools` (pybind11) in the same module:
  - C++ marker drawing helpers (`add_marker`, `draw_trajectory`, `draw_ee_frame`).
  - C++ batch trajectory forward-kinematics (`batch_trajectory_fk`).
- `example/reach_mpc_xarm6_nmpc_cpp.py` render path now uses this C++ controller:
  - Python keeps DDP replan + viewer render.
  - Per-step control math, render marker helpers, and trajectory FK preview are offloaded to C++.

### 5) One-click compile for all C++ targets
- Added top-level script: `build_cpp_all.sh`
- Added native build script: `cpp_backend/native/build.sh`
- Current C++ build targets included in one-click build:
  - `cpp_backend/native` (`nmpc_native` pybind11 module, includes `run_nmpc`, `RenderStepController`, and `RenderTools`)

## Dependencies

Install required packages with:

```bash
pip install -r requirements.txt
```

## Environment

Use `ke` conda environment:

```bash
conda activate ke
```

## Building C++ Extensions

From project root:

```bash
cd /path/to/xarm6_mpc
PYTHON_BIN=/path/to/python/bin/python bash build_cpp_all.sh
```

Low-memory mode (for RAM-limited machines):

```bash
cd /path/to/xarm6_mpc
LOW_MEM=1 PYTHON_BIN=/path/to/python/bin/python bash build_cpp_all.sh
```

This will compile and place the modules at:
- `cpp_backend/nmpc_native*.so`

## Running Examples

### A) Native C++ NMPC main loop (recommended)
```bash
cd /path/to/xarm6_mpc/example
conda activate ke
python reach_mpc_xarm6_nmpc_cpp.py
```

### B) Legacy native entry (same backend)
```bash
cd /path/to/xarm6_mpc/example
conda activate ke
python reach_mpc_xarm6_nmpc_native.py
```

## Repository Structure

```
xarm6_mpc/
├── README.md
├── requirements.txt
├── build_cpp_all.sh
├── callbacks.py
├── cpp_backend/          # C++ implementations and bindings
├── example/              # Example scripts
├── mpc/                  # MPC implementations
├── mujoco_viewer.py      # Custom MuJoCo viewer
├── ufactory_xarm6/       # Robot model files
├── ur5e_mj/              # UR5e robot models
├── utils/                # Utility functions
└── xarm_description/     # Alternative xarm model files
```

## Notes
- On some systems `./build_cpp_all.sh` may fail with `Permission denied` due mount options.
  In that case use:
  - `bash build_cpp_all.sh`
- Native C++ loop script runs without the custom Python viewer; it prints run statistics.

## License

See the [LICENSE](LICENSE) file for licensing information.
