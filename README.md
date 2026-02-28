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

### 0) This update (2026-02-28)
- Added `RenderNmpcRunner` C++ class and pybind export:
  - `cpp_backend/native/src/render_nmpc_runner.cpp`
  - `cpp_backend/native/src/nmpc_native/render_nmpc_runner.hpp`
  - `cpp_backend/native/src/module_bindings.cpp`
- `example/reach_mpc_xarm6_nmpc_cpp.py` now uses C++ `step()` in render mode:
  - Replan scheduling + control + marker submission are done in one C++ call.
- Improved native build ergonomics:
  - `cpp_backend/native/build.sh` now supports Ninja/ccache auto-detection and local `CCACHE_DIR`.
  - Auto-clean build directory when generator changes.
- Added `requestment.txt` as compatibility alias to `requirements.txt`.
- Added viewer-side explicit GLFW init failures in `mujoco_viewer.py` to avoid silent hangs.

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
- Added `RenderNmpcRunner` (pybind11):
  - Moves render-mode replan scheduling and per-step control/marker dispatch into a single C++ `step()` call.
- `example/reach_mpc_xarm6_nmpc_cpp.py` render path now uses this C++ controller:
  - Python keeps MuJoCo stepping + viewer render.
  - DDP replan scheduling, per-step control math, marker submission, and trajectory FK preview are offloaded to C++.

### 5) One-click compile for all C++ targets
- Added top-level script: `build_cpp_all.sh`
- Added native build script: `cpp_backend/native/build.sh`
- Current C++ build targets included in one-click build:
  - `cpp_backend/native` (`nmpc_native` pybind11 module, includes `run_nmpc`, `RenderStepController`, and `RenderTools`)

### 6) Architecture cleanup (legacy archive)
- Archived historical Python DDP implementation into `legacy/`:
  - `legacy/mpc/solver/reach_ddp.py`
  - `legacy/mpc/solver/reach_ddp_xarm6.py`
  - `legacy/mpc/utils/action.py`
  - `legacy/cpp_backend/reach_ddp_xarm6_cpp.py`
- Original import paths are preserved by compatibility wrappers under:
  - `mpc/solver/`
  - `mpc/utils/`
  - `cpp_backend/`

## Dependencies

Install required packages with:

```bash
pip install -r requirements.txt
```

Compatibility install command (same dependency set):

```bash
pip install -r requestment.txt
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

Faster rebuild (if available):

```bash
cd /path/to/xarm6_mpc
USE_NINJA=1 USE_CCACHE=1 PYTHON_BIN=/path/to/python/bin/python bash build_cpp_all.sh
```

Build knobs:
- `BUILD_JOBS=<N>`: override parallel jobs.
- `LOW_MEM=1`: force low-memory serial build.
- `USE_NINJA=0`: disable Ninja auto-selection.
- `USE_CCACHE=0`: disable ccache launcher.

This will compile and place the modules at:
- `cpp_backend/nmpc_native*.so`

## Running Examples

### Mainline
```bash
cd /path/to/xarm6_mpc/example
conda activate ke
python reach_mpc_xarm6_nmpc_cpp.py --backend render
python reach_mpc_xarm6_nmpc_cpp.py --backend native
```

If render blocks at viewer creation, first verify display access:

```bash
echo $DISPLAY
python - <<'PY'
import glfw
print("glfw.init() =", glfw.init())
if glfw.init():
    print("primary_monitor =", glfw.get_primary_monitor())
    glfw.terminate()
PY
```

### Compatibility
```bash
python reach_mpc_xarm6_nmpc_native.py
```

Compatibility scripts forward to `legacy/example/` and emit deprecation warnings.

## Repository Structure

```
xarm6_mpc/
├── README.md
├── requirements.txt
├── build_cpp_all.sh
├── callbacks.py
├── cpp_backend/          # C++ implementations and bindings
├── example/              # Example scripts
├── legacy/               # Archived legacy Python implementations
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
- Every functional update should be recorded in this README under `Recent changes`.

## License

See the [LICENSE](LICENSE) file for licensing information.
