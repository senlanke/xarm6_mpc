# Legacy Archive

This directory stores legacy Python implementations kept for compatibility.

## Archived modules

- `legacy/mpc/solver/reach_ddp.py`
- `legacy/mpc/solver/reach_ddp_xarm6.py`
- `legacy/mpc/utils/action.py`
- `legacy/cpp_backend/reach_ddp_xarm6_cpp.py`
- `legacy/example/reach_mpc_xarm6_nmpc.py`
- `legacy/example/reach_mpc.py`
- `legacy/example/bullet_ic.py`
- `legacy/example/finger.py`
- `legacy/example/ic_test.py`
- `legacy/example/ik_test.py`
- `legacy/example/shadow_test.py`
- `legacy/example/ur_test.py`
- `legacy/example/xarm_mujoco.py`

## Current canonical runtime path

- Render/backend entry: `example/reach_mpc_xarm6_nmpc_cpp.py`
- Native module: `cpp_backend/nmpc_native*.so`
- Build: `LOW_MEM=1 PYTHON_BIN=/home/kesl/miniconda3/envs/ke/bin/python bash build_cpp_all.sh`
