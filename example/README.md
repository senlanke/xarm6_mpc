# Example Entrypoints

## Canonical

- `python reach_mpc_xarm6_nmpc_cpp.py --backend render`
- `python reach_mpc_xarm6_nmpc_cpp.py --backend native`

## Compatibility wrappers

The following scripts are compatibility wrappers that forward to archived legacy files in `../legacy/example/`:

- `bullet_ic.py`
- `finger.py`
- `ic_test.py`
- `ik_test.py`
- `reach_mpc.py`
- `reach_mpc_xarm6_nmpc.py`
- `shadow_test.py`
- `ur_test.py`
- `xarm_mujoco.py`
- `reach_mpc_xarm6_nmpc_native.py`

Wrapper implementation is centralized in `_legacy_redirect.py`.
