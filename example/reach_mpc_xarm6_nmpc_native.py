"""Legacy alias entrypoint.

Canonical entrypoint is `reach_mpc_xarm6_nmpc_cpp.py`.
This file is kept for compatibility.
"""

from reach_mpc_xarm6_nmpc_cpp import run_native, SIM_TIME


if __name__ == "__main__":
    print("[legacy] use reach_mpc_xarm6_nmpc_cpp.py --backend native")
    run_native(sim_time=SIM_TIME, max_iter=50)
