"""Legacy alias entrypoint.

Canonical entrypoint is `reach_mpc_xarm6_nmpc_cpp.py --backend native`.
This file is kept for compatibility.
"""

from _legacy_redirect import run_legacy


if __name__ == "__main__":
    run_legacy(__file__)
