"""Legacy compatibility entry.

This script has been archived under `legacy/example/`.
Use `example/reach_mpc_xarm6_nmpc_cpp.py` as canonical entrypoint.
"""

import os
import runpy


if __name__ == "__main__":
    legacy_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "legacy", "example", os.path.basename(__file__))
    )
    print(f"[legacy] redirecting to {legacy_path}")
    runpy.run_path(legacy_path, run_name="__main__")
