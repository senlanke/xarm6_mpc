"""Legacy compatibility wrapper.

Canonical runtime path uses the native C++ module `nmpc_native`.
This file re-exports historical Python DDP action models from `legacy/`.
"""

from legacy.mpc.utils.action import DAM_fwd_example, DAM_fwd_exampleT

__all__ = ["DAM_fwd_example", "DAM_fwd_exampleT"]
