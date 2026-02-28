"""Legacy compatibility wrapper.

Canonical runtime path uses the native C++ module `nmpc_native`.
This file re-exports historical Python solver from `legacy/`.
"""

from legacy.mpc.solver.reach_ddp import solver_ddp_reach

__all__ = ["solver_ddp_reach"]
