"""Legacy compatibility wrapper.

Canonical runtime path uses the native C++ module `nmpc_native`.
This file re-exports historical Python solver from `legacy/`.
"""

from legacy.mpc.solver.reach_ddp_xarm6 import solver_ddp_reach_xarm6

__all__ = ["solver_ddp_reach_xarm6"]
