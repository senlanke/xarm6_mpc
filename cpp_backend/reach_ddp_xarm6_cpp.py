"""Legacy compatibility wrapper.

Canonical runtime path is `nmpc_native.DDPReachSolver`.
This module keeps the historical import path stable.
"""

from legacy.cpp_backend.reach_ddp_xarm6_cpp import SolverDDPReachXarm6Cpp

__all__ = ["SolverDDPReachXarm6Cpp"]
