"""Helpers for redirecting compatibility scripts to archived legacy files."""

from __future__ import annotations

import runpy
import warnings
from pathlib import Path


def run_legacy(script_path: str, *, quiet: bool = False) -> None:
    """Run the archived legacy script corresponding to `script_path`."""
    current = Path(script_path).resolve()
    root = current.parent.parent
    legacy_path = root / "legacy" / "example" / current.name
    warnings.warn(
        (
            f"{current.name} is a legacy compatibility script. "
            "Use example/reach_mpc_xarm6_nmpc_cpp.py instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    if not quiet:
        print(f"[legacy] redirecting to {legacy_path}")
    runpy.run_path(str(legacy_path), run_name="__main__")
