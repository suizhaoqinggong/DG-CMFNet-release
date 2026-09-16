"""TensorBoard entry point that does not require setuptools pkg_resources.

TensorBoard 2.20 still imports ``pkg_resources`` while newer setuptools builds
may no longer provide it. This wrapper installs the tiny subset TensorBoard uses
before importing TensorBoard itself.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterable


def _parse_version(version: str) -> object:
    try:
        from packaging.version import parse
    except Exception:
        from distutils.version import LooseVersion

        return LooseVersion(version)
    return parse(version)


def install_pkg_resources_shim(*, force: bool = False) -> None:
    """Install a minimal pkg_resources module for TensorBoard startup."""
    if not force and "pkg_resources" in sys.modules:
        return

    shim = types.ModuleType("pkg_resources")

    def iter_entry_points(group: str) -> Iterable[object]:
        return []

    shim.iter_entry_points = iter_entry_points  # type: ignore[attr-defined]
    shim.parse_version = _parse_version  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = shim


def main() -> int:
    """Run TensorBoard after installing the compatibility shim."""
    install_pkg_resources_shim()
    from tensorboard.main import run_main

    result = run_main()
    return int(result) if result is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
