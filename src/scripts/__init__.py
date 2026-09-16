"""Expose repository script modules from the installed ``src`` package layout."""

from pathlib import Path

_ROOT_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if _ROOT_SCRIPTS_DIR.is_dir():
    __path__.append(str(_ROOT_SCRIPTS_DIR))
