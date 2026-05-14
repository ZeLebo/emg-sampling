"""Compatibility package for running src-layout project without installation."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
_src_package = Path(__file__).resolve().parent.parent / "src" / "emg_sampling"
if _src_package.exists():
    __path__.append(str(_src_package))
