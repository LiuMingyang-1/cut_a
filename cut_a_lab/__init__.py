"""Development shim so src-layout imports work from the repository root."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path


__path__ = extend_path(__path__, __name__)

SRC_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "src" / "cut_a_lab"
if SRC_PACKAGE_DIR.is_dir():
    __path__.append(str(SRC_PACKAGE_DIR))
