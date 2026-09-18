"""Vendored MoFlow subset.

Upstream mixes package-relative imports (``from .context_encoder ...``) with
repo-root absolute ones (``from models... ``, ``from utils...``), and its
checkpoints unpickle against those same top-level names. Upstream handles this
with a shim in its repo root; this is the same trick pointed at the vendored
copy, so the aliases follow this package wherever it lives.

The aliases are only installed when this package is imported, which happens
only when the moflow backend is requested.
"""

from __future__ import annotations

import importlib
import sys

# Upstream subpackages that its absolute imports and checkpoints refer to.
_ALIASED_SUBPACKAGES = ("models", "utils")


def _alias_repo_root_package(alias: str, target: str) -> None:
    if alias in sys.modules:
        return
    try:
        sys.modules[alias] = importlib.import_module(target)
    except ModuleNotFoundError:
        # trainer/ and data/ are not vendored; only inference code is needed.
        pass


for _subpackage in _ALIASED_SUBPACKAGES:
    _alias_repo_root_package(_subpackage, f"{__name__}.{_subpackage}")
