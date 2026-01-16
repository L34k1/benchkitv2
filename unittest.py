"""Local wrapper to enable `python -m unittest -v` discovery.

This file preserves stdlib unittest APIs while defaulting to discovery when
invoked as a module.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import sysconfig
from types import ModuleType
from typing import Iterable


def _load_stdlib_unittest() -> ModuleType:
    stdlib_path = sysconfig.get_paths()["stdlib"]
    unittest_dir = os.path.join(stdlib_path, "unittest")
    unittest_path = os.path.join(unittest_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        "stdlib_unittest",
        unittest_path,
        submodule_search_locations=[unittest_dir],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load stdlib unittest module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_STDLIB_UNITTEST = _load_stdlib_unittest()

__all__: Iterable[str] = getattr(_STDLIB_UNITTEST, "__all__", [])

for name in __all__:
    globals()[name] = getattr(_STDLIB_UNITTEST, name)

globals().update(
    {
        "__path__": getattr(_STDLIB_UNITTEST, "__path__", []),
        "__file__": getattr(_STDLIB_UNITTEST, "__file__", None),
        "__spec__": getattr(_STDLIB_UNITTEST, "__spec__", None),
    }
)


if __name__ == "__main__":
    _STDLIB_UNITTEST.main(module=None, argv=[sys.argv[0], "discover", "-v"])
