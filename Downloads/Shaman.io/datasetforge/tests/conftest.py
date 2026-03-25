"""Shared pytest configuration for DatasetForge tests."""
import os
import sys
import types


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# The .pycache directory contains a broken numpy stub from the Xcode Python path.
# This causes hypothesis's isinstance() check to fail with TypeError.
# Remove it from sys.modules so hypothesis doesn't try to use it.
_numpy_stub = sys.modules.get("numpy")
if _numpy_stub is not None and not isinstance(_numpy_stub, types.ModuleType):
    del sys.modules["numpy"]
