"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

__version__ = '0.4.150'
__date__ = '2026-09-28'

import sys
import importlib
import types

# If the system-wide gemmi is already loaded into memory, use it directly (saves ABI)
if "gemmi" in sys.modules:
    gemmi = sys.modules["gemmi"]
else:
    # Intercept Gemmi's C++ internal lookups: Create an empty module entry in sys.modules
    # BEFORE calling import_module so that internal C++ imports resolve here instead of loading from disk.
    sys_gemmi = types.ModuleType("gemmi")
    sys.modules["gemmi"] = sys_gemmi
    
    try:
        # Load servalcat-bundled subpackage safely
        bundled_gemmi = importlib.import_module(".gemmi", package=__name__)
        
        # Unify the namespaces so everything targets the bundled copy uniformly
        sys_gemmi.__dict__.update(bundled_gemmi.__dict__)
        gemmi = sys_gemmi
        sys.modules["gemmi"] = gemmi
        
    except ImportError as e:
        # Roll back the placeholder footprint if loading failed completely
        if sys.modules.get("gemmi") is sys_gemmi:
            del sys.modules["gemmi"]
        try:
            import gemmi
        except ImportError:
            raise ImportError("Gemmi could not be found") from e

# Now it is safe to load the servalcat C++ extension
from . import ext

__all__ = ["gemmi", "ext", "__version__"]
