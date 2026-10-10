"""Path bootstrap so ``scripts/*.py`` can ``import mariodpo`` without install.

Importing this module (``import _bootstrap``) prepends ``../src`` to ``sys.path``.
Scripts also work after ``pip install -e .``; this is just a convenience.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
