# src/ingestion/__init__.py
"""Document ingestion modules.

This package intentionally avoids eager imports because some submodules
pull in heavy optional dependencies such as Camelot/OpenCV. Import the
specific submodule you need instead of importing the package for side effects.
"""

__all__ = []
