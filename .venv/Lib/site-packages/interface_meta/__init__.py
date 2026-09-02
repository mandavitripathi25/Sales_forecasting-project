from ._version import __version__  # noqa: F401
from .decorators import inherit_docs, override, skip
from .interface import InterfaceMeta

__author__ = "Matthew Wardrop"
__author_email__ = "mpwardrop@gmail.com"

__all__ = [
    "InterfaceMeta",
    "inherit_docs",
    "override",
    "skip",
]
