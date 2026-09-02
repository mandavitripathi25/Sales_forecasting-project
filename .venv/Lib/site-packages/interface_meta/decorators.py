from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, overload

from .interface import InterfaceMeta
from .utils.inspection import set_skip

_FuncT = TypeVar("_FuncT")


def inherit_docs(
    method: str | None = None,
    mro: bool = True,
) -> Callable[[_FuncT], _FuncT]:
    """
    Indicate to `InterfaceMeta` how the wrapped method should be documented.

    Methods need not normally be decorated with this decorator, except in the
    following cases:
    - documentation for quirks should be lifted not from overrides to a
      method, but from some other method (e.g. because subclasses or
      implementations of the interface should override behaviour in a "private"
      method rather than the top-level public method).
    - the method has been nominated by the interface configuration to be
      skipped, in which case decorating with this method will enable
      documentation generation as if it were not.

    Use this decorator as `@inherit_docs([method=...], [mro=...])`.

    Args:
        method: A method from which documentation for implementation
            specific quirks should be extracted. [Useful when implementations
            of an interface are supposed to change underlying methods rather
            than the public method itself].
        mro: Whether to include documentation from all levels of the
            MRO, starting from the most primitive class that implementated it.
            All higher levels will be considered as "quirks" to the interface's
            definition.

    Returns:
        A function wrapper that attaches attributes `_quirks_method` and
        `_quirks_mro` to the method, for interpretation by `InterfaceMeta`.
    """
    return InterfaceMeta.inherit_docs(method=method, mro=mro)


@overload
def override(func: _FuncT, force: bool = ...) -> _FuncT: ...
@overload
def override(func: None = None, force: bool = ...) -> Callable[[_FuncT], _FuncT]: ...
def override(
    func: _FuncT | None = None,
    force: bool = False,
) -> _FuncT | Callable[[_FuncT], _FuncT]:
    """
    Indicate to `InterfaceMeta` that this method has intentionally overridden an interface method.

    This decorator also allows one to indicate that the method should be
    overridden without warnings even when it does not conform to the API.

    Use this decorator as `@override` or `@override(force=True)`.

    A recommended convention is to use this decorator as the outermost decorator.

    Args:
        func: The function, if method is decorated by the decorator
            without arguments (e.g. @override), else None.
        force: Whether to force override of method even if the API does
            note match. Note that in this case, documentation is not inherited
            from the MRO.

    Returns:
        The wrapped function or function wrapper depending on which
            arguments are present.
    """
    if func is not None:
        return InterfaceMeta.override(func=func, force=force)
    return InterfaceMeta.override(force=force)


def skip(func: _FuncT) -> _FuncT:
    """
    Indicate to `InterfaceMeta` that this method should be skipped.

    Args:
        func: The function/member to mark as skipped.

    Returns:
        The marked function/member (same instance as that passed in).
    """
    set_skip(func, skip=True)
    return func
