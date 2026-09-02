from __future__ import annotations

import functools
import inspect
from inspect import signature
from typing import Any, TypeVar, overload

_FuncT = TypeVar("_FuncT")

# Abstract away differences between functions, methods and descriptors


def _get_member(member: object) -> object:
    if inspect.ismethod(member):
        return member.__func__
    if inspect.isdatadescriptor(member) and isinstance(member, property):
        return member.fget
    if inspect.ismethoddescriptor(member):
        if isinstance(member, classmethod):
            return member.__get__(object, object).__func__
        if isinstance(member, staticmethod):
            return member.__get__(object, object)
    return member


def is_method(member: object) -> bool:
    return inspect.isfunction(member) or inspect.ismethod(member)


def is_functional_member(member: object) -> bool:
    """
    Check whether a class member from the __dict__ attribute is a method.

    This can be true in two ways:
        - It is literally a Python function
        - It is a method descriptor (wrapping a function)

    Args:
        member: An object in the class __dict__.

    Returns:
        `True` if the member is a function (or acts like one).
    """
    return inspect.isfunction(member) or (inspect.ismethoddescriptor(member) and isinstance(member, (classmethod, staticmethod)))


def get_functional_signature(member: object) -> inspect.Signature:
    return signature(_get_member(member))  # type: ignore[arg-type]


def functional_hasattr(member: object, attr: str) -> bool:
    return hasattr(_get_member(member), attr)


def functional_getattr(member: object, attr: str, default: Any = None) -> Any:
    return getattr(_get_member(member), attr, default)


def functional_setattr(member: object, attr: str, value: object) -> None:
    setattr(_get_member(member), attr, value)


def functional_delattr(member: object, attr: str) -> None:
    delattr(_get_member(member), attr)


@overload
def get_functional_wrapper(member: classmethod[Any, Any, Any]) -> classmethod[Any, Any, Any]: ...
@overload
def get_functional_wrapper(member: property) -> property: ...
@overload
def get_functional_wrapper(member: staticmethod[Any, Any]) -> staticmethod[Any, Any]: ...
@overload
def get_functional_wrapper(member: _FuncT) -> _FuncT: ...
def get_functional_wrapper(member: object) -> object:
    """
    Return a new functional wrapper around the provided member.

    Since `interface_meta` rewrites documentation strings, if the affected
    method is from a parent class, then we need to create a new function wrapper
    so that mutating the documentation does not leak to higher classes. This
    function does just that.

    Args:
        member: The functional object to be wrapped.

    Returns:
        An object that is functionally equivalent to `member`,
            but which can have its own attributes.
    """
    class_method = isinstance(member, classmethod)
    property_method = isinstance(member, property)
    static_method = isinstance(member, staticmethod)

    function = _get_member(member)

    @functools.wraps(function)  # type: ignore[arg-type]
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return function(*args, **kwargs)  # type: ignore[operator]

    for attr in ["_quirks_method", "_quirks_mro", "__override__", "__override_force__"]:
        if hasattr(function, attr):
            setattr(wrapper, attr, getattr(function, attr))

    if class_method:
        return classmethod(wrapper)
    elif property_method:
        assert isinstance(member, property)
        return property(fget=wrapper, fset=member.fset, fdel=member.fdel)
    elif static_method:
        return staticmethod(wrapper)
    return wrapper


# Override checking


def has_explicit_override(member: object) -> bool:
    return bool(functional_getattr(member, "__override__", False))


def set_explicit_override(member: object, override: bool = True) -> None:
    functional_setattr(member, "__override__", override)


def has_forced_override(member: object) -> bool:
    return bool(functional_getattr(member, "__override_force__", False))


def set_forced_override(member: object, force: bool = True) -> None:
    functional_setattr(member, "__override_force__", force)


# Skip interface conformance checks


def should_skip(member: object) -> bool:
    return bool(functional_getattr(member, "__interface_meta_skip__", False))


def set_skip(member: object, skip: bool = True) -> None:
    functional_setattr(member, "__interface_meta_skip__", skip)


# Documentation helpers


def has_updatable_docs(member: object) -> bool:
    return is_functional_member(member) or (inspect.isdatadescriptor(member) and isinstance(member, property))


def get_functional_docs(member: object, orig: bool = True) -> str | None:
    if orig and functional_hasattr(member, "__doc_orig__"):
        return functional_getattr(member, "__doc_orig__")
    return functional_getattr(member, "__doc__")


def set_functional_docs(member: object, docs: str | None) -> None:
    if not functional_hasattr(member, "__doc_orig__"):
        functional_setattr(member, "__doc_orig__", functional_getattr(member, "__doc__"))
    functional_setattr(member, "__doc__", docs)


def has_class_attr_docs(cls: type) -> bool:
    return hasattr(cls, f"_{cls.__name__}__doc_attrs")


def get_class_attr_docs(cls: type) -> str | None:
    return getattr(cls, f"_{cls.__name__}__doc_attrs")


# Quirk documentation helpers


def has_quirk_docs_method(member: object) -> bool:
    return functional_hasattr(member, "_quirks_method")


def get_quirk_docs_method(member: object) -> str | None:
    return functional_getattr(member, "_quirks_method", None)


def set_quirk_docs_method(member: object, method: str | None) -> None:
    functional_setattr(member, "_quirks_method", method)


def has_quirk_docs_mro(member: object) -> bool:
    return functional_hasattr(member, "_quirks_mro")


def get_quirk_docs_mro(member: object) -> bool:
    return bool(functional_getattr(member, "_quirks_mro", True))


def set_quirk_docs_mro(member: object, mro: bool) -> None:
    functional_setattr(member, "_quirks_mro", mro)
