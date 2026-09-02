import inspect
import logging
from inspect import Parameter, Signature

from .inspection import (
    get_functional_signature,
    has_explicit_override,
    has_forced_override,
    is_functional_member,
    is_method,
    should_skip,
)
from .reporting import report_violation


def verify_conformance(
    name: str,
    clsname: str,
    member: object,
    ref_clsname: str,
    ref_member: object | None,
    explicit_overrides: bool = True,
    raise_on_violation: bool = False,
) -> None:
    """
    Verify that a member conforms to a nominated interface.

    Args:
        name: The name of the member method being checked.
        clsname: The name of the class parent of the checked member.
        member: The class member to check for conformance against ref_member.
        ref_clsname: The name of the reference class to be treated as an interface.
        ref_member: The referece member to be treated as an interface definition.
        explicit_overrides: Whether to require explicit overrides. (default: True)
        raise_on_violation: Whether any non-conformance should cause an
            exception to be raised. (default: False)
    """
    if hasattr(ref_member, "__objclass__"):  # pragma: no cover; Method is attached to metaclass, so should not be checked.
        return

    if ref_member is None or has_forced_override(member) or should_skip(ref_member):
        return

    # Check that type of member has not changed.
    if type(member) is not type(ref_member):
        if isinstance(member, property) and not is_method(ref_member):
            # This should be okay, provided the property is properly crafted.
            pass
        elif is_functional_member(member):
            # This means we are replacing a fixed attribute with a method,
            # or between different types of functional members
            report_violation(
                f"`{clsname}.{name}` changes the type of `{ref_clsname}.{name}` (`{type(member)}` instead of `{type(ref_member)}`) without using `@override(force=True)` decorator.",
                raise_on_violation,
            )
        else:  # Most other type changes should be fine
            pass

    # Check that overrides are present
    if is_functional_member(member) or inspect.isdatadescriptor(member) or inspect.ismethoddescriptor(member):
        if explicit_overrides and not has_explicit_override(member):
            report_violation(
                f"`{clsname}.{name}` overrides interface `{ref_clsname}.{name}` without using the `@override` decorator.",
                raise_on_violation,
            )

    if is_functional_member(member) and is_functional_member(ref_member):
        verify_signature(
            name,
            clsname,
            member,
            ref_clsname,
            ref_member,
            raise_on_violation=raise_on_violation,
        )


def verify_signature(
    name: str,
    clsname: str,
    member: object,
    ref_clsname: str,
    ref_member: object,
    raise_on_violation: bool = False,
) -> None:
    """
    Verify that the signature of a member is compatible with some reference member.

    Args:
        name: The name of the member method being checked.
        clsname: The name of the class parent of the checked member.
        member: The class member to check for conformance against ref_member.
        ref_clsname: The name of the reference class to be treated as an interface.
        ref_member: The referece member to be treated as an interface definition.
        raise_on_violation: Whether any non-conformance should cause an
            exception to be raised. (default: False)
    """
    sig = get_functional_signature(member)
    ref_sig = get_functional_signature(ref_member)

    if not check_signatures_compatible(sig, ref_sig):
        message = f"Signature `{clsname}.{name}{sig}` does not conform to interface `{ref_clsname}.{name}{ref_sig}`."
        if raise_on_violation:
            raise RuntimeError(message)
        else:
            logging.warning(message)


def check_signatures_compatible(sig: Signature, ref_sig: Signature) -> bool:
    """
    Check whether two signatures are compatible.

    Args:
        sig: A signature of a member to check for compatibility with `ref_sig`.
        ref_sig: The reference signature.

    Returns:
        `True` if the signatures are compatible, and `False` otherwise.
    """
    params = iter(sig.parameters.values())
    base_params = iter(ref_sig.parameters.values())

    try:
        for bp in base_params:
            cp = next(params)

            while bp.kind is Parameter.VAR_POSITIONAL and cp.kind is Parameter.POSITIONAL_OR_KEYWORD:
                cp = next(params)

            while bp.kind is Parameter.VAR_KEYWORD and cp.kind is not Parameter.VAR_KEYWORD:
                cp = next(params)

            if not (cp.name == bp.name and bp.kind == cp.kind and bp.default == cp.default):
                raise ValueError(bp, cp)

    except (StopIteration, ValueError):
        return False

    for param in params:
        if param.kind is Parameter.POSITIONAL_ONLY or (param.kind is Parameter.POSITIONAL_OR_KEYWORD and param.default is Parameter.empty):
            return False

    return True


def verify_not_overridden(
    name: str,
    clsname: str,
    member: object,
    raise_on_violation: bool = False,
) -> None:
    """
    Verify that a nominated member is *not* an override.

    Args:
        name: The name of the member method being checked.
        clsname: The name of the class parent of the checked member.
        member: The class member to check for conformance against ref_member.
        raise_on_violation: Whether any non-conformance should cause an
            exception to be raised. (default: False)
    """
    if has_explicit_override(member):
        report_violation(
            f"`{clsname}.{name}` claims to override interface method, but no such method exists.",
            raise_on_violation=raise_on_violation,
        )
