"""
Hook resolver for ABI encoding offset discovery.

Provides :func:`resolve_hooks` and :class:`EncodingContext` for callers
who need to discover the exact byte offsets of argument slots in
ABI-encoded output without modifying the core encoder.
"""

from eth_utils import (
    is_list_like,
)

from eth_abi import grammar
from eth_abi.encoding import (
    BaseArrayEncoder,
    DynamicArrayEncoder,
    TupleEncoder,
)


class EncodingContext:
    """
    Context information provided to callable hook values during ABI encoding.

    When a callable is passed as a value to be encoded, it will be called with
    an instance of this class and must return the actual value to encode.

    :param offset: The byte offset in the encoded output where this value's
        data begins.  For static types, this is the position of the value in
        the head section.  For dynamic types, this is the position of the data
        (length prefix included) in the tail section.
    :param type_str: The ABI type string for this value, e.g. ``"uint256"``
        or ``"bytes"``.
    :param size: The encoded byte size of the value at this position.  For
        static types this is always available (e.g. 32 for standard-ABI
        ``uint256``, 1 for packed ``uint8``, 20 for packed ``address``).  For
        dynamic types (``bytes``, ``string``, dynamic arrays) it is computed
        from the placeholder value returned by the hook and is ``None`` *while
        the hook is executing*; it will be set on this context object before
        :func:`resolve_hooks` returns, so callers that capture the context
        object can access ``ctx.size`` after :func:`resolve_hooks` returns.
    :param is_packed: ``True`` when the value is being resolved for packed
        (non-padded) encoding via :func:`~eth_abi.packed.encode_packed``.
        ``False`` for standard ABI encoding.
    """

    def __init__(
        self,
        offset: int,
        type_str: str | None,
        size: int | None,
        is_packed: bool,
    ) -> None:
        self.offset = offset
        self.type_str = type_str
        self.size = size
        self.is_packed = is_packed


def resolve_hooks(registry, types, values, *, is_packed=False):
    """
    Resolve callable hook values within ``values``, returning a new list
    with each hook replaced by its return value.

    A hook is any callable placed where a value is expected.  Before
    encoding, the hook is called with an :class:`EncodingContext` object
    that reports the absolute byte offset at which the value's encoded data
    will appear in the final output, the ABI type string for that position,
    the encoded byte size, and whether packed encoding is in use.

    Hooks may be placed at primitive leaf positions at any depth of nesting:
    directly in ``values``, inside tuple values, inside array values, or any
    combination thereof.

    :param registry: The :class:`~eth_abi.registry.ABIRegistry` to use for
        looking up encoders.
    :param types: A sequence of ABI type strings, e.g.
        ``['uint256', 'bytes[]', '(int,int)']``.
    :param values: A sequence of python values, possibly containing callable
        hooks.
    :param is_packed: ``True`` when resolving for packed encoding (sets
        :attr:`EncodingContext.is_packed` on every context object).

    :returns: A new list with all hooks replaced by their return values.
        The resolved list can be passed directly to
        :func:`~eth_abi.abi.encode` or :func:`~eth_abi.packed.encode_packed`.
    """
    encoders = [registry.get_encoder(t) for t in types]
    return _resolve_sequence(
        list(values), list(encoders), list(types), base_offset=0, is_packed=is_packed
    )


def _get_head_size(encoder, type_str):
    """
    Returns the number of bytes that ``encoder`` contributes to the head
    section of a standard ABI-encoded tuple.

    Dynamic encoders contribute a 32-byte offset pointer.  Static primitive
    encoders contribute exactly 32 bytes.  Static composite encoders
    (tuples, fixed-size arrays) contribute the combined head size of their
    elements.
    """
    if getattr(encoder, "is_dynamic", False):
        return 32
    if isinstance(encoder, TupleEncoder):
        abi_type = grammar.parse(type_str)
        sub_types = [c.to_type_str() for c in abi_type.components]
        return sum(_get_head_size(e, ts) for e, ts in zip(encoder.encoders, sub_types))
    if (
        hasattr(encoder, "array_size")
        and hasattr(encoder, "item_encoder")
        and encoder.array_size is not None
    ):
        abi_type = grammar.parse(type_str)
        item_type_str = abi_type.item_type.to_type_str()
        return encoder.array_size * _get_head_size(encoder.item_encoder, item_type_str)
    # All other primitive static types (uint, int, address, bool, bytesN, …)
    return 32


def _resolve_value(value, encoder, type_str, base_offset, is_packed):
    """
    Resolve a single (value, encoder, type_str) triple at the given offset.

    * Callables are invoked with an :class:`EncodingContext` and replaced by
      their return value.
    * Tuples are recursed into via :func:`_resolve_tuple`.
    * Arrays are recursed into via :func:`_resolve_array`.
    * All other values are returned unchanged.
    """
    if callable(value):
        # For fixed-size encoders (static primitives) data_byte_size is the
        # exact encoded byte width: 32 for standard-ABI types, 1/4/20/…  for
        # packed types.  Dynamic encoders have no data_byte_size, so we pass
        # None and fill it in after the hook returns (see below).
        size = getattr(encoder, "data_byte_size", None)
        ctx = EncodingContext(
            offset=base_offset,
            type_str=type_str,
            size=size,
            is_packed=is_packed,
        )
        resolved = value(ctx)
        # For dynamic types size wasn't known when the hook ran; compute it
        # from the placeholder the hook returned so callers who capture the ctx
        # object can read ctx.size after resolve_hooks() returns.
        if ctx.size is None:
            ctx.size = len(encoder(resolved))
        return resolved
    if isinstance(encoder, TupleEncoder) and is_list_like(value):
        abi_type = grammar.parse(type_str)
        sub_types = [c.to_type_str() for c in abi_type.components]
        return _resolve_tuple(value, encoder, sub_types, base_offset, is_packed)
    if isinstance(encoder, BaseArrayEncoder) and is_list_like(value):
        abi_type = grammar.parse(type_str)
        item_type_str = abi_type.item_type.to_type_str()
        return _resolve_array(value, encoder, item_type_str, base_offset, is_packed)
    return value


def _resolve_sequence(values, encoders, type_strs, base_offset, is_packed):
    """
    Resolve hooks in a flat sequence of ``(value, encoder, type_str)`` triples.

    Handles both standard ABI layout (static head + dynamic tail) and packed
    layout (all items sequential, no 32-byte padding for static items).

    ``base_offset`` is the absolute byte position where this sequence's
    encoding starts in the final output.
    """
    # Analytic head total — used only to compute absolute offsets for dynamic
    # items whose data lives in the tail section.
    head_total = sum(_get_head_size(e, ts) for e, ts in zip(encoders, type_strs))

    current_head_pos = 0
    current_tail_size = 0
    resolved = list(values)

    for i, (value, encoder, type_str) in enumerate(zip(values, encoders, type_strs)):
        is_dynamic = getattr(encoder, "is_dynamic", False)

        if is_dynamic:
            item_base = base_offset + head_total + current_tail_size
        else:
            item_base = base_offset + current_head_pos

        resolved_value = _resolve_value(value, encoder, type_str, item_base, is_packed)
        resolved[i] = resolved_value

        # Advance position counters using the actual encoded byte length so
        # that packed encoders (uint8 → 1 byte, address → 20 bytes, etc.) are
        # tracked correctly alongside standard 32-byte-slot ABI encoders.
        encoded = encoder(resolved_value)
        if is_dynamic:
            current_tail_size += len(encoded)
            current_head_pos += 32  # head slot is always 32 bytes (offset pointer)
        else:
            current_head_pos += len(encoded)

    return resolved


def _resolve_tuple(values, encoder, sub_types, base_offset, is_packed):
    """Recurse into a tuple/struct value."""
    return _resolve_sequence(
        list(values), list(encoder.encoders), sub_types, base_offset, is_packed
    )


def _resolve_array(values, encoder, item_type_str, base_offset, is_packed):
    """
    Recurse into an array value.

    ``base_offset`` is the absolute byte position where this array's encoding
    starts (including the 32-byte count prefix for
    :class:`~eth_abi.encoding.DynamicArrayEncoder`).

    ABI layout for arrays with dynamic items::

        [count?][ptr0][ptr1]...[data0][data1]...

    ABI layout for arrays with static items (or packed arrays)::

        [count?][item0][item1]...

    The count slot is only present for
    :class:`~eth_abi.encoding.DynamicArrayEncoder`.
    """
    item_enc = encoder.item_encoder
    items_dynamic = getattr(item_enc, "is_dynamic", False)

    # DynamicArrayEncoder prepends a 32-byte element count; others do not.
    if isinstance(encoder, DynamicArrayEncoder):
        elements_base = base_offset + 32
    else:
        elements_base = base_offset

    resolved = list(values)

    if items_dynamic:
        # Head section: one 32-byte offset pointer per element.
        head_total = 32 * len(values)
        current_tail_size = 0
        for i, item in enumerate(values):
            item_base = elements_base + head_total + current_tail_size
            resolved_item = _resolve_value(
                item, item_enc, item_type_str, item_base, is_packed
            )
            resolved[i] = resolved_item
            current_tail_size += len(item_enc(resolved_item))
    else:
        # Static ABI items or packed items: laid out sequentially.
        current_pos = 0
        for i, item in enumerate(values):
            item_base = elements_base + current_pos
            resolved_item = _resolve_value(
                item, item_enc, item_type_str, item_base, is_packed
            )
            resolved[i] = resolved_item
            current_pos += len(item_enc(resolved_item))

    return resolved
