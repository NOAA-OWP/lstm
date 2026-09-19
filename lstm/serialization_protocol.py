"""
The ngen BMI Serialization Protocol surface, independent of any model.

ngen checkpoints a BMI model through four reserved variables (protocol v0.2,
``doc/BMI_SERIALIZATION_PROTOCOL.md`` in ngen). :class:`SerializationProtocol`
owns the plain numpy arrays that back those variables and implements the
trigger, announce, and delivery dispatch on top of two callables supplied by
the model: one that captures the model's state to bytes and one that applies
bytes back to the model. It knows nothing about LSTMs, snapshots, or the
payload format, and imports neither torch nor the BMI class nor the codec.

Reserved variables
==================

+--------------------------------+-------------------+--------+-------------------------------------------+
| Name                           | Unit              | Type   | Role                                      |
+================================+===================+========+===========================================+
| ``ngen::serialization_create`` | ``ngen::trigger`` | int32  | set: capture state into the buffer        |
+--------------------------------+-------------------+--------+-------------------------------------------+
| ``ngen::serialization_free``   | ``ngen::trigger`` | int32  | set: release the buffer                   |
+--------------------------------+-------------------+--------+-------------------------------------------+
| ``ngen::serialization_size``   | ``bytes``         | int64  | get: buffer length; set: announce the     |
|                                |                   |        | length of the payload about to arrive     |
+--------------------------------+-------------------+--------+-------------------------------------------+
| ``ngen::serialization_state``  | ``ngen::opaque``  | uint8  | get: the buffer; set: deliver a payload   |
+--------------------------------+-------------------+--------+-------------------------------------------+

Invariant
=========

The size array always equals the length of the payload buffer. Every
transition goes through one private method that sets both:

- create stores an owned copy of the captured bytes;
- free and :meth:`SerializationProtocol.release` empty the buffer;
- announce allocates a zero-filled buffer of the announced length;
- delivery requires the delivered length to equal the buffer length (the
  announcement), invokes the restore callable, and on success stores the
  delivered bytes in the buffer.

Because the invariant holds at every moment, a BMI byte-count getter can be
the generic "item size times length" for every name, with no special case.

The lookup surface (membership, ``unit``, ``value``, ``set_value``) matches
the model's own state containers so a protocol object can sit alongside them
in the BMI getters' "first container holding this name" lookup.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .model_state import typing

__all__ = [
    "SERIALIZATION_CREATE",
    "SERIALIZATION_FREE",
    "SERIALIZATION_SIZE",
    "SERIALIZATION_STATE",
    "SERIALIZATION_TRIGGER_UNIT",
    "SERIALIZATION_SIZE_UNIT",
    "SERIALIZATION_OPAQUE_UNIT",
    "SERIALIZATION_VAR_NAMES",
    "SerializationProtocol",
]

SERIALIZATION_CREATE: typing.Final[str] = "ngen::serialization_create"
"""trigger: capture the model's state into the payload buffer"""
SERIALIZATION_FREE: typing.Final[str] = "ngen::serialization_free"
"""trigger: release the payload buffer"""
SERIALIZATION_SIZE: typing.Final[str] = "ngen::serialization_size"
"""byte count of the payload buffer (read after create; set before delivery)"""
SERIALIZATION_STATE: typing.Final[str] = "ngen::serialization_state"
"""the opaque payload bytes (read to serialize; set to restore)"""

SERIALIZATION_TRIGGER_UNIT: typing.Final[str] = "ngen::trigger"
SERIALIZATION_SIZE_UNIT: typing.Final[str] = "bytes"
SERIALIZATION_OPAQUE_UNIT: typing.Final[str] = "ngen::opaque"

SERIALIZATION_VAR_NAMES: typing.Final[tuple[str, ...]] = (
    SERIALIZATION_CREATE,
    SERIALIZATION_FREE,
    SERIALIZATION_SIZE,
    SERIALIZATION_STATE,
)
"""all reserved protocol names, in protocol-document order"""

_UNITS: typing.Final[dict[str, str]] = {
    SERIALIZATION_CREATE: SERIALIZATION_TRIGGER_UNIT,
    SERIALIZATION_FREE: SERIALIZATION_TRIGGER_UNIT,
    SERIALIZATION_SIZE: SERIALIZATION_SIZE_UNIT,
    SERIALIZATION_STATE: SERIALIZATION_OPAQUE_UNIT,
}

Payload = typing.Union[bytes, bytearray, memoryview, npt.NDArray[np.uint8]]
"""Anything exposing a byte buffer: what ngen delivers and what callers pass."""

CaptureFn = typing.Callable[[], Payload]
"""Returns the model's state as bytes. May raise; the buffer is then untouched."""

RestoreFn = typing.Callable[[bytes], None]
"""Applies previously captured bytes to the model. May raise; the buffer is then untouched."""


_ARRAYS: typing.Final[dict[str, str]] = {
    SERIALIZATION_CREATE: "_create",
    SERIALIZATION_FREE: "_free",
    SERIALIZATION_SIZE: "_size",
    SERIALIZATION_STATE: "_buffer",
}
"""reserved name -> attribute holding its backing array (the buffer is rebound, so look it up each time)"""

_T = typing.TypeVar("_T")


def _lookup(table: typing.Mapping[str, _T], name: str) -> _T:
    """Index ``table`` by a reserved name, raising ``KeyError`` with a clear message otherwise."""
    try:
        return table[name]
    except KeyError:
        raise KeyError(f"unknown serialization name: {name!s}") from None


def _to_bytes(src: Payload) -> bytes:
    """Return an independent ``bytes`` copy of any byte-buffer-like object."""
    return bytes(memoryview(src).cast("B"))


class SerializationProtocol:
    """
    Backing arrays and dispatch for the four reserved protocol variables.

    Construct with a ``capture`` callable returning payload bytes and a
    ``restore`` callable accepting payload bytes. The arrays exist from
    construction so the names resolve for introspection before the model is
    initialized; whether capture or restore can succeed at that point is the
    callables' concern.
    """

    def __init__(self, capture: CaptureFn, restore: RestoreFn) -> None:
        self._capture = capture
        self._restore = restore
        self._create = np.zeros(1, dtype="int32")
        self._free = np.zeros(1, dtype="int32")
        self._size = np.zeros(1, dtype="int64")
        self._buffer: npt.NDArray[np.uint8] = np.empty(0, dtype="uint8")
        self._setters: dict[str, typing.Callable[[typing.Any], None]] = {
            SERIALIZATION_CREATE: self._on_create,
            SERIALIZATION_FREE: self._on_free,
            SERIALIZATION_SIZE: self._on_announce,
            SERIALIZATION_STATE: self._on_deliver,
        }

    # lookup surface shared with the model's state containers

    def __contains__(self, name: object) -> bool:
        """Return whether ``name`` is one of the four reserved names."""
        return name in _UNITS

    def unit(self, name: str) -> str:
        """Return the exact unit string ngen probes for ``name``."""
        return _lookup(_UNITS, name)

    def value(self, name: str) -> npt.NDArray:
        """Return the array backing ``name`` by reference (never a copy)."""
        return getattr(self, _lookup(_ARRAYS, name))

    def set_value(self, name: str, src: typing.Any) -> None:
        """
        Dispatch a BMI ``set_value`` on a reserved name.

        The two triggers ignore ``src``. The size announces the incoming
        payload length. The state delivers a payload.
        """
        _lookup(self._setters, name)(src)

    def release(self) -> None:
        """Empty the payload buffer. Safe at any time; for ``finalize()``."""
        self._set_buffer(np.empty(0, dtype="uint8"))

    # the one place the size and the buffer change

    def _set_buffer(self, buffer: npt.NDArray[np.uint8]) -> None:
        """Replace the buffer and record its length, keeping size == len(buffer)."""
        self._buffer = buffer
        self._size[0] = buffer.size

    # setter handlers

    def _on_create(self, _src: typing.Any) -> None:
        """Create trigger: capture into an owned copy. A failed capture changes nothing."""
        payload = _to_bytes(self._capture())
        self._set_buffer(np.frombuffer(payload, dtype="uint8").copy())

    def _on_free(self, _src: typing.Any) -> None:
        """Free trigger: release the buffer. Safe before any create."""
        self.release()

    def _on_announce(self, src: typing.Any) -> None:
        """
        Size setter: allocate a zero-filled buffer of the announced length.

        ``src`` must hold exactly one non-negative integer; anything else
        raises ``ValueError`` and leaves the buffer as it was.
        """
        announced = np.asarray(src)
        if announced.size != 1:
            raise ValueError(
                f"{SERIALIZATION_SIZE} expects exactly one value, got {announced.size}"
            )
        if not np.issubdtype(announced.dtype, np.integer):
            raise ValueError(
                f"{SERIALIZATION_SIZE} expects an integer, got dtype {announced.dtype}"
            )
        count = int(announced.reshape(-1)[0])
        if count < 0:
            raise ValueError(f"{SERIALIZATION_SIZE} must be non-negative, got {count}")
        self._set_buffer(np.zeros(count, dtype="uint8"))

    def _on_deliver(self, src: Payload) -> None:
        """
        State setter: hand the delivered bytes to the restore callable.

        The delivered length must equal the current buffer length, which the
        preceding size announcement set; otherwise ``ValueError`` is raised
        before the restore callable is invoked. If the callable raises, the
        buffer is left as it was. On success the buffer holds the delivered
        bytes.
        """
        delivered = _to_bytes(src)
        if len(delivered) != self._buffer.size:
            raise ValueError(
                f"{SERIALIZATION_STATE}: delivered {len(delivered)} bytes but "
                f"{SERIALIZATION_SIZE} announced {self._buffer.size}"
            )
        self._restore(delivered)
        self._set_buffer(np.frombuffer(delivered, dtype="uint8").copy())
