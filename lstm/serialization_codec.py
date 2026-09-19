"""
Binary codec for the LSTM BMI serialization payload.

This module packs a :class:`Snapshot` of the module's computed state into a
versioned, explicit, little-endian byte layout and unpacks such bytes back into
a :class:`Snapshot`. It depends only on the standard library ``struct`` module
and numpy: no pickle, no torch, and no reference to the BMI class, so a payload
can be decoded without executing arbitrary code.

Payload layout (format version 1)
=================================

All multi-byte integers and floats are little-endian with no padding.

+---------------------------+------------------+-------------------------------------------+
| Section                   | Size (bytes)     | Content                                   |
+===========================+==================+===========================================+
| magic                     | 8                | ``b"LSTMBMI\\0"`` (:data:`MAGIC`)          |
+---------------------------+------------------+-------------------------------------------+
| version                   | 4 (uint32)       | format version (:data:`FORMAT_VERSION`)   |
+---------------------------+------------------+-------------------------------------------+
| timestep                  | 8 (int64)        | module timestep counter when captured     |
+---------------------------+------------------+-------------------------------------------+
| member_count              | 4 (uint32)       | number of ensemble members, ``M``         |
+---------------------------+------------------+-------------------------------------------+
| output_count              | 4 (uint32)       | number of output values, ``N``            |
+---------------------------+------------------+-------------------------------------------+
| fingerprint_len           | 4 (uint32)       | byte length of the fingerprint, ``F``     |
+---------------------------+------------------+-------------------------------------------+
| fingerprint               | ``F``            | UTF-8 model fingerprint text              |
+---------------------------+------------------+-------------------------------------------+
| member[i].hidden_size     | 4 (uint32)       | hidden size ``H_i`` of member ``i``       |
+---------------------------+------------------+-------------------------------------------+
| member[i].hidden          | ``4 * H_i``      | hidden state, raw float32                 |
+---------------------------+------------------+-------------------------------------------+
| member[i].cell            | ``4 * H_i``      | cell state, raw float32                   |
+---------------------------+------------------+-------------------------------------------+
| outputs                   | ``8 * N``        | output values, raw float64                |
+---------------------------+------------------+-------------------------------------------+

The three ``member[i]`` rows repeat ``M`` times, in ensemble order. The fixed
header (magic through ``fingerprint_len``) is :data:`HEADER_SIZE` bytes. The
total payload length is therefore::

    HEADER_SIZE + F + sum(MEMBER_HEADER_SIZE + 8 * H_i for i in range(M)) + 8 * N

Any change to this layout must increment :data:`FORMAT_VERSION`.

Validation
==========

:func:`unpack` reads the payload once, front to back, with a cursor. Every
read is bounds-checked, so a payload that ends inside any section raises
:class:`PayloadError` naming that section; the magic bytes and format version
are checked as soon as the header is read; and bytes left over after the
outputs section are reported as trailing. The :class:`Snapshot` is built only
after the last read, so a rejected payload returns nothing. Every failure
raises :class:`PayloadError`, a ``ValueError`` subclass, with a message naming
the cause.

The codec is a pure format: it carries the fingerprint but does not compare it
against anything. Whether a payload fits a particular model (fingerprint,
member count, hidden sizes, output count) is the caller's decision.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .model_state import dataclass_kwargs, typing

__all__ = [
    "MAGIC",
    "FORMAT_VERSION",
    "HEADER_STRUCT",
    "HEADER_SIZE",
    "MEMBER_HEADER_STRUCT",
    "MEMBER_HEADER_SIZE",
    "STATE_DTYPE",
    "OUTPUT_DTYPE",
    "Snapshot",
    "PayloadError",
    "pack",
    "unpack",
]

MAGIC: typing.Final[bytes] = b"LSTMBMI\0"
"""Eight magic bytes that open every payload."""

FORMAT_VERSION: typing.Final[int] = 1
"""Payload format version written by :func:`pack`."""

HEADER_STRUCT: typing.Final[struct.Struct] = struct.Struct("<8sIqIII")
"""
Fixed header: magic, version, timestep (int64), member_count, output_count,
fingerprint_len.
"""

HEADER_SIZE: typing.Final[int] = HEADER_STRUCT.size
"""Byte length of the fixed header (32)."""

MEMBER_HEADER_STRUCT: typing.Final[struct.Struct] = struct.Struct("<I")
"""Per-member header: hidden size as uint32."""

MEMBER_HEADER_SIZE: typing.Final[int] = MEMBER_HEADER_STRUCT.size
"""Byte length of the per-member header (4)."""

STATE_DTYPE: typing.Final[np.dtype] = np.dtype("<f4")
"""Element type of the hidden and cell state sections (little-endian float32)."""

OUTPUT_DTYPE: typing.Final[np.dtype] = np.dtype("<f8")
"""Element type of the outputs section (little-endian float64)."""


class PayloadError(ValueError):
    """
    Raised by :func:`unpack` when a payload is malformed, has an unknown
    format, or does not match its own header.
    """


@dataclass(**dataclass_kwargs)
class Snapshot:
    """
    The module's computed state as plain numpy data.

    Attributes:
        timestep: the module's integer timestep counter when captured.
        fingerprint: UTF-8 bytes identifying the model configuration.
        members: per ensemble member, in order, a ``(hidden, cell)`` pair of
            float32 arrays. :func:`pack` accepts any array shape and flattens
            it; :func:`unpack` always returns one-dimensional arrays of length
            equal to the member's hidden size.
        outputs: float64 array of output values in output variable order.
    """

    timestep: int
    fingerprint: bytes
    members: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]
    outputs: npt.NDArray[np.float64]


def _as_state_array(arr: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Return ``arr`` as a flat, contiguous little-endian float32 array."""
    return np.ascontiguousarray(arr, dtype=STATE_DTYPE).ravel()


def pack(snapshot: Snapshot) -> bytes:
    """
    Serialize a :class:`Snapshot` into payload bytes following the module
    layout table.

    Hidden and cell arrays are cast to float32 and flattened; outputs are cast
    to float64 and flattened. Raises ``ValueError`` if a member's hidden and
    cell arrays do not have the same number of elements.
    """
    fingerprint = bytes(snapshot.fingerprint)
    outputs = np.ascontiguousarray(snapshot.outputs, dtype=OUTPUT_DTYPE).ravel()

    chunks: list[bytes] = [
        HEADER_STRUCT.pack(
            MAGIC,
            FORMAT_VERSION,
            int(snapshot.timestep),
            len(snapshot.members),
            outputs.size,
            len(fingerprint),
        ),
        fingerprint,
    ]

    for index, (hidden, cell) in enumerate(snapshot.members):
        hidden_flat = _as_state_array(hidden)
        cell_flat = _as_state_array(cell)
        if hidden_flat.size != cell_flat.size:
            raise ValueError(
                f"member {index}: hidden state has {hidden_flat.size} elements "
                f"but cell state has {cell_flat.size}"
            )
        chunks.append(MEMBER_HEADER_STRUCT.pack(hidden_flat.size))
        chunks.append(hidden_flat.tobytes())
        chunks.append(cell_flat.tobytes())

    chunks.append(outputs.tobytes())
    return b"".join(chunks)


class _Cursor:
    """
    Bounds-checked forward reader over a payload buffer.

    Each read names the section it is reading so a short read raises a
    :class:`PayloadError` that says where the payload ended. Offsets are
    computed once, in the order the layout is read.
    """

    def __init__(self, buf: memoryview) -> None:
        self._buf = buf
        self._offset = 0

    def read(self, size: int, what: str) -> memoryview:
        """Return the next ``size`` bytes or raise if the payload ends inside ``what``."""
        end = self._offset + size
        if end > len(self._buf):
            raise PayloadError(
                f"payload is {len(self._buf)} bytes, truncated inside {what} "
                f"(which ends at byte {end})"
            )
        chunk = self._buf[self._offset : end]
        self._offset = end
        return chunk

    def unpack(self, layout: struct.Struct, what: str) -> tuple:
        """Read and decode one ``struct`` record."""
        return layout.unpack(self.read(layout.size, what))

    def array(self, dtype: np.dtype, count: int, what: str) -> np.ndarray:
        """Read ``count`` elements of ``dtype`` as an independent, writable array."""
        return np.frombuffer(self.read(count * dtype.itemsize, what), dtype=dtype).copy()

    def finish(self) -> None:
        """Raise if any bytes remain after the last section."""
        extra = len(self._buf) - self._offset
        if extra:
            raise PayloadError(
                f"payload is {len(self._buf)} bytes but the header implies "
                f"{self._offset} bytes ({extra} extra trailing bytes)"
            )


def unpack(
    payload: typing.Union[bytes, bytearray, memoryview, npt.NDArray[np.uint8]],
) -> Snapshot:
    """
    Deserialize payload bytes produced by :func:`pack` into a :class:`Snapshot`.

    Sections are read once, in layout order, with a bounds-checked cursor (see
    the module docstring). A :class:`PayloadError` is raised and nothing is
    returned for a malformed, unknown-version, truncated, or over-long payload.

    Returned arrays are independent copies, so mutating them does not alias
    ``payload``.
    """
    cursor = _Cursor(memoryview(payload).cast("B"))

    magic, version, timestep, member_count, output_count, fingerprint_len = cursor.unpack(
        HEADER_STRUCT, f"the {HEADER_SIZE}-byte header"
    )
    if magic != MAGIC:
        raise PayloadError(f"bad magic bytes: expected {MAGIC!r}, found {bytes(magic)!r}")
    if version != FORMAT_VERSION:
        raise PayloadError(
            f"unsupported payload format version {version}; this codec reads version {FORMAT_VERSION}"
        )

    fingerprint = bytes(cursor.read(fingerprint_len, f"the {fingerprint_len}-byte fingerprint"))

    members: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]] = []
    for index in range(member_count):
        (hidden_size,) = cursor.unpack(
            MEMBER_HEADER_STRUCT, f"the header of member {index} of {member_count}"
        )
        where = f"member {index} of {member_count}, hidden size {hidden_size}"
        hidden = cursor.array(STATE_DTYPE, hidden_size, f"the hidden state of {where}")
        cell = cursor.array(STATE_DTYPE, hidden_size, f"the cell state of {where}")
        members.append((hidden, cell))

    outputs = cursor.array(
        OUTPUT_DTYPE, output_count, f"the {output_count}-element outputs section"
    )
    cursor.finish()

    return Snapshot(
        timestep=int(timestep),
        fingerprint=fingerprint,
        members=members,
        outputs=outputs,
    )
