#
# Model-side implementation of the ngen BMI Serialization Protocol payload.
#
# The protocol treats the payload as opaque bytes; this module owns their
# layout. All fields are little-endian:
#
#   magic           4s      b"LSTM"
#   version         u16     PAYLOAD_VERSION
#   timestep        i64     bmi_LSTM._timestep
#   n_vars          u32     number of BMI `Var`s that follow
#     per var:
#       name_len    u16
#       name        name_len bytes (utf-8)
#       n_values    u32
#       values      n_values f64
#   n_members       u32     number of ensemble members that follow
#     per member:
#       n_values    u32     numel of h_t (== numel of c_t)
#       h_t         n_values f32
#       c_t         n_values f32
#
# Restores are all-or-nothing: the payload is fully decoded and validated
# against the live model before any state is mutated.
#
from __future__ import annotations

import struct
import typing
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

if typing.TYPE_CHECKING:
    from .bmi_lstm import EnsembleMember
    from .model_state import State

SERIALIZATION_CREATE: typing.Final = "ngen::serialization_create"
SERIALIZATION_FREE: typing.Final = "ngen::serialization_free"
SERIALIZATION_SIZE: typing.Final = "ngen::serialization_size"
SERIALIZATION_STATE: typing.Final = "ngen::serialization_state"

UNIT_TRIGGER: typing.Final = "ngen::trigger"
UNIT_OPAQUE: typing.Final = "ngen::opaque"
UNIT_BYTES: typing.Final = "bytes"

MAGIC: typing.Final = b"LSTM"
PAYLOAD_VERSION: typing.Final = 1

_HEADER = struct.Struct("<4sHq")
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_F64 = np.dtype("<f8")
_F32 = np.dtype("<f4")


class PayloadError(ValueError):
    """Raised when a payload is malformed or does not match the live model."""


@dataclass
class Snapshot:
    timestep: int
    vars: dict[str, npt.NDArray]
    """BMI variable name -> float64 values"""
    members: list[tuple[npt.NDArray, npt.NDArray]]
    """per ensemble member (h_t, c_t) as flat float32 arrays"""


def encode(
    timestep: int,
    states: typing.Iterable[State],
    members: typing.Sequence[EnsembleMember],
) -> bytes:
    """Serialize model state into the payload format described above."""
    vars = [var for state in states for var in state]

    parts = [_HEADER.pack(MAGIC, PAYLOAD_VERSION, timestep), _U32.pack(len(vars))]
    for var in vars:
        name = var.name.encode("utf-8")
        values = np.ascontiguousarray(var.value, dtype=_F64).ravel()
        parts.append(_U16.pack(len(name)))
        parts.append(name)
        parts.append(_U32.pack(values.size))
        parts.append(values.tobytes())

    parts.append(_U32.pack(len(members)))
    for member in members:
        h_t = member.h_t.detach().cpu().numpy().astype(_F32).ravel()
        c_t = member.c_t.detach().cpu().numpy().astype(_F32).ravel()
        assert h_t.size == c_t.size, "h_t and c_t must be the same size"
        parts.append(_U32.pack(h_t.size))
        parts.append(h_t.tobytes())
        parts.append(c_t.tobytes())

    return b"".join(parts)


class _Reader:
    def __init__(self, buf: bytes):
        self._buf = memoryview(buf)
        self._offset = 0

    def take(self, n: int) -> memoryview:
        end = self._offset + n
        if end > len(self._buf):
            raise PayloadError(
                f"payload truncated: need {end} bytes, have {len(self._buf)}"
            )
        view = self._buf[self._offset : end]
        self._offset = end
        return view

    def unpack(self, s: struct.Struct) -> tuple[typing.Any, ...]:
        return s.unpack(self.take(s.size))

    def array(self, dtype: np.dtype, count: int) -> npt.NDArray:
        return np.frombuffer(self.take(dtype.itemsize * count), dtype=dtype).copy()

    def remaining(self) -> int:
        return len(self._buf) - self._offset


def decode(buf: bytes) -> Snapshot:
    """Parse a payload produced by `encode`. Raises `PayloadError` if malformed."""
    r = _Reader(buf)
    magic, version, timestep = r.unpack(_HEADER)
    if magic != MAGIC:
        raise PayloadError(f"bad magic: expected {MAGIC!r}, got {bytes(magic)!r}")
    if version != PAYLOAD_VERSION:
        raise PayloadError(
            f"unsupported payload version: {version} (supported: {PAYLOAD_VERSION})"
        )

    vars: dict[str, npt.NDArray] = {}
    (n_vars,) = r.unpack(_U32)
    for _ in range(n_vars):
        (name_len,) = r.unpack(_U16)
        name = bytes(r.take(name_len)).decode("utf-8")
        (n_values,) = r.unpack(_U32)
        vars[name] = r.array(_F64, n_values)

    members: list[tuple[npt.NDArray, npt.NDArray]] = []
    (n_members,) = r.unpack(_U32)
    for _ in range(n_members):
        (n_values,) = r.unpack(_U32)
        h_t = r.array(_F32, n_values)
        c_t = r.array(_F32, n_values)
        members.append((h_t, c_t))

    if r.remaining() != 0:
        raise PayloadError(f"payload has {r.remaining()} unexpected trailing bytes")

    return Snapshot(timestep=timestep, vars=vars, members=members)


def apply(
    snapshot: Snapshot,
    states: typing.Iterable[State],
    members: typing.Sequence[EnsembleMember],
) -> None:
    """
    Validate `snapshot` against the live model and write it into `states` and
    `members`. Nothing is mutated unless the whole snapshot is compatible.

    Note: the caller is responsible for restoring the timestep.
    """
    states = list(states)
    names = [name for state in states for name in state.names()]
    if sorted(names) != sorted(snapshot.vars):
        raise PayloadError(
            f"variable mismatch: model has {sorted(names)}, payload has {sorted(snapshot.vars)}"
        )
    for state in states:
        for var in state:
            if snapshot.vars[var.name].size != var.value.size:
                raise PayloadError(
                    f"size mismatch for {var.name!r}: model has {var.value.size}, "
                    f"payload has {snapshot.vars[var.name].size}"
                )

    if len(members) != len(snapshot.members):
        raise PayloadError(
            f"ensemble member mismatch: model has {len(members)}, payload has {len(snapshot.members)}"
        )
    for i, (member, (h_t, c_t)) in enumerate(zip(members, snapshot.members)):
        if member.h_t.numel() != h_t.size or member.c_t.numel() != c_t.size:
            raise PayloadError(
                f"hidden state size mismatch for ensemble member {i}: "
                f"model has {member.h_t.numel()}, payload has {h_t.size}"
            )

    # validated; mutate.
    for state in states:
        for var in state:
            var.value[:] = snapshot.vars[var.name].astype(var.value.dtype)
    for member, (h_t, c_t) in zip(members, snapshot.members):
        member.h_t = torch.from_numpy(h_t).reshape(member.h_t.shape).to(member.h_t.dtype)
        member.c_t = torch.from_numpy(c_t).reshape(member.c_t.shape).to(member.c_t.dtype)
