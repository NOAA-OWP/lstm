"""
Fast tests for `lstm.serialization_protocol`.

The protocol object is driven with fake capture and restore callables: no
torch, no BMI instance, no codec. Every case asserts through the object's
public surface (membership, `unit`, `value`, `set_value`, `release`) and the
single invariant that the size array equals the buffer length.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from lstm import serialization_protocol as proto
from lstm.serialization_protocol import (
    SERIALIZATION_CREATE,
    SERIALIZATION_FREE,
    SERIALIZATION_OPAQUE_UNIT,
    SERIALIZATION_SIZE,
    SERIALIZATION_SIZE_UNIT,
    SERIALIZATION_STATE,
    SERIALIZATION_TRIGGER_UNIT,
    SERIALIZATION_VAR_NAMES,
    SerializationProtocol,
)

PAYLOAD = b"LSTMBMI\0fake payload bytes"


class FakeModel:
    """Records what the protocol object asks of its callables."""

    def __init__(
        self, payload: bytes = PAYLOAD, restore_error: Exception | None = None
    ):
        self.payload = payload
        self.restore_error = restore_error
        self.captures = 0
        self.restored: list[bytes] = []

    def capture(self) -> bytes:
        self.captures += 1
        return self.payload

    def restore(self, payload: bytes) -> None:
        if self.restore_error is not None:
            raise self.restore_error
        self.restored.append(payload)


@pytest.fixture
def model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def protocol(model: FakeModel) -> SerializationProtocol:
    return SerializationProtocol(capture=model.capture, restore=model.restore)


def _size(protocol: SerializationProtocol) -> int:
    return int(protocol.value(SERIALIZATION_SIZE)[0])


def _buffer(protocol: SerializationProtocol) -> np.ndarray:
    return protocol.value(SERIALIZATION_STATE)


def _assert_invariant(protocol: SerializationProtocol, expected_length: int) -> None:
    """The size array always equals the buffer length; here both equal `expected_length`."""
    assert _size(protocol) == expected_length
    assert _buffer(protocol).shape == (expected_length,)
    assert _buffer(protocol).dtype == np.uint8


# ---------------  variable surface  -----------------------------


def test_reserved_names_are_in_protocol_document_order():
    assert SERIALIZATION_VAR_NAMES == (
        "ngen::serialization_create",
        "ngen::serialization_free",
        "ngen::serialization_size",
        "ngen::serialization_state",
    )


@pytest.mark.parametrize(
    ("name", "unit", "dtype", "itemsize"),
    [
        (SERIALIZATION_CREATE, "ngen::trigger", np.int32, 4),
        (SERIALIZATION_FREE, "ngen::trigger", np.int32, 4),
        (SERIALIZATION_SIZE, "bytes", np.int64, 8),
        (SERIALIZATION_STATE, "ngen::opaque", np.uint8, 1),
    ],
    ids=["create", "free", "size", "state"],
)
def test_unit_type_and_itemsize_per_name(
    protocol: SerializationProtocol, name: str, unit: str, dtype: type, itemsize: int
):
    assert name in protocol
    assert protocol.unit(name) == unit
    array = protocol.value(name)
    assert isinstance(array, np.ndarray)
    assert array.dtype == dtype
    assert array.itemsize == itemsize


def test_unit_constants_match_the_protocol_document():
    assert SERIALIZATION_TRIGGER_UNIT == "ngen::trigger"
    assert SERIALIZATION_SIZE_UNIT == "bytes"
    assert SERIALIZATION_OPAQUE_UNIT == "ngen::opaque"


def test_fresh_object_reads_zero_size_and_empty_buffer(protocol: SerializationProtocol):
    _assert_invariant(protocol, 0)
    assert protocol.value(SERIALIZATION_CREATE).shape == (1,)
    assert protocol.value(SERIALIZATION_FREE).shape == (1,)
    assert protocol.value(SERIALIZATION_SIZE).shape == (1,)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "ngen::serialization",
        "streamflow",
        "atmosphere_water__liquid_equivalent_precipitation_rate",
    ],
)
def test_unknown_names_are_not_members_and_raise(
    protocol: SerializationProtocol, name: str
):
    assert name not in protocol
    with pytest.raises(KeyError, match="unknown serialization name"):
        protocol.unit(name)
    with pytest.raises(KeyError, match="unknown serialization name"):
        protocol.value(name)
    with pytest.raises(KeyError, match="unknown serialization name"):
        protocol.set_value(name, np.array([1]))


def test_value_returns_the_backing_array_by_reference(protocol: SerializationProtocol):
    for name in SERIALIZATION_VAR_NAMES:
        assert protocol.value(name) is protocol.value(name)
    # a size read after a transition sees the transition through the same array
    size = protocol.value(SERIALIZATION_SIZE)
    protocol.set_value(SERIALIZATION_CREATE, np.array([1], dtype="int32"))
    assert size[0] == len(PAYLOAD)
    assert protocol.value(SERIALIZATION_SIZE) is size


def test_module_imports_neither_torch_nor_the_bmi_class_nor_the_codec():
    """Run in a subprocess so this process's imports do not mask a transitive import."""
    script = (
        "import sys; import lstm.serialization_protocol; "
        "print(sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.') "
        "or m in ('lstm.bmi_lstm', 'lstm.serialization_codec')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


# ---------------  create and free  -----------------------------


def test_create_stores_payload_and_size_equals_buffer_length(
    protocol: SerializationProtocol, model: FakeModel
):
    protocol.set_value(SERIALIZATION_CREATE, np.array([1], dtype="int32"))
    assert model.captures == 1
    _assert_invariant(protocol, len(PAYLOAD))
    assert _buffer(protocol).tobytes() == PAYLOAD


def test_create_stores_an_owned_copy():
    source = bytearray(PAYLOAD)
    protocol = SerializationProtocol(capture=lambda: source, restore=lambda _: None)
    protocol.set_value(SERIALIZATION_CREATE, np.array([1], dtype="int32"))
    buffer = _buffer(protocol)
    source[0] = 0xFF
    assert buffer.tobytes() == PAYLOAD
    buffer[1] = 0xEE
    assert bytes(source[1:2]) == PAYLOAD[1:2]


def test_create_twice_replaces_the_buffer_and_captures_each_time(
    protocol: SerializationProtocol, model: FakeModel
):
    protocol.set_value(SERIALIZATION_CREATE, np.array([1], dtype="int32"))
    first = _buffer(protocol)
    model.payload = b"second"
    protocol.set_value(SERIALIZATION_CREATE, np.array([1], dtype="int32"))
    assert model.captures == 2
    assert _buffer(protocol) is not first
    _assert_invariant(protocol, len(b"second"))
    assert _buffer(protocol).tobytes() == b"second"


def test_free_empties_buffer_and_size(protocol: SerializationProtocol):
    protocol.set_value(SERIALIZATION_CREATE, np.array([1], dtype="int32"))
    protocol.set_value(SERIALIZATION_FREE, np.array([1], dtype="int32"))
    _assert_invariant(protocol, 0)


def test_free_before_create_does_not_raise(
    protocol: SerializationProtocol, model: FakeModel
):
    protocol.set_value(SERIALIZATION_FREE, np.array([1], dtype="int32"))
    protocol.set_value(SERIALIZATION_FREE, np.array([1], dtype="int32"))
    _assert_invariant(protocol, 0)
    assert model.captures == 0


def test_release_empties_buffer_and_size(protocol: SerializationProtocol):
    protocol.release()
    _assert_invariant(protocol, 0)
    protocol.set_value(SERIALIZATION_CREATE, np.array([1], dtype="int32"))
    protocol.release()
    _assert_invariant(protocol, 0)


@pytest.mark.parametrize(
    "src",
    [
        np.array([0], dtype="int32"),
        np.array([-5], dtype="int32"),
        np.array([1, 2, 3], dtype="float64"),
        np.empty(0, dtype="int32"),
        "not even an array",
        None,
    ],
    ids=["zero", "negative", "float-vector", "empty", "string", "none"],
)
def test_trigger_values_are_ignored(
    protocol: SerializationProtocol, model: FakeModel, src
):
    protocol.set_value(SERIALIZATION_CREATE, src)
    assert model.captures == 1
    _assert_invariant(protocol, len(PAYLOAD))
    protocol.set_value(SERIALIZATION_FREE, src)
    _assert_invariant(protocol, 0)
    # the trigger arrays themselves are never written
    assert protocol.value(SERIALIZATION_CREATE)[0] == 0
    assert protocol.value(SERIALIZATION_FREE)[0] == 0


def test_failed_capture_leaves_buffer_unchanged():
    """Mirrors create before initialize(): the capture callable raises."""
    calls = 0

    def capture() -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError(
                "cannot capture serialization state before initialize() is called"
            )
        return PAYLOAD

    protocol = SerializationProtocol(capture=capture, restore=lambda _: None)
    with pytest.raises(RuntimeError, match="initialize"):
        protocol.set_value(SERIALIZATION_CREATE, np.array([1], dtype="int32"))
    _assert_invariant(protocol, 0)

    # and after an earlier successful create, a failed create keeps that buffer
    protocol.set_value(SERIALIZATION_CREATE, np.array([1], dtype="int32"))
    kept = _buffer(protocol)
    calls = 0
    with pytest.raises(RuntimeError, match="initialize"):
        protocol.set_value(SERIALIZATION_CREATE, np.array([1], dtype="int32"))
    assert _buffer(protocol) is kept
    _assert_invariant(protocol, len(PAYLOAD))


# ---------------  announce and deliver  -----------------------------


@pytest.mark.parametrize(
    "src",
    [
        np.array([7], dtype="int64"),
        np.array([7], dtype="int32"),
        np.array([7], dtype="uint8"),
        np.array([[7]], dtype="int64"),
        7,
    ],
    ids=["int64", "int32", "uint8", "2d-one-element", "python-int"],
)
def test_announce_allocates_a_zero_filled_buffer_of_the_announced_length(
    protocol: SerializationProtocol, src: "np.ndarray | int"
):
    protocol.set_value(SERIALIZATION_SIZE, src)
    _assert_invariant(protocol, 7)
    assert not _buffer(protocol).any()


def test_announce_zero_is_allowed(protocol: SerializationProtocol):
    protocol.set_value(SERIALIZATION_CREATE, np.array([1], dtype="int32"))
    protocol.set_value(SERIALIZATION_SIZE, np.array([0], dtype="int64"))
    _assert_invariant(protocol, 0)


def test_announce_replaces_a_captured_buffer(protocol: SerializationProtocol):
    protocol.set_value(SERIALIZATION_CREATE, np.array([1], dtype="int32"))
    protocol.set_value(SERIALIZATION_SIZE, np.array([3], dtype="int64"))
    _assert_invariant(protocol, 3)
    assert _buffer(protocol).tobytes() == b"\0\0\0"


@pytest.mark.parametrize(
    "bad",
    [
        np.array([-1], dtype="int64"),
        np.array([], dtype="int64"),
        np.array([1, 2], dtype="int64"),
        np.array([4.0], dtype="float64"),
        np.array([True]),
        np.array(["4"]),
    ],
    ids=["negative", "empty", "two-values", "float", "bool", "string"],
)
def test_announce_rejects_bad_counts_and_keeps_the_buffer(
    protocol: SerializationProtocol, bad: np.ndarray
):
    protocol.set_value(SERIALIZATION_SIZE, np.array([5], dtype="int64"))
    before = _buffer(protocol)
    with pytest.raises(ValueError, match="ngen::serialization_size"):
        protocol.set_value(SERIALIZATION_SIZE, bad)
    assert _buffer(protocol) is before
    _assert_invariant(protocol, 5)


def test_announce_then_deliver_invokes_restore_with_the_delivered_bytes(
    protocol: SerializationProtocol, model: FakeModel
):
    protocol.set_value(SERIALIZATION_SIZE, np.array([len(PAYLOAD)], dtype="int64"))
    protocol.set_value(SERIALIZATION_STATE, np.frombuffer(PAYLOAD, dtype="uint8"))
    assert model.restored == [PAYLOAD]
    assert isinstance(model.restored[0], bytes)
    _assert_invariant(protocol, len(PAYLOAD))


@pytest.mark.parametrize(
    "delivered",
    [
        PAYLOAD,
        bytearray(PAYLOAD),
        memoryview(PAYLOAD),
        np.frombuffer(PAYLOAD, dtype="uint8"),
    ],
    ids=["bytes", "bytearray", "memoryview", "uint8-array"],
)
def test_delivery_accepts_any_byte_buffer(
    protocol: SerializationProtocol, model: FakeModel, delivered
):
    protocol.set_value(SERIALIZATION_SIZE, np.array([len(PAYLOAD)], dtype="int64"))
    protocol.set_value(SERIALIZATION_STATE, delivered)
    assert model.restored == [PAYLOAD]


@pytest.mark.parametrize(
    "announced", [0, len(PAYLOAD) - 1, len(PAYLOAD) + 1, 2 * len(PAYLOAD)]
)
def test_delivery_length_mismatch_raises_before_restore_is_called(
    protocol: SerializationProtocol, model: FakeModel, announced: int
):
    protocol.set_value(SERIALIZATION_SIZE, np.array([announced], dtype="int64"))
    before = _buffer(protocol)
    with pytest.raises(
        ValueError, match=f"delivered {len(PAYLOAD)} bytes but .* announced {announced}"
    ):
        protocol.set_value(SERIALIZATION_STATE, PAYLOAD)
    assert model.restored == []
    assert _buffer(protocol) is before
    _assert_invariant(protocol, announced)


def test_delivery_without_announcement_raises_unless_empty(
    protocol: SerializationProtocol, model: FakeModel
):
    with pytest.raises(ValueError, match="announced 0"):
        protocol.set_value(SERIALIZATION_STATE, PAYLOAD)
    assert model.restored == []
    _assert_invariant(protocol, 0)
    # a zero-length delivery matches the fresh (empty) buffer and reaches restore
    protocol.set_value(SERIALIZATION_STATE, b"")
    assert model.restored == [b""]
    _assert_invariant(protocol, 0)


def test_raising_restore_leaves_buffer_length_unchanged():
    model = FakeModel(restore_error=ValueError("fingerprint mismatch"))
    protocol = SerializationProtocol(capture=model.capture, restore=model.restore)
    protocol.set_value(SERIALIZATION_SIZE, np.array([len(PAYLOAD)], dtype="int64"))
    before = _buffer(protocol)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        protocol.set_value(SERIALIZATION_STATE, PAYLOAD)
    assert model.restored == []
    assert _buffer(protocol) is before
    _assert_invariant(protocol, len(PAYLOAD))
    # the object is still usable: free clears, and a fresh announce works
    protocol.set_value(SERIALIZATION_FREE, np.array([1], dtype="int32"))
    _assert_invariant(protocol, 0)
    protocol.set_value(SERIALIZATION_SIZE, np.array([2], dtype="int64"))
    _assert_invariant(protocol, 2)


def test_delivery_does_not_alias_the_delivered_array(
    protocol: SerializationProtocol, model: FakeModel
):
    delivered = np.frombuffer(PAYLOAD, dtype="uint8").copy()
    protocol.set_value(SERIALIZATION_SIZE, np.array([len(PAYLOAD)], dtype="int64"))
    protocol.set_value(SERIALIZATION_STATE, delivered)
    delivered[:] = 0
    assert model.restored == [PAYLOAD]
    assert _buffer(protocol).tobytes() == PAYLOAD


def test_full_ngen_save_then_restore_sequence(model: FakeModel):
    """create, read size, read state, free on the saving side; announce, deliver on the restoring side."""
    saver = SerializationProtocol(capture=model.capture, restore=model.restore)
    saver.set_value(SERIALIZATION_CREATE, np.array([1], dtype="int32"))
    size = int(saver.value(SERIALIZATION_SIZE)[0])
    state = saver.value(SERIALIZATION_STATE).copy()
    assert size == len(state)
    saver.set_value(SERIALIZATION_FREE, np.array([1], dtype="int32"))
    _assert_invariant(saver, 0)

    restorer = SerializationProtocol(capture=model.capture, restore=model.restore)
    restorer.set_value(SERIALIZATION_SIZE, np.array([size], dtype="int64"))
    # ngen sizes the delivered array from the announced byte count
    assert len(restorer.value(SERIALIZATION_STATE)) == size
    restorer.set_value(SERIALIZATION_STATE, state)
    assert model.restored == [PAYLOAD]
    _assert_invariant(restorer, size)


def test_public_names_are_exported():
    for name in proto.__all__:
        assert hasattr(proto, name)
