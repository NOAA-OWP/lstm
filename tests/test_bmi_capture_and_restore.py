"""
Tests for the ngen BMI Serialization Protocol save and restore sequences on a
real `lstm.bmi_lstm.bmi_LSTM`: capturing state through the create trigger,
releasing it through free and finalize, restoring it through the size
announcement and state delivery, rejecting payloads that do not fit, and the
member-owned state transfer underneath.

Cases that only exercise the protocol object (trigger values, announced
counts, length mismatches, bytes-like deliveries) live with the fake-callable
tests in `test_serialization_protocol.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from lstm import bmi_lstm
from lstm import serialization_codec as codec
from lstm import serialization_protocol as protocol

from helpers import (
    announce,
    assert_member_arrays_equal,
    assert_size_equals_buffer_length,
    assert_untouched,
    captured_after,
    create,
    deliver,
    free,
    initialized,
    member_arrays,
    observed,
    outputs,
    restore,
    size,
    state_bytes,
    step,
    synthetic_forcing,
)


# ---------------  capture and release (create / free triggers)  -----------------------------


def test_create_size_equals_state_length_and_packed_bytes(config: Path):
    model = initialized(config)
    for inputs in synthetic_forcing(3):
        step(model, inputs)

    create(model)

    packed = codec.pack(model.snapshot())
    state_ptr = model.get_value_ptr(protocol.SERIALIZATION_STATE)
    assert state_ptr.dtype == np.dtype("uint8")
    assert size(model) == len(state_ptr) == len(packed)
    assert size(model) == model.get_var_nbytes(protocol.SERIALIZATION_STATE)
    assert size(model) > codec.HEADER_SIZE
    assert state_ptr.tobytes() == packed
    assert packed.startswith(codec.MAGIC)


def test_consecutive_size_and_state_reads_are_identical(config: Path):
    model = initialized(config)
    step(model, synthetic_forcing(1)[0])
    create(model)

    size_a = model.get_value(protocol.SERIALIZATION_SIZE, np.empty(1, dtype="int64"))
    size_b = model.get_value(protocol.SERIALIZATION_SIZE, np.empty(1, dtype="int64"))
    np.testing.assert_array_equal(size_a, size_b)
    assert model.get_value_ptr(protocol.SERIALIZATION_SIZE) is model.get_value_ptr(
        protocol.SERIALIZATION_SIZE
    )

    n = model.get_var_nbytes(protocol.SERIALIZATION_STATE) // model.get_var_itemsize(
        protocol.SERIALIZATION_STATE
    )
    state_a = model.get_value(protocol.SERIALIZATION_STATE, np.empty(n, dtype="uint8"))
    state_b = model.get_value(protocol.SERIALIZATION_STATE, np.empty(n, dtype="uint8"))
    np.testing.assert_array_equal(state_a, state_b)
    assert model.get_value_ptr(protocol.SERIALIZATION_STATE) is model.get_value_ptr(
        protocol.SERIALIZATION_STATE
    )
    assert state_a.tobytes() == state_bytes(model)


def test_create_free_then_update_matches_update_without_capture(config: Path):
    plain = initialized(config)
    captured = initialized(config)

    for inputs in synthetic_forcing(6, seed=1):
        expected = step(plain, inputs)
        create(captured)
        free(captured)
        actual = step(captured, inputs)
        # bitwise: capturing must not perturb the computed state
        assert actual == expected

    assert captured.get_current_time() == plain.get_current_time()
    for (h_a, c_a), (h_b, c_b) in zip(member_arrays(plain), member_arrays(captured)):
        np.testing.assert_array_equal(h_a, h_b)
        np.testing.assert_array_equal(c_a, c_b)


def test_capture_between_updates_does_not_change_state(config: Path):
    """A create left outstanding (no free) across updates is equally inert."""
    plain = initialized(config)
    captured = initialized(config)

    for inputs in synthetic_forcing(4, seed=2):
        expected = step(plain, inputs)
        actual = step(captured, inputs)
        assert actual == expected
        create(captured)  # outstanding until the next iteration

    for (h_a, c_a), (h_b, c_b) in zip(member_arrays(plain), member_arrays(captured)):
        np.testing.assert_array_equal(h_a, h_b)
        np.testing.assert_array_equal(c_a, c_b)


def test_create_before_initialize_raises_and_free_still_safe():
    model = bmi_lstm.bmi_LSTM()
    with pytest.raises(RuntimeError, match="initialize"):
        create(model)
    # a failed create leaves the buffer untouched ...
    assert size(model) == 0
    assert len(model.get_value_ptr(protocol.SERIALIZATION_STATE)) == 0
    # ... and free afterwards is still safe
    free(model)
    assert size(model) == 0
    with pytest.raises(RuntimeError, match="initialize"):
        model.snapshot()


@pytest.mark.parametrize("steps", [0, 1, 5])
def test_captured_bytes_unpack_to_current_state(config: Path, steps: int):
    model = initialized(config)
    for inputs in synthetic_forcing(steps, seed=3):
        step(model, inputs)

    create(model)
    snapshot = codec.unpack(state_bytes(model))

    assert snapshot.timestep == steps
    assert model.get_current_time() == steps * model.get_time_step()
    assert snapshot.fingerprint == model.fingerprint

    expected_outputs = np.array(
        [model.get_value_ptr(name)[0] for name in model.get_output_var_names()],
        dtype="float64",
    )
    np.testing.assert_array_equal(snapshot.outputs, expected_outputs)
    if steps == 0:
        np.testing.assert_array_equal(snapshot.outputs, np.zeros(2))

    assert len(snapshot.members) == len(model.ensemble_members)
    for (hidden, cell), (h_t, c_t) in zip(snapshot.members, member_arrays(model)):
        assert hidden.dtype == np.dtype("float32")
        assert hidden.shape == h_t.shape == (model.ensemble_members[0].model.hidden_size,)
        np.testing.assert_array_equal(hidden, h_t)
        np.testing.assert_array_equal(cell, c_t)
        if steps == 0:
            assert not hidden.any() and not cell.any()
        else:
            assert hidden.any() and cell.any()


def test_snapshot_arrays_do_not_alias_member_tensors(golden_config: Path):
    model = initialized(golden_config)
    step(model, synthetic_forcing(1)[0])
    before = member_arrays(model)

    snapshot = model.snapshot()
    for hidden, cell in snapshot.members:
        hidden[:] = 123.0
        cell[:] = 456.0
    snapshot.outputs[:] = -1.0

    for (h_t, c_t), (h_before, c_before) in zip(member_arrays(model), before):
        np.testing.assert_array_equal(h_t, h_before)
        np.testing.assert_array_equal(c_t, c_before)
    for name in model.get_output_var_names():
        assert model.get_value_ptr(name)[0] != -1.0


def test_free_releases_captured_buffer(golden_config: Path):
    model = initialized(golden_config)
    step(model, synthetic_forcing(1)[0])
    create(model)
    assert size(model) > 0

    free(model)
    assert size(model) == 0
    state_ptr = model.get_value_ptr(protocol.SERIALIZATION_STATE)
    assert state_ptr.dtype == np.dtype("uint8")
    assert state_ptr.shape == (0,)
    assert model.get_var_nbytes(protocol.SERIALIZATION_STATE) == 0
    # type and units survive a release
    assert model.get_var_type(protocol.SERIALIZATION_STATE) == "uint8"
    assert model.get_var_units(protocol.SERIALIZATION_STATE) == "ngen::opaque"


def test_finalize_releases_captured_buffer(golden_config: Path):
    model = initialized(golden_config)
    step(model, synthetic_forcing(1)[0])
    create(model)
    assert size(model) > 0

    model.finalize()
    assert size(model) == 0
    assert len(model.get_value_ptr(protocol.SERIALIZATION_STATE)) == 0


def test_finalize_safe_without_initialize():
    model = bmi_lstm.bmi_LSTM()
    model.finalize()
    assert size(model) == 0


def test_recapture_replaces_buffer_with_new_state(golden_config: Path):
    model = initialized(golden_config)
    forcing = synthetic_forcing(2, seed=4)
    step(model, forcing[0])
    create(model)
    first = state_bytes(model)

    step(model, forcing[1])
    create(model)
    second = state_bytes(model)

    assert len(first) == len(second) == size(model)
    assert first != second
    assert codec.unpack(first).timestep == 1
    assert codec.unpack(second).timestep == 2


def test_captured_buffer_is_independent_of_later_updates(golden_config: Path):
    model = initialized(golden_config)
    forcing = synthetic_forcing(3, seed=5)
    step(model, forcing[0])
    create(model)
    ptr = model.get_value_ptr(protocol.SERIALIZATION_STATE)
    frozen = ptr.tobytes()

    for inputs in forcing[1:]:
        step(model, inputs)

    # the buffer is a snapshot, not a view onto the live tensors
    assert model.get_value_ptr(protocol.SERIALIZATION_STATE) is ptr
    assert ptr.tobytes() == frozen
    assert codec.unpack(frozen).timestep == 1
    assert size(model) == len(frozen)


def test_size_and_state_stay_within_serialization_state_only(golden_config: Path):
    """Capturing must not leak the reserved names into the public var lists."""
    model = initialized(golden_config)
    create(model)
    for name in protocol.SERIALIZATION_VAR_NAMES:
        assert name not in model.get_input_var_names()
        assert name not in model.get_output_var_names()
    assert model.get_input_item_count() == len(bmi_lstm._dynamic_input_vars)
    assert model.get_output_item_count() == len(bmi_lstm._output_vars)


# ---------------  restore (size announcement and state delivery)  -----------------------------


def test_announce_then_deliver_restores_members_and_outputs(config: Path):
    source, payload = captured_after(config, steps=4)
    target = initialized(config)
    assert outputs(target) != outputs(source)

    announce(target, len(payload))
    # between the two calls ngen sizes the incoming array from nbytes / itemsize
    assert target.get_var_nbytes(protocol.SERIALIZATION_STATE) == len(payload)
    assert target.get_var_itemsize(protocol.SERIALIZATION_STATE) == 1
    assert size(target) == len(payload)
    # the announcement allocates the buffer ngen is about to fill
    assert_size_equals_buffer_length(target)

    deliver(target, payload)

    assert_size_equals_buffer_length(target)
    assert_member_arrays_equal(member_arrays(target), member_arrays(source))
    assert outputs(target) == outputs(source)


def test_restore_does_not_apply_timestep_or_clock(config: Path):
    source, payload = captured_after(config, steps=5)
    assert source.get_current_time() == 5 * bmi_lstm.bmi_LSTM._timestep_size_s
    assert codec.unpack(payload).timestep == 5

    target = initialized(config)
    restore(target, payload)
    assert target._timestep == 0
    assert target.get_current_time() == 0.0

    step(target, synthetic_forcing(1)[0])
    assert target.get_current_time() == bmi_lstm.bmi_LSTM._timestep_size_s


def test_nbytes_after_create_equals_buffer_length_and_free_resets(config: Path):
    """The announced count and the capture bookkeeping share the size variable."""
    model = initialized(config)
    announce(model, 99)
    create(model)
    n = len(model.get_value_ptr(protocol.SERIALIZATION_STATE))
    assert n != 99
    assert model.get_var_nbytes(protocol.SERIALIZATION_STATE) == n == size(model)
    free(model)
    assert model.get_var_nbytes(protocol.SERIALIZATION_STATE) == 0


def test_restore_into_differently_configured_module_raises_and_leaves_state(
    two_member_config: Path,
    golden_config: Path,
):
    for src_cfg, dst_cfg in [
        (golden_config, two_member_config),
        (two_member_config, golden_config),
    ]:
        _, payload = captured_after(src_cfg, steps=3)
        target = initialized(dst_cfg)
        before = observed(target)
        # initialized values: zero states and zero outputs
        assert all(not h.any() and not c.any() for h, c in before["members"])
        assert all(v == 0.0 for v in before["outputs"].values())

        announce(target, len(payload))
        with pytest.raises(codec.PayloadError, match="fingerprint"):
            deliver(target, payload)

        assert_untouched(target, before)
        assert target.get_current_time() == 0.0


def test_restore_into_module_with_computed_state_leaves_it_on_failure(
    two_member_config: Path,
    golden_config: Path,
):
    """A module that has already stepped keeps its non-trivial state on rejection."""
    _, payload = captured_after(golden_config, steps=3)
    target = initialized(two_member_config)
    for inputs in synthetic_forcing(2, seed=7):
        step(target, inputs)
    before = observed(target)
    assert any(h.any() for h, _ in before["members"])

    with pytest.raises(codec.PayloadError, match="fingerprint"):
        restore(target, payload)
    assert_untouched(target, before)


@pytest.mark.parametrize("cut", [codec.HEADER_SIZE - 1, codec.HEADER_SIZE + 3, -1])
def test_truncated_payload_raises_and_leaves_state(config: Path, cut: int):
    _, payload = captured_after(config, steps=2)
    truncated = payload[:cut]
    assert len(truncated) < len(payload)

    target = initialized(config)
    for inputs in synthetic_forcing(2, seed=3):
        step(target, inputs)
    before = observed(target)

    with pytest.raises(codec.PayloadError) as info:
        restore(target, truncated)
    assert "truncated" in str(info.value) or "short" in str(info.value)
    assert_untouched(target, before)


def test_over_long_and_garbage_payloads_raise_and_leave_state(config: Path):
    _, payload = captured_after(config, steps=2)
    target = initialized(config)
    before = observed(target)

    with pytest.raises(codec.PayloadError, match="trailing"):
        restore(target, payload + b"\x00")
    assert_untouched(target, before)

    garbage = b"NOTLSTM!" + payload[8:]
    with pytest.raises(codec.PayloadError, match="magic"):
        restore(target, garbage)
    assert_untouched(target, before)

    with pytest.raises(codec.PayloadError):
        restore(target, b"")
    assert_untouched(target, before)


def test_restore_before_initialize_raises(golden_config: Path):
    _, payload = captured_after(golden_config, steps=1)
    model = bmi_lstm.bmi_LSTM()
    announce(model, len(payload))
    with pytest.raises(RuntimeError, match="initialize"):
        deliver(model, payload)
    # the announcement is still recorded and free is still safe afterwards
    assert size(model) == len(payload)
    free(model)
    assert size(model) == 0


def test_size_equals_buffer_length_through_capture_announce_and_deliver(config: Path):
    """The size variable tracks the buffer length at every protocol transition."""
    source, payload = captured_after(config, steps=5)
    target = initialized(config)
    for inputs in synthetic_forcing(2, seed=11):
        step(target, inputs)
    assert_size_equals_buffer_length(target)

    create(target)
    captured = state_bytes(target)
    assert captured != payload
    assert size(target) == len(captured)
    assert_size_equals_buffer_length(target)

    announce(target, len(payload))
    assert size(target) == len(payload)
    assert_size_equals_buffer_length(target)

    deliver(target, payload)
    assert size(target) == len(payload)
    assert_size_equals_buffer_length(target)
    assert_member_arrays_equal(member_arrays(target), member_arrays(source))
    assert outputs(target) == outputs(source)


def test_restore_then_continue_matches_uninterrupted_run(config: Path):
    forcing = synthetic_forcing(6, seed=42)

    reference = initialized(config)
    expected = [step(reference, inputs) for inputs in forcing]

    first_half = initialized(config)
    observed_outputs = [step(first_half, inputs) for inputs in forcing[:3]]
    create(first_half)
    payload = state_bytes(first_half)
    free(first_half)

    second_half = initialized(config)
    restore(second_half, payload)
    observed_outputs.extend(step(second_half, inputs) for inputs in forcing[3:])

    assert observed_outputs == expected
    assert_member_arrays_equal(member_arrays(second_half), member_arrays(reference))


def test_apply_snapshot_direct_restores_members_and_outputs(config: Path):
    """Standalone callers may bypass set_value and apply a decoded snapshot directly."""
    source, payload = captured_after(config, steps=3)

    target = initialized(config)
    target.apply_snapshot(codec.unpack(payload))
    assert_member_arrays_equal(member_arrays(target), member_arrays(source))
    assert outputs(target) == outputs(source)


def test_matching_fingerprint_with_wrong_output_count_raises(config: Path):
    source, _ = captured_after(config, steps=2)
    snapshot = source.snapshot()
    snapshot.outputs = np.append(snapshot.outputs, 1.0)
    payload = codec.pack(snapshot)

    target = initialized(config)
    before = observed(target)
    with pytest.raises(codec.PayloadError, match="output"):
        restore(target, payload)
    assert_untouched(target, before)


def test_matching_fingerprint_with_wrong_hidden_size_raises(config: Path):
    source, _ = captured_after(config, steps=2)
    snapshot = source.snapshot()
    hidden, cell = snapshot.members[-1]
    snapshot.members[-1] = (np.append(hidden, np.float32(0)), np.append(cell, np.float32(0)))
    payload = codec.pack(snapshot)

    target = initialized(config)
    before = observed(target)
    with pytest.raises(codec.PayloadError, match="hidden size"):
        restore(target, payload)
    assert_untouched(target, before)


def test_matching_fingerprint_with_wrong_member_count_raises(config: Path):
    source, _ = captured_after(config, steps=2)
    snapshot = source.snapshot()
    snapshot.members = snapshot.members + [snapshot.members[0]]
    payload = codec.pack(snapshot)

    target = initialized(config)
    before = observed(target)
    with pytest.raises(codec.PayloadError, match="ensemble members"):
        restore(target, payload)
    assert_untouched(target, before)


def test_apply_snapshot_rejects_foreign_fingerprint_before_mutation(config: Path):
    source, _ = captured_after(config, steps=2)
    snapshot = source.snapshot()
    snapshot.fingerprint = snapshot.fingerprint + b";tampered"

    target = initialized(config)
    before = observed(target)
    with pytest.raises(codec.PayloadError, match="fingerprint"):
        target.apply_snapshot(snapshot)
    assert_untouched(target, before)


def test_reserved_names_stay_out_of_var_lists_after_restore(config: Path):
    _, payload = captured_after(config, steps=1)
    target = initialized(config)
    restore(target, payload)
    for name in protocol.SERIALIZATION_VAR_NAMES:
        assert name not in target.get_input_var_names()
        assert name not in target.get_output_var_names()


# ---------------  ensemble member state transfer  -----------------------------


def test_member_state_arrays_are_flat_float32_copies(golden_config: Path):
    model = initialized(golden_config)
    member = model.ensemble_members[0]
    hidden, cell = member.state_arrays()
    assert hidden.dtype == cell.dtype == np.float32
    assert hidden.shape == cell.shape == (member.model.hidden_size,)
    assert member.model.hidden_size == 126
    # copies: writing to the returned arrays does not touch the member
    hidden[0] = 123.0
    cell[0] = 456.0
    again_hidden, again_cell = member.state_arrays()
    assert again_hidden[0] != 123.0
    assert again_cell[0] != 456.0


def test_member_set_state_arrays_keeps_tensor_shape_and_dtype(golden_config: Path):
    model = initialized(golden_config)
    member = model.ensemble_members[0]
    shape_before = tuple(member.h_t.shape)
    hidden = np.arange(member.model.hidden_size, dtype="float64")
    cell = -np.arange(member.model.hidden_size, dtype="float64")

    member.set_state_arrays(hidden, cell)

    assert tuple(member.h_t.shape) == tuple(member.c_t.shape) == shape_before
    assert member.h_t.dtype == member.c_t.dtype == torch.float32
    got_hidden, got_cell = member.state_arrays()
    assert np.array_equal(got_hidden, hidden.astype("float32"))
    assert np.array_equal(got_cell, cell.astype("float32"))


@pytest.mark.parametrize("bad", ["hidden", "cell"])
def test_member_set_state_arrays_rejects_wrong_size_without_mutation(bad: str, golden_config: Path):
    model = initialized(golden_config)
    member = model.ensemble_members[0]
    before = member.state_arrays()
    good = np.ones(member.model.hidden_size, dtype="float32")
    wrong = np.ones(member.model.hidden_size + 1, dtype="float32")
    hidden, cell = (wrong, good) if bad == "hidden" else (good, wrong)

    with pytest.raises(ValueError, match=f"{bad} state has {member.model.hidden_size + 1} elements"):
        member.set_state_arrays(hidden, cell)

    assert_member_arrays_equal([member.state_arrays()], [before])
