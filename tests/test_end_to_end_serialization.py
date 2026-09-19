"""
End-to-end equivalence tests for the ngen BMI Serialization Protocol on real
forcing: a run that is interrupted, captured, restored into a fresh module, and
continued must reproduce an uninterrupted run bitwise, for both outputs and for
every member's hidden and cell tensors.

These reuse the golden integration test's basin config and 24 hours of the
bundled NLDAS forcing, driven through the same forcing-name mapping, for both
the single-member golden config and a temporary two-member config (see
`conftest.py`). Payloads must also never cross between single- and two-member
modules.
"""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

from lstm import bmi_lstm
from lstm import serialization_codec as codec
from lstm import serialization_protocol as protocol

from helpers import (
    BASIN_ID,
    FORCING_FILE,
    FORCING_VARIABLE_NAME_MAPPING,
    SPLIT_STEP,
    TOTAL_STEPS,
    Forcing,
    announce,
    assert_members_equal,
    assert_untouched,
    capture_and_free,
    deliver,
    initialized,
    observed,
    restore,
    run,
)


def test_config_builds_the_expected_member_count(config: Path, member_count: int):
    assert len(initialized(config).ensemble_members) == member_count


def test_split_and_restore_matches_uninterrupted_run(
    config: Path, member_count: int, nldas_forcing: Forcing
):
    """
    24 hours uninterrupted versus 12 hours, capture, restore into a fresh
    module, and 12 more hours: both output variables and every member's
    hidden and cell tensors must match bitwise.
    """
    assert len(nldas_forcing) == TOTAL_STEPS

    # an uninterrupted 24-step run recording both outputs every step
    uninterrupted = initialized(config)
    assert len(uninterrupted.ensemble_members) == member_count
    expected = run(uninterrupted, nldas_forcing)
    assert set(expected) == {name for name, _ in bmi_lstm._output_vars}
    assert all(len(values) == TOTAL_STEPS for values in expected.values())
    # the golden dataset actually exercises the model (not an all-zero run)
    assert any(value != 0.0 for value in expected["land_surface_water__runoff_depth"])

    # first half: run 12 steps, then perform the ngen save sequence
    first_half = initialized(config)
    observed_outputs = run(first_half, nldas_forcing[:SPLIT_STEP])
    size, state = capture_and_free(first_half)
    assert size == len(state) > 0
    assert first_half.get_value_ptr(protocol.SERIALIZATION_STATE).size == 0
    assert len(codec.unpack(state).members) == member_count

    # second half: fresh module, ngen restore sequence, remaining 12 steps
    second_half = initialized(config)
    assert len(second_half.ensemble_members) == member_count
    restore(second_half, state)
    continued = run(second_half, nldas_forcing[SPLIT_STEP:])
    for name in observed_outputs:
        observed_outputs[name].extend(continued[name])

    # bitwise equality of all 24 recorded values of each output variable
    assert observed_outputs.keys() == expected.keys()
    for name in expected:
        assert len(observed_outputs[name]) == TOTAL_STEPS
        assert observed_outputs[name] == expected[name], name

    # exact equality of every member's hidden and cell tensors after step 24
    assert len(second_half.ensemble_members) == len(uninterrupted.ensemble_members)
    assert_members_equal(second_half, uninterrupted)


def test_two_member_payload_is_larger_than_single_member_payload(
    golden_config: Path, two_member_config: Path, nldas_forcing: Forcing
):
    """The payload grows with the member count: two members carry more state than one."""
    payloads = {}
    for members, cfg in [(1, golden_config), (2, two_member_config)]:
        model = initialized(cfg)
        run(model, nldas_forcing[:SPLIT_STEP])
        size, state = capture_and_free(model)
        assert size == len(state)
        assert len(codec.unpack(state).members) == members
        payloads[members] = state
    assert len(payloads[2]) > len(payloads[1])


def test_restored_module_matches_source_at_split(
    config: Path, member_count: int, nldas_forcing: Forcing
):
    """Immediately after restore the fresh module equals the source at step 12."""
    source = initialized(config)
    at_split = run(source, nldas_forcing[:SPLIT_STEP])
    size, state = capture_and_free(source)

    target = initialized(config)
    announce(target, size)
    deliver(target, state)

    assert_members_equal(target, source)
    assert len(target.ensemble_members) == member_count
    for name, values in at_split.items():
        assert float(target.get_value_ptr(name)[0]) == values[-1]
    # the timestep is carried in the payload but not applied on restore
    assert target.get_current_time() == target.get_start_time()
    assert source.get_current_time() == SPLIT_STEP * source.get_time_step()


@pytest.mark.parametrize(
    "source_kind, target_kind",
    [("two", "single"), ("single", "two")],
    ids=["two_member_payload_into_single_member", "single_member_payload_into_two_member"],
)
def test_cross_member_count_restore_raises_and_leaves_target_untouched(
    source_kind: str,
    target_kind: str,
    golden_config: Path,
    two_member_config: Path,
    nldas_forcing: Forcing,
):
    """
    Restoring a two-member payload into a single-member module, and vice
    versa, is rejected on fingerprint mismatch before anything is mutated,
    whether the target is freshly initialized or has already stepped.
    """
    configs = {"single": golden_config, "two": two_member_config}

    source = initialized(configs[source_kind])
    run(source, nldas_forcing[:SPLIT_STEP])
    size, state = capture_and_free(source)

    # freshly initialized target: zeros stay zeros
    fresh = initialized(configs[target_kind])
    before = observed(fresh)
    assert all(not h.any() and not c.any() for h, c in before["members"])
    assert all(v == 0.0 for v in before["outputs"].values())
    announce(fresh, size)
    with pytest.raises(codec.PayloadError, match="fingerprint"):
        deliver(fresh, state)
    assert_untouched(fresh, before)
    assert fresh.get_current_time() == fresh.get_start_time()

    # a target that has already stepped keeps its non-trivial state
    stepped = initialized(configs[target_kind])
    run(stepped, nldas_forcing[:3])
    before = observed(stepped)
    assert any(h.any() for h, _ in before["members"])
    announce(stepped, size)
    with pytest.raises(codec.PayloadError, match="fingerprint"):
        deliver(stepped, state)
    assert_untouched(stepped, before)

    # and the rejected target still runs normally afterwards, matching an untouched twin
    twin = initialized(configs[target_kind])
    run(twin, nldas_forcing[:3])
    assert run(stepped, nldas_forcing[3:6]) == run(twin, nldas_forcing[3:6])
    assert_members_equal(stepped, twin)


def test_upfront_forcing_read_matches_golden_style_per_step_reads(
    golden_config: Path, nldas_forcing: Forcing
):
    """
    Guard the forcing read used here against the golden test's driving loop:
    reading the 24 forcing values once up front and passing them as float64
    arrays yields the same runoff series as the golden test's per-step reads.
    """
    upfront = initialized(golden_config)
    recorded = run(upfront, nldas_forcing)["land_surface_water__runoff_depth"]

    with nc.Dataset(FORCING_FILE, "r") as ds:
        (basin_idxs,) = np.where(ds.variables["basin"][:] == BASIN_ID)
        basin_idx = basin_idxs[0]
        per_step = initialized(golden_config)
        golden_style = np.zeros(TOTAL_STEPS)
        for ts in range(TOTAL_STEPS):
            for nc_name, bmi_name in FORCING_VARIABLE_NAME_MAPPING.items():
                per_step.set_value(bmi_name, ds.variables[nc_name][basin_idx, ts])
            per_step.update()
            per_step.get_value("land_surface_water__runoff_depth", golden_style[ts : ts + 1])

    assert np.array_equal(np.array(recorded, dtype="float64"), golden_style)
