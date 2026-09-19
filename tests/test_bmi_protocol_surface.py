"""
Tests for the reserved ngen BMI Serialization Protocol variables as seen
through the BMI introspection surface of `lstm.bmi_lstm.bmi_LSTM`: names,
units, types, item sizes, byte counts, absence from the public variable
lists, and the values a fresh module reads. Every case holds both before and
after `initialize()`.

The protocol object's own behaviour (triggers, announcements, length checks)
is covered with fake callables in `test_serialization_protocol.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lstm import bmi_lstm
from lstm import serialization_protocol as protocol

from helpers import initialized

RESERVED = {
    protocol.SERIALIZATION_CREATE: ("ngen::trigger", "int32", 4),
    protocol.SERIALIZATION_FREE: ("ngen::trigger", "int32", 4),
    protocol.SERIALIZATION_SIZE: ("bytes", "int64", 8),
    protocol.SERIALIZATION_STATE: ("ngen::opaque", "uint8", 1),
}
"""expected (unit, type, itemsize) per reserved name, per the ngen protocol"""


def test_reserved_name_constants_are_exact():
    assert protocol.SERIALIZATION_CREATE == "ngen::serialization_create"
    assert protocol.SERIALIZATION_FREE == "ngen::serialization_free"
    assert protocol.SERIALIZATION_SIZE == "ngen::serialization_size"
    assert protocol.SERIALIZATION_STATE == "ngen::serialization_state"
    assert protocol.SERIALIZATION_VAR_NAMES == tuple(RESERVED)
    assert protocol.SERIALIZATION_TRIGGER_UNIT == "ngen::trigger"
    assert protocol.SERIALIZATION_SIZE_UNIT == "bytes"
    assert protocol.SERIALIZATION_OPAQUE_UNIT == "ngen::opaque"


@pytest.fixture(params=["uninitialized", "initialized"])
def module(request: pytest.FixtureRequest, golden_config: Path) -> bmi_lstm.bmi_LSTM:
    """The reserved names must resolve both before and after `initialize()`."""
    if request.param == "initialized":
        return initialized(golden_config)
    return bmi_lstm.bmi_LSTM()


@pytest.mark.parametrize("name", list(RESERVED))
def test_reserved_units_exact(module: bmi_lstm.bmi_LSTM, name: str):
    unit, _, _ = RESERVED[name]
    # ngen's support probe compares this string exactly
    assert module.get_var_units(name) == unit


@pytest.mark.parametrize("name", list(RESERVED))
def test_reserved_type_and_itemsize(module: bmi_lstm.bmi_LSTM, name: str):
    _, dtype, itemsize = RESERVED[name]
    assert module.get_var_type(name) == dtype
    assert module.get_var_itemsize(name) == itemsize
    assert module.get_value_ptr(name).dtype == np.dtype(dtype)


@pytest.mark.parametrize("name", list(RESERVED))
def test_reserved_nbytes_matches_ptr(module: bmi_lstm.bmi_LSTM, name: str):
    ptr = module.get_value_ptr(name)
    assert module.get_var_nbytes(name) == ptr.nbytes
    # ngen's Python adapter sizes arrays from nbytes / itemsize
    assert module.get_var_nbytes(name) // module.get_var_itemsize(name) == len(ptr)


def test_reserved_names_absent_from_var_lists_and_counts(module: bmi_lstm.bmi_LSTM):
    input_names = module.get_input_var_names()
    output_names = module.get_output_var_names()
    for name in RESERVED:
        assert name not in input_names
        assert name not in output_names
    assert module.get_input_item_count() == len(input_names)
    assert module.get_output_item_count() == len(output_names)
    assert not any(n.startswith("ngen::") for n in (*input_names, *output_names))


def test_public_var_lists_unchanged_by_protocol(golden_config: Path):
    """Adding the reserved names must not alter the pre-existing BMI surface."""
    model = initialized(golden_config)
    assert model.get_input_var_names() == tuple(
        name for name, _ in bmi_lstm._dynamic_input_vars
    )
    assert model.get_output_var_names() == tuple(
        name for name, _ in bmi_lstm._output_vars
    )
    assert model.get_input_item_count() == len(bmi_lstm._dynamic_input_vars)
    assert model.get_output_item_count() == len(bmi_lstm._output_vars)


@pytest.mark.parametrize("name", list(RESERVED))
def test_reserved_names_have_no_grid_or_location(
    module: bmi_lstm.bmi_LSTM, name: str
):
    with pytest.raises(KeyError):
        module.get_var_grid(name)
    with pytest.raises(KeyError):
        module.get_var_location(name)


def test_unknown_name_still_raises_same_as_reserved(module: bmi_lstm.bmi_LSTM):
    """Reserved names get the standard unknown-variable signal from grid/location."""
    with pytest.raises(KeyError):
        module.get_var_grid("ngen::not_a_variable")
    with pytest.raises(KeyError):
        module.get_var_location("ngen::not_a_variable")
    with pytest.raises(KeyError):
        module.get_var_units("ngen::not_a_variable")


def test_size_reads_zero_and_state_reads_empty_when_fresh(module: bmi_lstm.bmi_LSTM):
    size_ptr = module.get_value_ptr(protocol.SERIALIZATION_SIZE)
    assert size_ptr.shape == (1,)
    assert size_ptr[0] == 0

    size_copy = module.get_value(protocol.SERIALIZATION_SIZE, np.empty(1, dtype="int64"))
    assert size_copy.dtype == np.dtype("int64")
    assert size_copy[0] == 0

    state_ptr = module.get_value_ptr(protocol.SERIALIZATION_STATE)
    assert state_ptr.dtype == np.dtype("uint8")
    assert state_ptr.shape == (0,)
    assert module.get_var_nbytes(protocol.SERIALIZATION_STATE) == 0

    state = module.get_value(protocol.SERIALIZATION_STATE, np.empty(0, dtype="uint8"))
    assert state.shape == (0,)


@pytest.mark.parametrize("name", list(RESERVED))
def test_reserved_reads_are_non_mutating(module: bmi_lstm.bmi_LSTM, name: str):
    first = module.get_value_ptr(name)
    before = first.copy()
    for _ in range(3):
        again = module.get_value_ptr(name)
        assert again is first
        np.testing.assert_array_equal(again, before)
        copied = module.get_value(name, np.empty_like(before))
        np.testing.assert_array_equal(copied, before)
    assert module.get_var_units(name) == RESERVED[name][0]
    assert module.get_var_nbytes(name) == before.nbytes


def test_reserved_state_is_per_instance():
    a = bmi_lstm.bmi_LSTM()
    b = bmi_lstm.bmi_LSTM()
    for name in RESERVED:
        assert a.get_value_ptr(name) is not b.get_value_ptr(name)
