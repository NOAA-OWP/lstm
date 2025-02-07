#
# Copyright (C) 2025 Austin Raney, Lynker
#
# Author: Austin Raney <araney@lynker.com>
#
from __future__ import annotations

import typing

import numpy as np
import pytest

from lstm import model_state


def test_var_construction():
    """
    Test a Var can be created.

    `Var` utilizes the `slots` feature of `dataclass` if supported by the current Python version.
    Ensure version compatibility.
    """
    model_state.Var(name="some_variable", unit="m", value=np.array([0.0]))


Vars: typing.TypeAlias = tuple[model_state.Var]


@pytest.fixture
def vars() -> tuple[model_state.Var, ...]:
    vars = (
        model_state.Var("temperature", "K", np.array([25.0])),
        model_state.Var("pressure", "Pa", np.array([101325])),
    )
    return vars


StateAndVars: typing.TypeAlias = tuple[model_state.State, tuple[model_state.Var, ...]]


@pytest.fixture
def state_and_vars(vars: tuple[model_state.Var, ...]) -> StateAndVars:
    state = model_state.State(vars)
    return state, vars


def test_container_methods(state_and_vars: StateAndVars):
    state, vars = state_and_vars

    # __len__()
    assert len(vars) == len(state)

    # __iter__()
    # SAFETY:
    # Changed in version 3.7: Dictionary order is guaranteed to be insertion order.
    # This behavior was an implementation detail of CPython from 3.6.
    # see: https://docs.python.org/3.12/library/stdtypes.html#mapping-types-dict
    assert vars == tuple(state)
    # names()
    assert tuple(v.name for v in vars) == tuple(state.names())
    # vars()
    assert vars == tuple(state.vars())

    for var in vars:
        # __contains__()
        assert var.name in state

        # unit()
        assert var.unit == state.unit(var.name)

        # value()
        assert var.value is state.value(var.name)


def test_value_returns_reference(state_and_vars: StateAndVars):
    state, vars = state_and_vars
    for var in vars:
        assert var.value is state.value(var.name)


def test_value_at_indices():
    var = model_state.Var("temperature", "C", np.array([25.0]))
    state = model_state.State([var])

    idx = np.array([0], dtype="int")
    dest = np.empty(1, dtype=var.value.dtype)
    state.value_at_indices(var.name, dest, idx)

    np.testing.assert_array_equal(var.value, dest)
    assert not np.shares_memory(var.value, dest), "not set by copy"


def test_set_value_sets_inplace():
    var = model_state.Var("temperature", "C", np.array([25.0]))
    state = model_state.State([var])

    new_value = np.array([30.0], dtype=var.value.dtype)
    state.set_value(var.name, new_value)

    np.testing.assert_array_equal(state.value(var.name), new_value)
    assert var.value is state.value(var.name), "did not set in-place"


def test_set_value_at_indices():
    var = model_state.Var("temperature", "C", np.array([25.0]))
    state = model_state.State([var])

    idx = np.array([0], dtype="int")
    src = np.array([232.778], dtype=var.value.dtype)
    state.set_value_at_indices(var.name, idx, src)

    np.testing.assert_array_equal(var.value, src)
    assert not np.shares_memory(var.value, src), "not set by copy"
