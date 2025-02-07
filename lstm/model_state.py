#
# Copyright (C) 2025 Austin Raney, Lynker
#
# Author: Austin Raney <araney@lynker.com>
#
from __future__ import annotations

import sys
import typing
from dataclasses import dataclass

import numpy.typing as npt

if sys.version_info < (3, 10):
    import typing_extensions as typing
else:
    import typing

# `slots` feature added to of `dataclass` in 3.10
# see: https://docs.python.org/3.12/library/dataclasses.html#dataclasses.dataclass
if sys.version_info < (3, 10):
    dataclass_kwargs = {}
else:
    dataclass_kwargs = {"slots": True}


@dataclass(**dataclass_kwargs)
class Var:
    """
    State variable representation.
    """

    name: str
    unit: str
    value: npt.NDArray


class State:
    """
    State represents a collection of variables (`Var`), and provides methods to
    access and mutate a `Var`'s internal state. Method names and signatures are
    similar to the BMI's getter and setter methods to enable interoperability.

    A State is initialized with a collection of `Var`s, each of which has a
    name, unit, and value. State provides methods to retrieve the unit and
    value of a variable by name, modify values, and access specific indexed
    elements within arrays.
    """

    def __init__(self, vars: typing.Iterable[Var]):
        self._name_mapping: dict[str, Var] = {var.name: var for var in vars}

    def unit(self, name: str) -> str:
        """Given a variable name, return its unit"""
        return self._name_mapping[name].unit

    def value(self, name: str) -> npt.NDArray:
        """Given a variable name, return a value reference"""
        return self._name_mapping[name].value

    def value_at_indices(
        self, name: str, dest: npt.NDArray, indices: npt.NDArray
    ) -> npt.NDArray:
        # This must copy into dest!!!
        src = self.value(name)
        for i in range(indices.shape[0]):
            value_index = indices[i]
            dest[i] = src[value_index]
        return dest

    def set_value(self, name: str, value: npt.NDArray):
        self._name_mapping[name].value[:] = value

    def set_value_at_indices(self, name: str, inds: npt.NDArray, src: npt.NDArray):
        arr = self.value(name)
        for i in range(inds.shape[0]):
            arr[inds[i]] = src[i]

    def names(self) -> typing.Iterable[str]:
        yield from self._name_mapping

    def vars(self) -> typing.Iterable[Var]:
        yield from self._name_mapping.values()

    def __contains__(self, name: str) -> bool:
        """Return if the variable name is present in the collection."""
        return name in self._name_mapping

    def __iter__(self) -> typing.Iterator[Var]:
        return iter(self.vars())

    def __len__(self) -> int:
        return len(self._name_mapping)


class StateFacade:
    """
    Treat multiple `State` objects as if they were a single `State` object.
    The first value successfully retrieved will be returned.
    """

    def __init__(self, *states: State):
        self.states = states

    def unit(self, name: str) -> str:
        """Given a variable name, return its unit"""
        return state_proxy(self.states, State.unit, name=name)

    def value(self, name: str) -> npt.NDArray:
        """Given a variable name, return a value reference"""
        return state_proxy(self.states, State.value, name=name)

    def value_at_indices(
        self, name: str, dest: npt.NDArray, indices: npt.NDArray
    ) -> npt.NDArray:
        return state_proxy(
            self.states, State.value_at_indices, name=name, dest=dest, indices=indices
        )

    def set_value(self, name: str, value: npt.NDArray):
        return state_proxy(self.states, State.set_value, name=name, value=value)

    def set_value_at_indices(self, name: str, inds: npt.NDArray, src: npt.NDArray):
        return state_proxy(
            self.states, State.set_value_at_indices, name=name, inds=inds, src=src
        )

    def names(self) -> typing.Iterable[str]:
        for state in self.states:
            yield from state.names()

    def vars(self) -> typing.Iterable[Var]:
        for state in self.states:
            yield from state.vars()

    def __contains__(self, name: str) -> bool:
        """Return if the variable name is present in the collection."""
        return any(name in state for state in self.states)

    def __iter__(self) -> typing.Iterator[Var]:
        return iter(self.vars())

    def __len__(self) -> int:
        return sum(len(state) for state in self.states)


P = typing.ParamSpec("P")
R = typing.TypeVar("R")


def state_proxy(
    states: typing.Iterable[State],
    fn: typing.Callable[typing.Concatenate[State, P], R],
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    errs = []
    for state in states:
        try:
            return fn(state, *args, **kwargs)
        except BaseException as e:
            errs.append(e)
    if errs:
        raise KeyError(errs)
    raise RuntimeError(
        "No State objects present. See initialization of `StateValues` instance."
    )
