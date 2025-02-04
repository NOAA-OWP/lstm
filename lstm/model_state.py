from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy.typing as npt


@dataclass(slots=True)
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

    def __iter__(self) -> typing.Iterable[Var]:
        return self.vars()

    def __len__(self) -> int:
        return len(self._name_mapping)


# TODO: aaraney / reviewer: think of a better name
class StateValues:
    """
    Treat multiple `State` objects `value` method as if they were a single `StateValues` object.
    The first value successfully retrieved will be returned.

    Conforms to `Valuer` interface.
    """

    def __init__(self, *states: State):
        self.states = states

    def value(self, name: str) -> npt.NDArray:
        assert len(self.states) > 0, (
            "No State objects present. See initialization of `StateValues` instance."
        )
        errs = []
        for state in self.states:
            try:
                return state.value(name)
            except BaseException as e:
                errs.append(e)
        raise KeyError(errs)
