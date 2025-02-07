#
# Copyright (C) 2025 Austin Raney, Lynker
#
# Author: Austin Raney <araney@lynker.com>
#
from __future__ import annotations

import functools
import sys

import bmipy
import numpy as np

if sys.version_info < (3, 10):
    import typing_extensions as typing
else:
    import typing

_R = typing.TypeVar("_R")
_P = typing.ParamSpec("_P")


def raise_not_implemented(
    fn: typing.Callable[_P, _R],
) -> typing.Callable[_P, typing.NoReturn]:
    name = fn.__name__

    @functools.wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> typing.NoReturn:
        raise NotImplementedError(name)

    return wrapper


class BmiBase(bmipy.Bmi):
    # Uniform rectilinear
    @raise_not_implemented
    def get_grid_shape(self, grid: int, shape: np.ndarray) -> typing.NoReturn: ...
    @raise_not_implemented
    def get_grid_spacing(self, grid: int, spacing: np.ndarray) -> typing.NoReturn: ...
    @raise_not_implemented
    def get_grid_origin(self, grid: int, origin: np.ndarray) -> typing.NoReturn: ...
    # Non-uniform rectilinear, curvilinear
    @raise_not_implemented
    def get_grid_x(self, grid: int, x: np.ndarray) -> typing.NoReturn: ...
    @raise_not_implemented
    def get_grid_y(self, grid: int, y: np.ndarray) -> typing.NoReturn: ...
    @raise_not_implemented
    def get_grid_z(self, grid: int, z: np.ndarray) -> typing.NoReturn: ...
    @raise_not_implemented
    def get_grid_node_count(self, grid: int) -> typing.NoReturn: ...
    @raise_not_implemented
    def get_grid_edge_count(self, grid: int) -> typing.NoReturn: ...
    @raise_not_implemented
    def get_grid_face_count(self, grid: int) -> typing.NoReturn: ...
    @raise_not_implemented
    def get_grid_edge_nodes(
        self, grid: int, edge_nodes: np.ndarray
    ) -> typing.NoReturn: ...
    @raise_not_implemented
    def get_grid_face_edges(
        self, grid: int, face_edges: np.ndarray
    ) -> typing.NoReturn: ...
    @raise_not_implemented
    def get_grid_face_nodes(
        self, grid: int, face_nodes: np.ndarray
    ) -> typing.NoReturn: ...
    @raise_not_implemented
    def get_grid_nodes_per_face(
        self, grid: int, nodes_per_face: np.ndarray
    ) -> typing.NoReturn: ...
