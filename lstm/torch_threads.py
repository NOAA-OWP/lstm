"""
Bound torch's intra-op thread pool once per process.

NextGen runs many MPI ranks per node, each with one core's worth of work, and
each rank's embedded interpreter loads torch. By default torch may size its
thread pool to the whole node, so every rank oversubscribes the node and the
ranks contend for cores. The LSTM runs batch-size-1, single-step inference,
whose matrix products are too small for more threads to pay off even on an
idle machine, so the module uses one thread unless its config or
``OMP_NUM_THREADS`` says otherwise. CPU affinity is not consulted: ranks are
not necessarily pinned, so their affinity may span the whole node.

The inter-op pool is left alone: plain LSTM inference does not use it, and
``torch.set_num_interop_threads`` raises once any parallel work has run.
"""

from __future__ import annotations

import os
import typing

import torch

from .logger import logger

CONFIG_KEY = "torch_num_threads"
"""optional BMI config key giving an explicit intra-op thread count"""

DEFAULT_NUM_THREADS = 1

_applied: int | None = None
"""the thread count this process set, or `None` before the first call"""


def _positive_int(value: typing.Any) -> int | None:
    """Return `value` as an int if it is a positive integer, else `None`."""
    if isinstance(value, bool):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def resolve_num_threads(
    configured: typing.Any = None, environ: typing.Mapping[str, str] = os.environ
) -> int:
    """
    Return the intra-op thread count to use.

    In order of precedence: `configured` (the BMI config value) if it is a
    positive integer; the first entry of ``OMP_NUM_THREADS`` (which may be a
    comma-separated list of per-nesting-level counts) if it is one; otherwise
    `DEFAULT_NUM_THREADS`. Invalid values fall through to the next source and
    are logged.
    """
    n = _positive_int(configured)
    if n is not None:
        return n
    if configured is not None:
        logger.warning("Ignoring invalid %s=%r; expected a positive integer", CONFIG_KEY, configured)

    omp = environ.get("OMP_NUM_THREADS")
    if omp is not None:
        n = _positive_int(omp.split(",")[0].strip())
        if n is not None:
            return n
        logger.warning("Ignoring invalid OMP_NUM_THREADS=%r; expected a positive integer", omp)

    return DEFAULT_NUM_THREADS


def configure_torch_threads(configured: typing.Any = None) -> int:
    """
    Set torch's intra-op thread count, once per process, and return it.

    The first call resolves the count (see `resolve_num_threads`), applies it,
    and logs the result. Later calls, one per BMI instance, change nothing; a
    later call asking for a different explicit count is logged as ignored so
    every instance in a process shares one setting.
    """
    global _applied
    if _applied is None:
        torch.set_num_threads(resolve_num_threads(configured))
        _applied = torch.get_num_threads()
        logger.info("torch intra-op threads: %d", _applied)
    elif _positive_int(configured) not in (None, _applied):
        logger.warning(
            "Ignoring %s=%r; this process already uses %d torch threads",
            CONFIG_KEY, configured, _applied,
        )
    return _applied
