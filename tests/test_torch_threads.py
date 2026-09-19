"""
Resolution and once-per-process application of torch's intra-op thread count.
"""

from __future__ import annotations

import logging

import pytest
import torch

from lstm import torch_threads
from lstm.bmi_lstm import bmi_LSTM
from lstm.torch_threads import configure_torch_threads, resolve_num_threads

from helpers import GOLDEN_CONFIG


@pytest.fixture
def fresh_process(monkeypatch):
    """Forget any earlier application and restore torch's thread count afterwards."""
    before = torch.get_num_threads()
    monkeypatch.setattr(torch_threads, "_applied", None)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize(
    "configured, environ, expected",
    [
        (None, {}, 1),
        (3, {"OMP_NUM_THREADS": "2"}, 3),
        ("3", {}, 3),
        (None, {"OMP_NUM_THREADS": "2"}, 2),
        (None, {"OMP_NUM_THREADS": " 2,4 "}, 2),
        (None, {"OMP_NUM_THREADS": ""}, 1),
        (None, {"OMP_NUM_THREADS": "zero"}, 1),
        (None, {"OMP_NUM_THREADS": "0"}, 1),
        (0, {"OMP_NUM_THREADS": "2"}, 2),
        (-1, {}, 1),
        (True, {}, 1),
        ("many", {}, 1),
    ],
    ids=[
        "default",
        "config-beats-env",
        "config-string",
        "env",
        "env-nested-list",
        "empty-env",
        "bad-env",
        "zero-env",
        "zero-config-falls-to-env",
        "negative-config",
        "bool-config",
        "bad-config",
    ],
)
def test_resolve_num_threads(configured, environ, expected):
    assert resolve_num_threads(configured, environ) == expected


def test_resolve_reads_process_env(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    assert resolve_num_threads() == 4
    monkeypatch.delenv("OMP_NUM_THREADS")
    assert resolve_num_threads() == 1


def test_configure_applies_once(fresh_process, caplog):
    with caplog.at_level(logging.INFO, logger="bmi.lstm"):
        assert configure_torch_threads(1) == 1
        assert configure_torch_threads(2) == 1
        assert configure_torch_threads(None) == 1
    assert torch.get_num_threads() == 1
    assert [r.message for r in caplog.records if r.levelno == logging.INFO] == [
        "torch intra-op threads: 1"
    ]
    assert sum(r.levelno == logging.WARNING for r in caplog.records) == 1


def test_initialize_defaults_to_one_thread(fresh_process, monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    torch.set_num_threads(2)
    bmi_LSTM().initialize(str(GOLDEN_CONFIG))
    assert torch.get_num_threads() == 1
