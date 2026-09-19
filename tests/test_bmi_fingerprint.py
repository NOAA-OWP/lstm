"""
Tests for the model fingerprint that guards ngen BMI Serialization Protocol
payloads on `lstm.bmi_lstm.bmi_LSTM`: the pure `compute_fingerprint` function
on stand-in members, and the `fingerprint` property of an initialized module.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from lstm import bmi_lstm

from helpers import initialized


def _fake_member(hidden_size: int, inputs: list[str], run_dir: str, epochs: int):
    """Stand-in with the attributes `member_fingerprint` reads; no torch model."""
    cfg = {"hidden_size": hidden_size, "run_dir": Path(run_dir), "epochs": epochs}
    model = types.SimpleNamespace(cfg=cfg, hidden_size=hidden_size, input_names=tuple(inputs))
    return types.SimpleNamespace(model=model)


def test_compute_fingerprint_is_readable_text():
    members = [
        _fake_member(8, ["a", "b", "elev_mean"], "./runs/demo_run", 3),
        _fake_member(4, ["a"], "/abs/other_run", 12),
    ]
    fp = bmi_lstm.compute_fingerprint(members)
    assert isinstance(fp, bytes)
    assert fp == (
        b"lstm-bmi;members=2;"
        b"0:hidden=8,inputs=a|b|elev_mean,run=demo_run,epoch=3;"
        b"1:hidden=4,inputs=a,run=other_run,epoch=12"
    )
    # decodes cleanly as UTF-8 and is readable
    assert fp.decode("utf-8").startswith(bmi_lstm.FINGERPRINT_PREFIX)


def test_compute_fingerprint_no_members():
    assert bmi_lstm.compute_fingerprint([]) == b"lstm-bmi;members=0"


@pytest.mark.parametrize(
    "change",
    [
        dict(hidden_size=9),
        dict(inputs=["b", "a"]),
        dict(inputs=["a"]),
        dict(run_dir="./runs/demo_run_2"),
        dict(epochs=4),
    ],
)
def test_compute_fingerprint_sensitive_to_each_field(change: dict):
    base = dict(hidden_size=8, inputs=["a", "b"], run_dir="./runs/demo_run", epochs=3)
    reference = bmi_lstm.compute_fingerprint([_fake_member(**base)])
    changed = bmi_lstm.compute_fingerprint([_fake_member(**{**base, **change})])
    assert reference != changed


def test_compute_fingerprint_ignores_run_dir_parent():
    """Only the run directory's name participates, not where it is checked out."""
    here = _fake_member(8, ["a"], "./trained/demo_run", 3)
    elsewhere = _fake_member(8, ["a"], "/somewhere/else/demo_run", 3)
    assert bmi_lstm.compute_fingerprint([here]) == bmi_lstm.compute_fingerprint(
        [elsewhere]
    )


def test_fingerprint_set_at_initialize(golden_config: Path):
    model = bmi_lstm.bmi_LSTM()
    with pytest.raises(RuntimeError, match="initialize"):
        model.fingerprint
    model.initialize(str(golden_config))
    assert isinstance(model.fingerprint, bytes)
    assert model.fingerprint == (
        b"lstm-bmi;members=1;"
        b"0:hidden=126,inputs=APCP_surface|TMP_2maboveground|elev_mean|slope_mean,"
        b"run=nh_AORC_hourly_slope_elev_precip_temp_seq999_seed101_2801_191806,"
        b"epoch=9"
    )
    assert model.fingerprint == bmi_lstm.compute_fingerprint(model.ensemble_members)


def test_fingerprint_identical_for_same_config(golden_config: Path):
    a = initialized(golden_config)
    b = initialized(golden_config)
    assert a.fingerprint == b.fingerprint


def test_fingerprint_differs_between_single_and_two_member_config(
    two_member_config: Path,
    golden_config: Path,
):
    single = initialized(golden_config)
    double = initialized(two_member_config)
    assert len(double.ensemble_members) == 2
    assert single.fingerprint != double.fingerprint
    assert double.fingerprint.startswith(b"lstm-bmi;members=2;")
    # both members are the bundled model, so their sections differ only by index
    sections = double.fingerprint.decode("utf-8").split(";")[2:]
    assert [s.split(":", 1)[0] for s in sections] == ["0", "1"]
    assert sections[0].split(":", 1)[1] == sections[1].split(":", 1)[1]


def test_fingerprint_stable_across_updates(golden_config: Path):
    model = initialized(golden_config)
    before = bytes(model.fingerprint)
    for name in model.get_input_var_names():
        model.set_value(name, np.array([1.0], dtype="float64"))
    for _ in range(3):
        model.update()
    assert model.get_current_time() == 3 * model.get_time_step()
    assert model.fingerprint == before
