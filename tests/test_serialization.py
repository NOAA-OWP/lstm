from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np
import pytest

from lstm import bmi_lstm, serialization

REPO_ROOT = Path(__file__).parent.parent
TEST_DIR = Path(__file__).parent
BMI_CFG_FILE = REPO_ROOT / "bmi_config_files/02064000_nh_NLDAS_hourly.yml"

CREATE = serialization.SERIALIZATION_CREATE
FREE = serialization.SERIALIZATION_FREE
SIZE = serialization.SERIALIZATION_SIZE
STATE = serialization.SERIALIZATION_STATE
RESERVED = (CREATE, FREE, SIZE, STATE)

TRIGGER = np.zeros(1, dtype="int32")

# arbitrary, but physically plausible, forcing
FORCING = {
    "atmosphere_water__liquid_equivalent_precipitation_rate": 2.5,
    "land_surface_air__temperature": 285.0,
    "land_surface_radiation~incoming~longwave__energy_flux": 300.0,
    "land_surface_radiation~incoming~shortwave__energy_flux": 150.0,
    "land_surface_air__pressure": 100000.0,
    "atmosphere_air_water~vapor__relative_saturation": 0.008,
    "land_surface_wind__x_component_of_velocity": 1.0,
    "land_surface_wind__y_component_of_velocity": -1.0,
}


@contextlib.contextmanager
def pushd(target: Path):
    saved = os.getcwd()
    os.chdir(target)
    try:
        yield saved
    finally:
        os.chdir(saved)


def new_model() -> bmi_lstm.bmi_LSTM:
    model = bmi_lstm.bmi_LSTM()
    with pushd(TEST_DIR):
        model.initialize(str(BMI_CFG_FILE))
    return model


def run(model: bmi_lstm.bmi_LSTM, n_steps: int, start: int = 0) -> np.ndarray:
    """Run `n_steps` with time-varying forcing; return runoff depth per step."""
    out = np.zeros(n_steps)
    for i in range(n_steps):
        scale = 1.0 + 0.1 * np.sin(start + i)
        for name, value in FORCING.items():
            model.set_value(name, np.array([value * scale]))
        model.update()
        model.get_value("land_surface_water__runoff_depth", out[i : i + 1])
    return out


def capture(model: bmi_lstm.bmi_LSTM) -> np.ndarray:
    """Drive the protocol's save sequence like ngen does."""
    model.set_value(CREATE, TRIGGER)
    size = np.zeros(1, dtype=model.get_var_type(SIZE))
    model.get_value(SIZE, size)
    dest = np.zeros(int(size[0]), dtype=model.get_var_type(STATE))
    model.get_value(STATE, dest)
    model.set_value(FREE, TRIGGER)
    return dest


@pytest.fixture(scope="module")
def model() -> bmi_lstm.bmi_LSTM:
    return new_model()


def test_reserved_names_not_enumerated(model: bmi_lstm.bmi_LSTM):
    names = set(model.get_input_var_names()) | set(model.get_output_var_names())
    assert names.isdisjoint(RESERVED)


def test_reserved_units(model: bmi_lstm.bmi_LSTM):
    assert model.get_var_units(CREATE) == "ngen::trigger"
    assert model.get_var_units(FREE) == "ngen::trigger"
    assert model.get_var_units(SIZE) == "bytes"
    assert model.get_var_units(STATE) == "ngen::opaque"


def test_reserved_introspection(model: bmi_lstm.bmi_LSTM):
    for name in (CREATE, FREE, SIZE):
        assert model.get_var_type(name) == "int32"
        assert model.get_var_itemsize(name) == 4
        assert model.get_var_nbytes(name) == 4
    assert model.get_var_type(STATE) == "uint8"
    assert model.get_var_itemsize(STATE) == 1


def test_reserved_names_have_no_spatial_metadata(model: bmi_lstm.bmi_LSTM):
    for name in RESERVED:
        with pytest.raises(KeyError):
            model.get_var_grid(name)
        with pytest.raises(KeyError):
            model.get_var_location(name)


def test_nbytes_valid_before_create():
    model = new_model()
    size = np.zeros(1, dtype="int32")
    model.get_value(SIZE, size)
    assert size[0] == 0, "size should be 0 before create"

    nbytes = model.get_var_nbytes(STATE)
    assert nbytes > 0
    assert nbytes == capture(model).nbytes


def test_create_and_free_do_not_alter_state():
    reference = new_model()
    expected = run(reference, 10)

    model = new_model()
    actual = np.concatenate([run(model, 5), (capture(model), run(model, 5, start=5))[1]])
    np.testing.assert_array_equal(actual, expected)


def test_free_is_safe_without_create():
    model = new_model()
    model.set_value(FREE, TRIGGER)
    model.set_value(FREE, TRIGGER)
    assert model.get_value_ptr(STATE).nbytes == 0


def test_round_trip():
    model = new_model()
    run(model, 7)
    payload = capture(model)
    expected = run(model, 10, start=7)

    restored = new_model()
    # mirror an adapter that sizes the restore buffer by `get_var_nbytes`
    assert restored.get_var_nbytes(STATE) == payload.nbytes
    restored.set_value(STATE, payload.copy())
    assert restored.get_current_time() == 7 * restored.get_time_step()
    np.testing.assert_array_equal(run(restored, 10, start=7), expected)


def test_restore_does_not_touch_serialization_buffer():
    model = new_model()
    run(model, 3)
    payload = capture(model)

    restored = new_model()
    restored.set_value(STATE, payload)
    size = np.zeros(1, dtype="int32")
    restored.get_value(SIZE, size)
    assert size[0] == 0
    assert restored.get_value_ptr(STATE).nbytes == 0


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda b: b"XXXX" + b[4:], id="bad-magic"),
        pytest.param(lambda b: b[:4] + b"\xff\xff" + b[6:], id="bad-version"),
        pytest.param(lambda b: b[:-1], id="truncated"),
        pytest.param(lambda b: b + b"\x00", id="trailing"),
    ],
)
def test_malformed_payload_rejected_without_mutation(corrupt):
    model = new_model()
    run(model, 4)
    payload = corrupt(capture(model).tobytes())

    restored = new_model()
    run(restored, 2)
    before = capture(restored)
    with pytest.raises(serialization.PayloadError):
        restored.set_value(STATE, np.frombuffer(payload, dtype="uint8"))
    np.testing.assert_array_equal(capture(restored), before)


def test_mismatched_hidden_size_rejected():
    model = new_model()
    payload = capture(model)
    snapshot = serialization.decode(payload.tobytes())
    h_t, c_t = snapshot.members[0]
    snapshot.members[0] = (h_t[:-1], c_t[:-1])
    with pytest.raises(serialization.PayloadError, match="hidden state size"):
        serialization.apply(
            snapshot, (model._dynamic_inputs, model._outputs), model.ensemble_members
        )
