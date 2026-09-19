"""
Shared constants and helpers for the ngen BMI Serialization Protocol tests.

Everything here drives `lstm.bmi_lstm.bmi_LSTM` through its public BMI surface
the way ngen does: the save sequence (create, read size and state, free), the
restore sequence (announce the byte count, deliver the bytes), stepping with
forcing, and reading member arrays back through the member accessor. Each
helper is defined once here so that no test file carries its own copy.

Fixtures built on these (repository-root working directory, config paths, the
24 hours of bundled forcing) live in `conftest.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from lstm import bmi_lstm
from lstm import serialization_protocol as protocol

REPO_ROOT = Path(__file__).parent.parent
BASIN_ID = "02064000"
GOLDEN_CONFIG = REPO_ROOT / f"configs/{BASIN_ID}_nh_NLDAS_hourly.yml"
FORCING_FILE = REPO_ROOT / "data/usgs-streamflow-nldas_hourly.nc"
BUNDLED_MODEL_CONFIG = (
    "./trained_neuralhydrology_models/"
    "nh_AORC_hourly_slope_elev_precip_temp_seq999_seed101_2801_191806/config.yml"
)

# identical to the mapping used by tests/integration_test.py
FORCING_VARIABLE_NAME_MAPPING = {
    "total_precipitation": "atmosphere_water__liquid_equivalent_precipitation_rate",
    "temperature": "land_surface_air__temperature",
    "longwave_radiation": "land_surface_radiation~incoming~longwave__energy_flux",
    "shortwave_radiation": "land_surface_radiation~incoming~shortwave__energy_flux",
    "pressure": "land_surface_air__pressure",
    "specific_humidity": "atmosphere_air_water~vapor__relative_saturation",
    "wind_u": "land_surface_wind__x_component_of_velocity",
    "wind_v": "land_surface_wind__y_component_of_velocity",
}

TOTAL_STEPS = 24
"""hours of bundled forcing the end-to-end tests drive"""
SPLIT_STEP = 12
"""the step at which the end-to-end tests capture and restore"""

TRIGGER = np.array([1], dtype="int32")
"""a trigger value; the protocol says the value is ignored"""

Forcing = list[dict[str, "float | np.ndarray"]]
"""one dict per step keyed by BMI input name"""
MemberArrays = list[tuple[np.ndarray, np.ndarray]]
"""flat float32 (hidden, cell) copies, one pair per ensemble member"""


# ---------------  building and stepping  -----------------------------


def initialized(config: Path) -> bmi_lstm.bmi_LSTM:
    model = bmi_lstm.bmi_LSTM()
    model.initialize(str(config))
    return model


def synthetic_forcing(steps: int, seed: int = 0) -> Forcing:
    """Deterministic per-step input values for every dynamic input name."""
    rng = np.random.default_rng(seed)
    names = [name for name, _ in bmi_lstm._dynamic_input_vars]
    return [{name: float(rng.uniform(0.0, 10.0)) for name in names} for _ in range(steps)]


def step(model: bmi_lstm.bmi_LSTM, inputs: Mapping[str, float | np.ndarray]) -> dict[str, float]:
    """Set every given input, update once, and return the outputs by name."""
    for name, value in inputs.items():
        model.set_value(name, np.array([value], dtype="float64"))
    model.update()
    return outputs(model)


def run(model: bmi_lstm.bmi_LSTM, forcing: Forcing) -> dict[str, list[float]]:
    """Run `model` over `forcing`, returning per-output lists of recorded values."""
    recorded: dict[str, list[float]] = {name: [] for name in model.get_output_var_names()}
    for inputs in forcing:
        for name, value in step(model, inputs).items():
            recorded[name].append(value)
    return recorded


# ---------------  the ngen save sequence  -----------------------------


def create(model: bmi_lstm.bmi_LSTM) -> None:
    model.set_value(protocol.SERIALIZATION_CREATE, TRIGGER)


def free(model: bmi_lstm.bmi_LSTM) -> None:
    model.set_value(protocol.SERIALIZATION_FREE, TRIGGER)


def size(model: bmi_lstm.bmi_LSTM) -> int:
    return int(model.get_value_ptr(protocol.SERIALIZATION_SIZE)[0])


def state_bytes(model: bmi_lstm.bmi_LSTM) -> bytes:
    return model.get_value_ptr(protocol.SERIALIZATION_STATE).tobytes()


def capture_and_free(model: bmi_lstm.bmi_LSTM) -> tuple[int, bytes]:
    """The ngen save sequence: create, read size and state, free."""
    create(model)
    announced = size(model)
    payload = state_bytes(model)
    free(model)
    return announced, payload


def captured_after(
    config: Path, steps: int, seed: int = 0
) -> tuple[bmi_lstm.bmi_LSTM, bytes]:
    """Run a module for `steps` of synthetic forcing; return it with the payload captured then."""
    source = initialized(config)
    for inputs in synthetic_forcing(steps, seed=seed):
        step(source, inputs)
    _, payload = capture_and_free(source)
    return source, payload


# ---------------  the ngen restore sequence  -----------------------------


def announce(model: bmi_lstm.bmi_LSTM, count: int) -> None:
    model.set_value(protocol.SERIALIZATION_SIZE, np.array([count], dtype="int64"))


def deliver(model: bmi_lstm.bmi_LSTM, payload: bytes) -> None:
    """Deliver the payload the way ngen's Python adapter does: as a uint8 array."""
    model.set_value(protocol.SERIALIZATION_STATE, np.frombuffer(payload, dtype="uint8"))


def restore(model: bmi_lstm.bmi_LSTM, payload: bytes) -> None:
    """The protocol's ordered restore sequence: announce the size, then deliver."""
    announce(model, len(payload))
    deliver(model, payload)


# ---------------  reading state back  -----------------------------


def outputs(model: bmi_lstm.bmi_LSTM) -> dict[str, float]:
    return {name: float(model.get_value_ptr(name)[0]) for name in model.get_output_var_names()}


def member_arrays(model: bmi_lstm.bmi_LSTM) -> MemberArrays:
    """Flat float32 copies of every member's (hidden, cell) state via the member accessor."""
    return [m.state_arrays() for m in model.ensemble_members]


def observed(model: bmi_lstm.bmi_LSTM) -> dict:
    """Everything a rejected restore must leave alone, copied for later comparison."""
    return {
        "members": member_arrays(model),
        "outputs": outputs(model),
        "time": model.get_current_time(),
    }


# ---------------  equality assertions  -----------------------------


def assert_member_arrays_equal(actual: MemberArrays, expected: MemberArrays) -> None:
    """Every (hidden, cell) pair matches bitwise as float32."""
    assert len(actual) == len(expected)
    for (h_a, c_a), (h_e, c_e) in zip(actual, expected):
        assert h_a.dtype == h_e.dtype == np.float32
        assert h_a.shape == h_e.shape
        assert np.array_equal(h_a, h_e)
        assert np.array_equal(c_a, c_e)


def assert_members_equal(actual: bmi_lstm.bmi_LSTM, expected: bmi_lstm.bmi_LSTM) -> None:
    """Every member's hidden and cell state matches bitwise as float32."""
    assert_member_arrays_equal(member_arrays(actual), member_arrays(expected))


def assert_size_equals_buffer_length(model: bmi_lstm.bmi_LSTM) -> None:
    """The protocol invariant: the size variable always equals the state buffer's length."""
    state_ptr = model.get_value_ptr(protocol.SERIALIZATION_STATE)
    assert size(model) == len(state_ptr)
    assert model.get_var_nbytes(protocol.SERIALIZATION_STATE) == len(state_ptr)


def assert_untouched(model: bmi_lstm.bmi_LSTM, before: dict) -> None:
    """The model's members, outputs, and clock equal `before`, and the invariant holds."""
    after = observed(model)
    assert_member_arrays_equal(after["members"], before["members"])
    assert after["outputs"] == before["outputs"]
    assert after["time"] == before["time"]
    # the protocol invariant survives a rejected delivery: size == buffer length
    assert_size_equals_buffer_length(model)
