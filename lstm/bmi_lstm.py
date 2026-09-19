#
# Copyright (C) 2025 Austin Raney, Lynker
#
# Author:
#   Austin Raney <araney@lynker.com>
#   Jonathan Frame <jmframe@ua.edu>
#
#  In hopes to avoid confusion, there are five main abstractions of note:
#
#  bmi_LSTM:
#       Class that implements the BMI interface and interoperates with the
#       NextGen Framework. Its roles are to (1) manage variables passed to and
#       from the framework, (2) orchestrate running one or more LSTM models
#       (each contained in an `EnsembleMember`), and (3) ensemble output from
#       aforementioned `EnsembleMember`s model for consumption by NextGen
#       Framework.
#
# State:
#       Represents a collection of variables (`Var`s), and provides methods to
#       access and mutate individual `Vars`. Not to be confused with an LSTM
#       model's 'hidden states' or 'cell states', or the dynamic modeling
#       typical taxonomy of 'state-space'. This is simply a container type for
#       passing data to and from the framework. An `EnsembleMember`, which
#       contains an LSTM model, queries (see `EnsembleMember.update()`) a
#       `State` container to receive its input (e.g. precipitation).
#
# Var:
#       Representation of a model variable with a name, unit, and value
#       (`np.NDArray`) that may or may not be exposed via BMI (e.g. static LSTM
#       attributes). `Var`s exposed over BMI are queried and mutated inplace
#       using their `value` property (`np.NDArray`).
#
# EnsembleMember:
#       Class that encapsulates a _single_ LSTM model and all necessary
#       subcomponents. Its roles are to (1) manage an LSTM model's internal
#       states (tensors) and (2) run an LSTM model by querying a `State`
#       container (see: return its output. `EnsembleMember.update()`) for
#       input.
#
# Nextgen_CudaLSTM:
#       Represents a PyTorch-based LSTM model used to make predictions. This
#       class is responsible for (1) defining the LSTM architecture, (2)
#       managing hidden and cell states, and (3) performing forward inference
#       (i.e. takes input data, processes it through the LSTM, and returns
#       predicted outputs along with updated hidden and cell states) based on
#       the information collected in the BMI. An `Nextgen_CudaLSTM` contains a
#       single instance of an LSTM model that performs stepwise predictions
#       without handling model orchestration or input-output management.
#
from __future__ import annotations

import collections
import functools
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import yaml
try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

from . import nextgen_cuda_lstm
from . import serialization_codec
from .serialization_protocol import SerializationProtocol
# The reserved protocol names are defined by `serialization_protocol`. They are
# imported here (not re-defined) so callers written against this module before
# the protocol had its own module can still reach them through it.
from .serialization_protocol import (  # noqa: F401
    SERIALIZATION_CREATE,
    SERIALIZATION_FREE,
    SERIALIZATION_SIZE,
    SERIALIZATION_STATE,
)
from .base import BmiBase
from .logger import configure_logging, logger
from .model_state import State, StateFacade, Var
from .torch_threads import CONFIG_KEY as TORCH_THREADS_KEY, configure_torch_threads

# --------------   Dynamic Attributes -----------------------------
_dynamic_input_vars = [
    ("land_surface_radiation~incoming~longwave__energy_flux", "W m-2"),
    ("land_surface_air__pressure", "Pa"),
    ("atmosphere_air_water~vapor__relative_saturation", "kg kg-1"),
    ("atmosphere_water__liquid_equivalent_precipitation_rate", "mm h-1"),
    ("land_surface_radiation~incoming~shortwave__energy_flux", "W m-2"),
    ("land_surface_air__temperature", "degK"),
    ("land_surface_wind__x_component_of_velocity", "m s-1"),
    ("land_surface_wind__y_component_of_velocity", "m s-1"),
]

# --------------    Name Mappings    -----------------------------
DYNAMIC_INPUT_NAME_CROSSWALK = {
    "DLWRF_surface": "land_surface_radiation~incoming~longwave__energy_flux",
    "PRES_surface": "land_surface_air__pressure",
    "SPFH_2maboveground": "atmosphere_air_water~vapor__relative_saturation",
    "APCP_surface": "atmosphere_water__liquid_equivalent_precipitation_rate",
    "DSWRF_surface": "land_surface_radiation~incoming~shortwave__energy_flux",
    "TMP_2maboveground": "land_surface_air__temperature",
    "UGRD_10maboveground": "land_surface_wind__x_component_of_velocity",
    "VGRD_10maboveground": "land_surface_wind__y_component_of_velocity",
}

# --------------   Static Attributes -----------------------------

_output_vars = [
    ("land_surface_water__runoff_volume_flux", "m3 s-1"),
    ("land_surface_water__runoff_depth", "m"),
]


def crosswalk_to_external(name: str):
    """Return the external name (the name exposed via BMI) for a given internal name."""
    return DYNAMIC_INPUT_NAME_CROSSWALK.get(name, name)

# ---------------  Trained Model -----------------------------


def bmi_array(arr: list[float]) -> npt.NDArray:
    """Trivial wrapper function to ensure the expected numpy array datatype is used."""
    return np.array(arr, dtype="float64")


class Valuer(typing.Protocol):
    """Thin interface with the same signature as `State.value`."""

    def value(self, name: str) -> npt.NDArray: ...


@dataclass
class TrainingScalars:
    input_mean: npt.NDArray
    input_std: npt.NDArray
    output_mean: npt.NDArray
    output_std: npt.NDArray


@dataclass(frozen=True)
class TrainedModel:
    """
    A trained neuralhydrology model as loaded for inference: its training
    config, feature scalars, frozen weights, and the order of its inputs.

    Instances are shared. `load_trained_model` returns the same object for
    every call naming the same config file, so a process running many
    catchments holds one copy of each model's weights. Sharing is safe because
    the module holds parameters only (each `EnsembleMember` owns its hidden
    and cell state), inference runs under `torch.no_grad()`, and the
    parameters are frozen after loading. Treat `cfg` as read-only.
    """

    cfg: dict[str, typing.Any]
    scalars: TrainingScalars
    lstm: nextgen_cuda_lstm.Nextgen_CudaLSTM
    input_names: tuple[str, ...]
    """order of the model input tensor: dynamic inputs then static attributes"""
    hidden_size: int
    """element count of each of the hidden and cell state tensors"""


@functools.lru_cache(maxsize=None)
def load_trained_model(config_file: Path) -> TrainedModel:
    """
    Load a `TrainedModel` from its training config, once per distinct path.

    Every dynamic input the model was trained on must have a BMI name in
    `DYNAMIC_INPUT_NAME_CROSSWALK`; otherwise `ValueError` is raised.
    """
    cfg = yaml.load(config_file.read_text(), Loader=SafeLoader)
    coerce_config(cfg)

    unmapped = [n for n in cfg["dynamic_inputs"] if n not in DYNAMIC_INPUT_NAME_CROSSWALK]
    if unmapped:
        raise ValueError(
            f"Dynamic inputs {unmapped} in {config_file} have no BMI name.\n"
            f"Known dynamic inputs: {sorted(DYNAMIC_INPUT_NAME_CROSSWALK)}\n"
        )

    scaler_file = cfg["run_dir"] / "train_data/train_data_scaler.yml"
    with scaler_file.open("r") as fp:
        train_data_scaler = yaml.load(fp, Loader=SafeLoader)

    return TrainedModel(
        cfg=cfg,
        scalars=load_training_scalars(cfg, train_data_scaler),
        lstm=initialize_lstm(cfg),
        input_names=tuple(cfg["dynamic_inputs"]) + tuple(cfg["static_attributes"]),
        hidden_size=int(cfg["hidden_size"]),
    )


# ---------------  Ensemble Member -----------------------------


class EnsembleMember:
    """
    One catchment's instance of a `TrainedModel`: the hidden and cell state
    that evolve with each timestep, and the output scaling for the catchment
    area. The model itself is shared with every member built from the same
    training config.
    """

    def __init__(self, model: TrainedModel, output_scaling_factor_cms: float):
        self.model = model
        self.output_scaling_factor_cms = output_scaling_factor_cms
        # WARNING: This implementation of the LSTM can only handle a batch size of 1
        # NOTE: aaraney: assume initial state is always zero (ask jframe about this. no other option now)
        self.h_t = torch.zeros(1, 1, model.hidden_size).float()
        self.c_t = torch.zeros(1, 1, model.hidden_size).float()

    def update(self, state: Valuer) -> typing.Iterable[Var]:
        """
        Run a single model timestep and return the model inference values.

        `state` contains the input variable names, units, and values for the
        current iteration.
        """
        model = self.model
        with torch.no_grad():
            inputs = gather_inputs(state, model.input_names)
            scaled = scale_inputs(
                inputs, model.scalars.input_mean, model.scalars.input_std
            )
            input_tensor = torch.tensor(scaled)
            lstm_output, self.h_t, self.c_t = model.lstm.forward(
                input_tensor, self.h_t, self.c_t
            )
            # TODO: aaraney, there is gap here between mapping 'internal'
            # output names to 'external' output names. Right now this is
            # hard-coded and handled in `scale_outputs`. Introduce semantics
            # for more generally handling outputs.
            yield from scale_outputs(
                model.cfg,
                lstm_output,
                model.scalars.output_mean,
                model.scalars.output_std,
                self.output_scaling_factor_cms,
            )

    def state_arrays(
        self,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Return the hidden and cell state as a pair of flat float32 numpy copies.

        The tensors are shaped (1, batch size 1, hidden size); each copy is
        flattened to `hidden_size` elements and does not alias the tensor.
        """
        return (
            np.array(self.h_t.detach().cpu().numpy(), dtype="float32").ravel(),
            np.array(self.c_t.detach().cpu().numpy(), dtype="float32").ravel(),
        )

    def set_state_arrays(
        self, hidden: npt.ArrayLike, cell: npt.ArrayLike
    ) -> None:
        """
        Replace the hidden and cell state from a pair of flat arrays.

        Each array must hold exactly `hidden_size` elements; they are converted
        to float32 tensors of the member's state shape and assigned. A wrong
        element count raises `ValueError` before either tensor is touched.
        """
        hidden_arr = np.asarray(hidden, dtype="float32").ravel()
        cell_arr = np.asarray(cell, dtype="float32").ravel()
        for label, arr in (("hidden", hidden_arr), ("cell", cell_arr)):
            if arr.size != self.model.hidden_size:
                raise ValueError(
                    f"{label} state has {arr.size} elements but this member "
                    f"has hidden size {self.model.hidden_size}"
                )
        shape = tuple(self.h_t.shape)
        self.h_t = torch.tensor(hidden_arr, dtype=torch.float32).reshape(shape)
        self.c_t = torch.tensor(cell_arr, dtype=torch.float32).reshape(shape)


def load_training_scalars(
    cfg: dict[str, typing.Any], train_data_scalar: dict[str, typing.Any]
) -> TrainingScalars:
    out_mean = train_data_scalar["xarray_feature_center"]["data_vars"][
        cfg["target_variables"][0]
    ]["data"]
    out_std = train_data_scalar["xarray_feature_scale"]["data_vars"][
        cfg["target_variables"][0]
    ]["data"]

    input_mean = bmi_array(
        [
            train_data_scalar["xarray_feature_center"]["data_vars"][x]["data"]
            for x in cfg["dynamic_inputs"]
        ]
        + [train_data_scalar["attribute_means"][x] for x in cfg["static_attributes"]]
    )

    input_std = bmi_array(
        [
            train_data_scalar["xarray_feature_scale"]["data_vars"][x]["data"]
            for x in cfg["dynamic_inputs"]
        ]
        + [train_data_scalar["attribute_stds"][x] for x in cfg["static_attributes"]]
    )
    return TrainingScalars(
        input_mean=input_mean,
        input_std=input_std,
        output_mean=out_mean,
        output_std=out_std,
    )


def initialize_lstm(cfg: dict[str, typing.Any]) -> nextgen_cuda_lstm.Nextgen_CudaLSTM:
    """Build the inference-only LSTM for a training config and load its trained weights."""
    input_size = len(cfg["dynamic_inputs"]) + len(cfg["static_attributes"])
    hidden_layer_size = cfg["hidden_size"]
    output_size = len(cfg["target_variables"])
    lstm = nextgen_cuda_lstm.Nextgen_CudaLSTM(
        input_size=input_size,
        hidden_layer_size=hidden_layer_size,
        output_size=output_size,
        batch_size=1,
        seq_length=1,
    )
    # ------------ Load in the trained weights ----------------------------#
    # Save the default model weights. We need to make sure we have the same keys.
    default_state_dict = lstm.state_dict()

    trained_model_file = cfg["run_dir"] / "model_epoch{}.pt".format(
        str(cfg["epochs"]).zfill(3)
    )
    trained_state_dict = torch.load(
        trained_model_file, map_location=torch.device("cpu")
    )

    # Changing the name of the head weights, since different in NH
    trained_state_dict["head.weight"] = trained_state_dict.pop("head.net.0.weight")
    trained_state_dict["head.bias"] = trained_state_dict.pop("head.net.0.bias")
    trained_state_dict = {x: trained_state_dict[x] for x in default_state_dict.keys()}

    # Load in the trained weights.
    lstm.load_state_dict(trained_state_dict)
    lstm.eval()
    lstm.requires_grad_(False)
    return lstm


def gather_inputs(
    state: Valuer, train_input_names: typing.Iterable[str]
) -> npt.NDArray:
    logger.debug("Collecting LSTM inputs ...")

    input_list = []
    for name in train_input_names:
        bmi_name = crosswalk_to_external(name)
        value = state.value(bmi_name)
        assert value.size == 1, "`value` should a single scalar in a 1d array"
        input_list.append(value[0])
        logger.debug("  var_name=%s", bmi_name)
        logger.debug("  type(value)=%s", type(value))
        logger.debug("  value=%s", value)

    collected = bmi_array(input_list)
    logger.debug("Collected inputs: %s",collected)
    return collected


def scale_inputs(
    input: npt.NDArray, mean: npt.NDArray, std: npt.NDArray
) -> npt.NDArray:
    logger.debug("Normalizing the tensor...")
    logger.debug("  input_mean =", mean)
    logger.debug("  input_std  =", std)

    # Center and scale the input values for use in torch
    input_array_scaled = (input - mean) / std
    logger.debug("### input_array =%s", input)
    logger.debug("### dtype(input_array) =%s", input.dtype)
    logger.debug("### type(input_array_scaled) =%s", type(input_array_scaled))
    logger.debug("### dtype(input_array_scaled) =%s", input_array_scaled.dtype)
    return input_array_scaled


def scale_outputs(
    cfg: dict[str, typing.Any],
    output: torch.tensor,
    output_mean: npt.NDArray,
    output_std: npt.NDArray,
    output_scale_factor_cms: float,
):
    logger.debug("model output: %s", output[0, 0, 0].numpy().tolist())

    if cfg["target_variables"][0] in ["qobs_mm_per_hour", "QObs(mm/hr)", "QObs(mm/h)"]:
        surface_runoff_mm = output[0, 0, 0].numpy() * output_std + output_mean
    elif cfg["target_variables"][0] in ["QObs(mm/d)"]:
        # daily to hourly
        surface_runoff_mm = (output[0, 0, 0].numpy() * output_std + output_mean) * (
            1 / 24
        )
    else:
        raise RuntimeError("unreachable")

    # clamp
    surface_runoff_mm = max(surface_runoff_mm, 0.0)
    # mm -> m
    surface_runoff_m = surface_runoff_mm / 1000.0

    # TODO: aaraney, this is kind of gross. think of a better way to do this.
    # The output is area normalized, this is needed to un-normalize it
    # mm->m                             km2 -> m2          hour->s
    # (1/1000) * (self.cfg_bmi['area_sqkm'] * 1000*1000) * (1/3600)
    surface_runoff_volume_m3_s = surface_runoff_mm * output_scale_factor_cms

    # TODO: aaraney: consider making this into a class or closure to avoid so
    # many small allocations.
    yield from (
        Var(
            name="land_surface_water__runoff_depth",
            unit="m",
            value=bmi_array([surface_runoff_m]),
        ),
        Var(
            name="land_surface_water__runoff_volume_flux",
            unit="m3 s-1",
            value=bmi_array([surface_runoff_volume_m3_s]),
        ),
    )


# ---------------  Model Fingerprint  -----------------------------

FINGERPRINT_PREFIX: typing.Final[str] = "lstm-bmi"
"""Leading token of every fingerprint, so the text is recognizable in a hex dump."""


def member_fingerprint(index: int, member: EnsembleMember) -> str:
    """
    Return the deterministic textual identity of a single ensemble member.

    The text records, in order, the member's hidden size, its ordered input
    names (dynamic inputs followed by static attributes, exactly as fed to the
    model), the name of the trained-model run directory, and the epoch number
    of the loaded weights. It is meant to be readable in a hex dump and is
    compared as bytes, never parsed.
    """
    cfg = member.model.cfg
    run_dir = Path(cfg["run_dir"]).name
    return (
        f"{index}:hidden={member.model.hidden_size},"
        f"inputs={'|'.join(member.model.input_names)},"
        f"run={run_dir},"
        f"epoch={int(cfg['epochs'])}"
    )


def compute_fingerprint(members: typing.Sequence[EnsembleMember]) -> bytes:
    """
    Derive the UTF-8 fingerprint identifying an ensemble of members.

    The fingerprint is ``lstm-bmi;members=<count>`` followed by one
    :func:`member_fingerprint` section per member, in ensemble order, all
    joined by ``;``. Two modules initialized from equivalent configurations
    produce identical bytes; a change in member count, order, hidden size,
    input names, trained-model run directory, or epoch changes the bytes.
    """
    sections = [FINGERPRINT_PREFIX, f"members={len(members)}"]
    sections.extend(
        member_fingerprint(index, member) for index, member in enumerate(members)
    )
    return ";".join(sections).encode("utf-8")


# ---------------  LSTM BMI Wrapper  -----------------------------


def build_state(vars: typing.Iterable[tuple[str, str]]) -> State:
    """
    Create a `State` object from a collection of (name: str, unit: str) tuples.
    Each `Var`'s `value` array is initialized to an `np.array([0.0], dtype="float64"))`.
    """
    g = (Var(name=name, unit=unit, value=bmi_array([0.0])) for (name, unit) in vars)
    return State(vars=g)


def resolve_static_attributes(
    cfg_bmi: dict[str, typing.Any], models: typing.Iterable[TrainedModel]
) -> dict[str, typing.Any]:
    """
    Return ``{name: value}`` for every static attribute the models require.

    The names come from the trained model configs, in model order without
    duplicates. The values come from the config's ``static_attributes``
    mapping when it has one and from the top-level keys otherwise, so the
    nested and the flat (generated, often JSON) config shapes are read the
    same way. A missing value raises ``ValueError`` naming every absent
    attribute.
    """
    source = cfg_bmi.get("static_attributes", cfg_bmi)
    if not isinstance(source, dict):
        raise ValueError(
            "'static_attributes' must be a mapping of attribute name to value, "
            f"got {type(source).__name__}"
        )

    names = list(dict.fromkeys(n for m in models for n in m.cfg["static_attributes"]))
    missing = [n for n in names if n not in source]
    if missing:
        raise ValueError(
            f"Missing static attributes: {sorted(missing)}.\n"
            f"Expected by the lstm: {names}\n"
            "Provide them under a 'static_attributes' mapping or as top-level keys.\n"
        )
    return {n: source[n] for n in names}


class bmi_LSTM(BmiBase):
    _timestep_size_s: typing.Final[int] = 3600
    """model timestep size in seconds"""

    def __init__(self) -> None:
        # _bmi_ variable state; this is separate from lstm ensemble member state.
        self._dynamic_inputs = build_state(_dynamic_input_vars)
        self._outputs = build_state(_output_vars)
        # the four reserved ngen serialization protocol variables (see
        # `serialization_protocol`), wired to pack and unpack this module's
        # `snapshot()`. They resolve by name only and are deliberately kept out
        # of the input / output name lists.
        self._serialization = SerializationProtocol(
            capture=lambda: serialization_codec.pack(self.snapshot()),
            restore=lambda payload: self.apply_snapshot(
                serialization_codec.unpack(payload)
            ),
        )

        # current model timestep.
        # e.g. current time = self._timestep * self._timestep_size_s
        self._timestep: int = 0

        ### type hints ###
        # for clarify and type checking, the following type hints are defined
        # here, however the names are bound and initialized in `initialize`.
        self.cfg_bmi: dict[str, typing.Any]
        self.ensemble_members: list[EnsembleMember]
        self._fingerprint: bytes
        """model identity derived from the ensemble members; see `compute_fingerprint`"""

    def initialize(self, config_file: str) -> None:
        # read and setup main configuration file
        with open(config_file, "r") as fp:
            self.cfg_bmi = yaml.load(fp, Loader=SafeLoader)
        coerce_config(self.cfg_bmi)

        # TODO: aaraney: config logging levels to python logging levels
        # setup logging
        # self.cfg_bmi["verbose"]
        configure_logging()

        # Before any inference: ranks pinned to one core must not run a node-sized thread pool.
        configure_torch_threads(self.cfg_bmi.get(TORCH_THREADS_KEY))

        # ----------- The output is area normalized, this is needed to un-normalize it
        #                         mm->m                             km2 -> m2          hour->s
        output_factor_cms = (
            (1 / 1000) * (self.cfg_bmi["area_sqkm"] * 1000 * 1000) * (1 / 3600)
        )

        # `resolve()` so every spelling of one config path shares a single loaded model.
        self.ensemble_members = [
            EnsembleMember(load_trained_model(cfg_file.resolve()), output_factor_cms)
            for cfg_file in self.cfg_bmi["train_cfg_file"]
        ]

        static_attributes = resolve_static_attributes(
            self.cfg_bmi, (m.model for m in self.ensemble_members)
        )
        self._static_inputs = State(
            vars=(
                Var(name=name, unit="1", value=bmi_array([value]))
                for name, value in static_attributes.items()
            )
        )

        # identity of the fully constructed ensemble, used to reject state
        # payloads produced by a differently configured module.
        self._fingerprint = compute_fingerprint(self.ensemble_members)

    def update(self) -> None:
        """update a single timestep."""

        # wrap dynamic and static inputs in a container so an `EnsembleMember`
        # can access both like they are from the same `State` object.
        #
        # each ensemble member will query the `state` object for its required inputs.
        # this could ensemble members with a different number of required features in the future.
        state = StateFacade(self._dynamic_inputs, self._static_inputs)

        outputs: dict[str, list[float]] = collections.defaultdict(list)
        for member in self.ensemble_members:
            for output in member.update(state):
                assert len(output.value) == 1, (
                    f"expected output of length 1, got {len(output.value)}"
                )
                outputs[output.name].append(output.value[0])

        # ensemble output and set output variables
        for name, values in outputs.items():
            self._outputs.set_value(name, np.mean(values, dtype="float64"))

        # increment model timestep
        self._timestep += 1

    def update_until(self, time: float) -> None:
        if time <= self.get_current_time():
            current_time = self.get_current_time()
            logger.warning("no update performed: time=%s <= current_time=%s", time, current_time)
            return None

        n_steps, remainder = divmod(
            time - self.get_current_time(), self.get_time_step()
        )

        if remainder != 0:
            logger.warning(
                "time is not multiple of time step size. updating until: %s", (time - remainder)
            )

        for _ in range(int(n_steps)):
            self.update()

    def finalize(self) -> None:
        # release any captured serialization payload; nothing else is held.
        self._serialization.release()

    def get_component_name(self) -> str:
        return "LSTM"

    def get_input_item_count(self) -> int:
        return len(self._dynamic_inputs)

    def get_output_item_count(self) -> int:
        return len(self._outputs)

    def get_input_var_names(self) -> tuple[str, ...]:  # type: ignore
        return tuple(self._dynamic_inputs.names())

    def get_output_var_names(self) -> tuple[str, ...]:  # type: ignore
        return tuple(self._outputs.names())

    def get_var_grid(self, name: str) -> int:
        # Note: all vars have grid 0 but check if its in names list first
        # raises KeyError on failure
        first_containing(name, self._outputs, self._dynamic_inputs)
        return 0

    def get_var_type(self, name: str) -> str:
        return self.get_value_ptr(name).dtype.name

    def get_var_units(self, name: str) -> str:
        return first_containing(
            name, self._outputs, self._dynamic_inputs, self._serialization
        ).unit(name)

    def get_var_itemsize(self, name: str) -> int:
        return self.get_value_ptr(name).itemsize

    def get_var_nbytes(self, name: str) -> int:
        return self.get_var_itemsize(name) * len(self.get_value_ptr(name))

    def get_var_location(self, name: str) -> str:
        # raises KeyError on failure
        first_containing(name, self._outputs, self._dynamic_inputs)
        return "node"

    def get_current_time(self) -> float:
        return self._timestep * self._timestep_size_s

    def get_start_time(self) -> float:
        return 0

    def get_end_time(self) -> float:
        return np.finfo("d").max  # type: ignore

    def get_time_units(self) -> str:
        return "s"

    def get_time_step(self) -> float:
        return self._timestep_size_s

    def get_value(self, name: str, dest: np.ndarray) -> np.ndarray:
        """_Copies_ a variable's np.NDArray into `dest` and returns `dest`."""
        dest[:] = self.get_value_ptr(name)
        return dest

    def get_value_ptr(self, name: str) -> np.ndarray:
        """Returns a _reference_ to a variable's np.NDArray."""
        return first_containing(
            name, self._outputs, self._dynamic_inputs, self._serialization
        ).value(name)

    def get_value_at_indices(
        self, name: str, dest: np.ndarray, inds: np.ndarray
    ) -> np.ndarray:
        return first_containing(
            name, self._outputs, self._dynamic_inputs
        ).value_at_indices(name, dest, inds)

    def set_value(self, name: str, src: np.ndarray) -> None:
        # the reserved protocol names are triggers, an announcement, or a
        # payload delivery rather than values; the protocol object dispatches.
        if name in self._serialization:
            return self._serialization.set_value(name, src)
        return first_containing(name, self._outputs, self._dynamic_inputs).set_value(
            name, src
        )

    def set_value_at_indices(
        self, name: str, inds: np.ndarray, src: np.ndarray
    ) -> None:
        return first_containing(
            name, self._outputs, self._dynamic_inputs
        ).set_value_at_indices(name, inds, src)

    # Grid information
    def get_grid_rank(self, grid: int) -> int:
        # 0 is the only id we have
        if grid == 0:
            return 1
        raise RuntimeError(f"unsupported grid rank: {grid!s}. only support 0")

    def get_grid_size(self, grid: int) -> int:
        # 0 is the only id we have
        if grid == 0:
            return 1
        raise RuntimeError(f"unsupported grid size: {grid!s}. only support 0")

    def get_grid_type(self, grid: int) -> str:
        # 0 is the only id we have
        if grid == 0:
            return "scalar"
        raise RuntimeError(f"unsupported grid type: {grid!s}. only support 0")

    # Computed-state transfer for the ngen BMI Serialization Protocol. The
    # protocol's reserved variables, buffer, and dispatch live in
    # `serialization_protocol`; the payload format lives in
    # `serialization_codec`. This class contributes only the model identity
    # and the two methods that move state into and out of a `Snapshot`.

    def _require_initialized(self, action: str) -> None:
        """Raise `RuntimeError` naming `action` unless `initialize()` has run."""
        if not hasattr(self, "_fingerprint"):
            raise RuntimeError(f"cannot {action} before initialize() is called")

    @property
    def fingerprint(self) -> bytes:
        """
        Identity of the initialized ensemble; see `compute_fingerprint`.

        Read-only. Raises `RuntimeError` before `initialize()`.
        """
        self._require_initialized("read the model fingerprint")
        return self._fingerprint

    def snapshot(self) -> serialization_codec.Snapshot:
        """
        Gather the module's computed state into a codec `Snapshot`.

        The snapshot holds the timestep counter, the fingerprint, each ensemble
        member's hidden and cell state as flat float32 copies (in ensemble
        order), and the output values as float64 (in output variable order).
        Nothing on the module is modified, and the returned arrays do not alias
        the member tensors. Raises `RuntimeError` before `initialize()`.
        """
        self._require_initialized("capture serialization state")
        members = [member.state_arrays() for member in self.ensemble_members]
        outputs = np.array(
            [self._outputs.value(name)[0] for name in self._outputs.names()],
            dtype=serialization_codec.OUTPUT_DTYPE,
        )
        return serialization_codec.Snapshot(
            timestep=self._timestep,
            fingerprint=self._fingerprint,
            members=members,
            outputs=outputs,
        )

    def apply_snapshot(self, snapshot: serialization_codec.Snapshot) -> None:
        """
        Write a codec `Snapshot` into the member tensors and output values.

        The checks run in order: fingerprint, member count, each member's
        hidden and cell sizes, output count. Every check is made before the
        first assignment, so a rejected snapshot mutates nothing. This is the
        only place the fingerprint is compared. The snapshot's timestep is
        deliberately not applied: the module's clock after a restore is
        whatever `initialize()` set, and ngen advances the model relative to
        its own time (see the README's serialization section).
        """
        self._require_initialized("restore serialization state")
        fingerprint = bytes(snapshot.fingerprint)
        if fingerprint != self._fingerprint:
            raise serialization_codec.PayloadError(
                "fingerprint mismatch: payload carries "
                f"{fingerprint!r} but this module expects {self._fingerprint!r}"
            )

        members = self.ensemble_members
        if len(snapshot.members) != len(members):
            raise serialization_codec.PayloadError(
                f"payload carries {len(snapshot.members)} ensemble members but this "
                f"module has {len(members)}"
            )
        for index, (member, (hidden, cell)) in enumerate(zip(members, snapshot.members)):
            if hidden.size != member.model.hidden_size or cell.size != member.model.hidden_size:
                raise serialization_codec.PayloadError(
                    f"payload member {index} carries hidden size {hidden.size} "
                    f"(cell {cell.size}) but this module's member has "
                    f"{member.model.hidden_size}"
                )
        output_names = list(self._outputs.names())
        if len(snapshot.outputs) != len(output_names):
            raise serialization_codec.PayloadError(
                f"payload carries {len(snapshot.outputs)} output values but this "
                f"module has {len(output_names)}"
            )
        outputs = np.asarray(snapshot.outputs, dtype="float64")

        # all checks passed: apply.
        for member, (hidden, cell) in zip(members, snapshot.members):
            member.set_state_arrays(hidden, cell)
        for name, value in zip(output_names, outputs):
            self._outputs.set_value(name, value)


def coerce_config(cfg: dict[str, typing.Any]):
    for key, val in cfg.items():
        # Handle 'train_cfg_file' specifically to ensure it is always a list
        if key == "train_cfg_file":
            if val is not None and val != "None":
                if isinstance(val, list):
                    cfg[key] = [Path(element) for element in val]
                else:
                    cfg[key] = [Path(val)]
            else:
                cfg[key] = []

        # Convert all path strings to PosixPath objects for other keys
        elif any([key.endswith(x) for x in ["_dir", "_path", "_file", "_files"]]):
            if val is not None and val != "None":
                if isinstance(val, list):
                    temp_list = []
                    for element in val:
                        temp_list.append(Path(element))
                    cfg[key] = temp_list
                else:
                    cfg[key] = Path(val)
            else:
                cfg[key] = None

        # Convert Dates to pandas Datetime indexs
        elif key.endswith("_date"):
            if isinstance(val, list):
                temp_list = []
                for elem in val:
                    temp_list.append(pd.to_datetime(elem, format="%d/%m/%Y"))
                cfg[key] = temp_list
            else:
                cfg[key] = pd.to_datetime(val, format="%d/%m/%Y")


def first_containing(name: str, *states: State) -> State:
    """
    Return the first `State` object containing `name` in `states`.
    Otherwise, raise `KeyError`.
    """
    for state in states:
        if name in state:
            return state
    raise KeyError(f"unknown name: {name!s}")
