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
import typing
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
import pickle
import torch
import yaml
try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

from . import nextgen_cuda_lstm
from .base import BmiBase
from .model_state import State, StateFacade, Var

import ewts
LOG = ewts.get_logger(ewts.LSTM_ID)


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
# --------------   Static Attributes -----------------------------
_static_input_vars = [
    ("basin__mean_of_elevation", "m"),
    ("basin__mean_of_slope", "m km-1"),
]

_output_vars = [
    ("land_surface_water__runoff_volume_flux", "m3 s-1"),
    ("land_surface_water__runoff_depth", "m"),
    ("precipitation_rate", "mm s-1"),
]

# --------------    Name Mappings    -----------------------------
INTERNAL_NAME_CROSSWALK = {
    # dynamic inputs
    "DLWRF_surface": "land_surface_radiation~incoming~longwave__energy_flux",
    "PRES_surface": "land_surface_air__pressure",
    "SPFH_2maboveground": "atmosphere_air_water~vapor__relative_saturation",
    "APCP_surface": "atmosphere_water__liquid_equivalent_precipitation_rate",
    "DSWRF_surface": "land_surface_radiation~incoming~shortwave__energy_flux",
    "TMP_2maboveground": "land_surface_air__temperature",
    "UGRD_10maboveground": "land_surface_wind__x_component_of_velocity",
    "VGRD_10maboveground": "land_surface_wind__y_component_of_velocity",
    # static inputs
    "elev_mean": "basin__mean_of_elevation",
    "slope_mean": "basin__mean_of_slope",
    # outputs
    "streamflow_cms": "land_surface_water__runoff_volume_flux",
    "streamflow_m": "land_surface_water__runoff_depth",
}
"""
Mapping from 'internal' names to 'external' names (names exposed via bmi).
Internal names are meaningful to trained lstm models.
Its healthy to think of internal names as aliases to external names.
"""

EXTERNAL_NAME_CROSSWALK = {v: k for k, v in INTERNAL_NAME_CROSSWALK.items()}
"""Mapping from 'external' names to 'internal' names."""


def crosswalk_to_external(name: str):
    """Return the external name (the name exposed via BMI) for a given internal name."""
    return INTERNAL_NAME_CROSSWALK[name]


def crosswalk_to_interal(name: str):
    """Return the internal name for a given external name (the name exposed via BMI)."""
    return EXTERNAL_NAME_CROSSWALK[name]


# ---------------  Ensemble Member -----------------------------


class EnsembleMember:
    """
    An `EnsembleMember` is responsible for initializing and maintaining an LSTM model,
    handling input scaling, managing hidden and cell states, and performing
    inference using the trained model.
    """

    def __init__(self, cfg: dict[str, typing.Any], output_scaling_factor_cms: float):
        self.cfg = cfg
        # NOTE: aaraney: not sure if this *should* go here. leaving it for now.
        self.output_scaling_factor_cms = output_scaling_factor_cms

        # load training feature scales
        scaler_file = cfg["run_dir"] / "train_data/train_data_scaler.yml"
        with scaler_file.open("r") as fp:
            train_data_scaler = yaml.load(fp, Loader=SafeLoader)
        self.scalars = load_training_scalars(cfg, train_data_scaler)

        # initialize torch lstm object
        self.lstm = initialize_lstm(cfg)

        # TODO: aaraney: how to handle input mapping conceptually?
        # NOTE: this is the expected order of variables in the model input
        # tensor, which is required to match the training order when used
        self.input_names = cfg["dynamic_inputs"] + cfg["static_attributes"]

        # WARNING: This implementation of the LSTM can only handle a batch size of 1
        # No need to included different batch sizes
        batch_size = 1
        hidden_layer_size = cfg["hidden_size"]
        # if init_config['initial_state'] == 'zero':
        # NOTE: aaraney: assume initial state is always zero (ask jframe about this. no other option now)
        self.h_t = torch.zeros(1, batch_size, hidden_layer_size).float()
        self.c_t = torch.zeros(1, batch_size, hidden_layer_size).float()

    def update(self, state: Valuer) -> typing.Iterable[Var]:
        """
        Run a single model timestep and return the model inference values.

        `state` contains the input variable names, units, and values for the
        current iteration.
        """
        with torch.no_grad():
            inputs = gather_inputs(state, self.input_names)

            # Retrieve precipitation value for output
            precipitation_mm_h = state.value("atmosphere_water__liquid_equivalent_precipitation_rate")

            scaled = scale_inputs(
                inputs, self.scalars.input_mean, self.scalars.input_std
            )
            input_tensor = torch.tensor(scaled)
            lstm_output, self.h_t, self.c_t = self.lstm.forward(
                input_tensor, self.h_t, self.c_t
            )
            # TODO: aaraney, there is gap here between mapping 'internal'
            # output names to 'external' output names. Right now this is
            # hard-coded and handled in `scale_outputs`. Introduce semantics
            # for more generally handling outputs.
            yield from scale_outputs(
                self.cfg,
                lstm_output,
                self.scalars.output_mean,
                self.scalars.output_std,
                self.output_scaling_factor_cms,
                precipitation_mm_h
            )

    def serialize(self):
        return {
            "h": self.h_t.numpy(),
            "c": self.c_t.numpy()
        }

    def deserialize(self, data: dict):
        self.h_t = torch.from_numpy(data["h"])
        self.c_t = torch.from_numpy(data["c"])


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
    # Collect the LSTM model architecture details from the configuration file
    input_size = len(cfg["dynamic_inputs"]) + len(cfg["static_attributes"])
    # TODO: aaraney: verify there is a mapping from internal names to external names
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
    return lstm


def gather_inputs(
    state: Valuer, internal_input_names: typing.Iterable[str]
) -> npt.NDArray:
    LOG.debug("Collecting LSTM inputs ...")

    input_list = []
    for lstm_name in internal_input_names:
        bmi_name = crosswalk_to_external(lstm_name)
        value = state.value(bmi_name)
        assert value.size == 1, "`value` should a single scalar in a 1d array"
        input_list.append(value[0])

        LOG.debug(f"  {lstm_name=}")
        LOG.debug(f"  {bmi_name=}")
        LOG.debug(f"  {type(value)=}")
        LOG.debug(f"  {value=}")

    collected = bmi_array(input_list)
    LOG.debug(f"Collected inputs: %s", collected)
    return collected


def scale_inputs(
    input: npt.NDArray, mean: npt.NDArray, std: npt.NDArray
) -> npt.NDArray:
    LOG.debug("Normalizing the tensor...")
    LOG.debug("  input_mean = %s", mean)
    LOG.debug("  input_std  = %s", std)

    # Center and scale the input values for use in torch
    input_array_scaled = (input - mean) / std
    LOG.debug("### input_array = %s", input)
    LOG.debug("### dtype(input_array) = %s", input.dtype)
    LOG.debug("### type(input_array_scaled) = %s", type(input_array_scaled))
    LOG.debug("### dtype(input_array_scaled) = %s", input_array_scaled.dtype)
    return input_array_scaled


def scale_outputs(
    cfg: dict[str, typing.Any],
    output: torch.tensor,
    output_mean: npt.NDArray,
    output_std: npt.NDArray,
    output_scale_factor_cms: float,
    precipitation_value: npt.NDArray,
):
    LOG.debug(f"model output: {output[0, 0, 0].numpy().tolist()}")

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

    # Convert precipitation for mm/h to mm/s for output
    precip_mms = precipitation_value[0] / 3600.0

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
        Var(
            name="precipitation_rate",
            unit="mm s-1",
            value=bmi_array([precip_mms])
        ),
    )


# ---------------  LSTM BMI Wrapper  -----------------------------


def build_state(vars: typing.Iterable[tuple[str, str]]) -> State:
    """
    Create a `State` object from a collection of (name: str, unit: str) tuples.
    Each `Var`'s `value` array is initialized to an `np.array([0.0], dtype="float64"))`.
    """
    g = (Var(name=name, unit=unit, value=bmi_array([0.0])) for (name, unit) in vars)
    return State(vars=g)


def load_static_attributes(cfg: dict[str, typing.Any], state: State):
    for external_name in state.names():
        internal_name = crosswalk_to_interal(external_name)
        value = cfg[internal_name]
        state.set_value(external_name, bmi_array([value]))


class bmi_LSTM(BmiBase):
    _timestep_size_s: typing.Final[int] = 3600
    """model timestep size in seconds"""

    def __init__(self) -> None:
        # _bmi_ variable state; this is separate from lstm ensemble member state.
        self._dynamic_inputs = build_state(_dynamic_input_vars)
        self._static_inputs = build_state(_static_input_vars)
        self._outputs = build_state(_output_vars)

        # current model timestep.
        # e.g. current time = self._timestep * self._timestep_size_s
        self._timestep: int = 0

        ### type hints ###
        # for clarify and type checking, the following type hints are defined
        # here, however the names are bound and initialized in `initialize`.
        self.cfg_bmi: dict[str, typing.Any]
        self.ensemble_members: list[EnsembleMember]

        # statically stored seriaized data
        self._serialized_size = np.array([0], dtype=np.uint64)
        self._serialized = np.array([], dtype=np.uint8)

    def initialize(self, config_file: str) -> None:

        # This is required prior to the first log message is issued by t-route.
        LOG.bind()
        
        LOG.info(f"Initializing with {config_file}")

        # read and setup main configuration file
        with open(config_file, "r") as fp:
            self.cfg_bmi = yaml.load(fp, Loader=SafeLoader)
        coerce_config(self.cfg_bmi)

        # ----------- The output is area normalized, this is needed to un-normalize it
        #                         mm->m                             km2 -> m2          hour->s
        output_factor_cms = (
            (1 / 1000) * (self.cfg_bmi["area_sqkm"] * 1000 * 1000) * (1 / 3600)
        )

        # initialize ensemble members
        self.ensemble_members = []
        for member_cfg_file in self.cfg_bmi["train_cfg_file"]:
            cfg = yaml.load(member_cfg_file.read_text(), Loader=SafeLoader)
            coerce_config(cfg)
            member = EnsembleMember(cfg, output_factor_cms)
            self.ensemble_members.append(member)

        # load static variables from config into state
        load_static_attributes(self.cfg_bmi, self._static_inputs)

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
            LOG.warning(f"no update performed: {time=} <= {current_time=}")
            return None

        n_steps, remainder = divmod(
            time - self.get_current_time(), self.get_time_step()
        )

        if remainder != 0:
            LOG.warning(
                f"time is not multiple of time step size. updating until: {time - remainder=} "
            )

        for _ in range(int(n_steps)):
            self.update()

    def finalize(self) -> None: ...

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
        if name == "serialization_state":
            return self._serialized.dtype.name
        elif name == "serialization_size" or name == "serialization_create":
            return self._serialized_size.dtype.name
        elif name == "serialization_free":
            return np.dtype(np.intc).name
        elif name == "reset_time":
            return np.dtype(np.double).name
        return self.get_value_ptr(name).dtype.name

    def get_var_units(self, name: str) -> str:
        return first_containing(name, self._outputs, self._dynamic_inputs).unit(name)

    def get_var_itemsize(self, name: str) -> int:
        if name == "serialization_state":
            return self._serialized.dtype.itemsize
        if name == "serialization_size" or name == "serialization_create":
            return self._serialized_size.dtype.itemsize
        if name == "serialization_free":
            return np.dtype(np.intc).itemsize
        if name == "reset_time":
            return np.dtype(np.double).itemsize
        return self.get_value_ptr(name).itemsize

    def get_var_nbytes(self, name: str) -> int:
        if name == "serialization_create":
            return self._serialized_size.nbytes
        if name == "serialization_free":
            return np.dtype(np.intc).itemsize
        if name == "reset_time":
            return np.dtype(np.double).itemsize
        return self.get_value_ptr(name).nbytes

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
        if name == "serialization_state":
            return self._serialized
        elif name == "serialization_size":
            return self._serialized_size
        return first_containing(name, self._outputs, self._dynamic_inputs).value(name)

    def get_value_at_indices(
        self, name: str, dest: np.ndarray, inds: np.ndarray
    ) -> np.ndarray:
        return first_containing(
            name, self._outputs, self._dynamic_inputs
        ).value_at_indices(name, dest, inds)

    def set_value(self, name: str, src: np.ndarray) -> None:
        if name == "serialization_state":
            self._deserialize(src)
        elif name == "serialization_create":
            self._serialize()
        elif name == "serialization_free":
            self._free_serialized()
        elif name == "reset_time":
            self._timestep = 0
        else:
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

    def _serialize(self):
        """Convert all dynamic properties that can change after the `bmi_LSTM` has had `initialize()` called into an object that can be serialized through `pickle`. 
        Then, set the BMI's `_serialized` property to the byte representation of that pickled data and adjust the static `_serialized_size` property."""
        data = {
            "dynamic_inputs": self._dynamic_inputs.serialize(),
            "static_inputs": self._static_inputs.serialize(),
            "outputs": self._outputs.serialize(),
            "ensemble": [em.serialize() for em in self.ensemble_members],
            "timestep": self._timestep,
        }
        serialized = pickle.dumps(data)
        self._serialized = np.array(bytearray(serialized), dtype=np.uint8)
        self._serialized_size[0] = len(self._serialized)

    def _deserialize(self, array: np.ndarray):
        """Interpret the bytes of the numpy array as previously pickled data from `_serialize()` and update the current values. 
        No data structure check will be made on the input array or loaded bytes. It will be assumed that the input data is of the same structure as what is generated from `_serialize()`."""
        data = bytes(array)
        deserialized = pickle.loads(data)
        self._dynamic_inputs.deserialize(deserialized["dynamic_inputs"])
        self._static_inputs.deserialize(deserialized["static_inputs"])
        self._outputs.deserialize(deserialized["outputs"])
        for bmi_em, data_em in zip(self.ensemble_members, deserialized["ensemble"], strict=True):
            bmi_em.deserialize(data_em)
        self._timestep = deserialized["timestep"]
        self._free_serialized()

    def _free_serialized(self):
        """Clear the current serialized data and set the size property value to 0."""
        self._serialized_size[0] = 0
        self._serialized = np.array([], dtype=self._serialized.dtype)


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
