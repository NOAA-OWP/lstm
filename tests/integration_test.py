#
# Copyright (C) 2025 Austin Raney, Lynker
#
# Author: Austin Raney <araney@lynker.com>
#
from __future__ import annotations

import contextlib
import os
from pathlib import Path

import netCDF4 as nc
import numpy as np

from lstm import bmi_lstm

REPO_ROOT = Path(__file__).parent.parent
TEST_DIR = Path(__file__).parent


def test_single_lstm_member_nldas_configuration():
    # "02064000", "01547700", "03015500", "01022500"
    basin_id = "02064000"
    bmi_cfg_file = REPO_ROOT / f"configs/{basin_id}_nh_NLDAS_hourly.yml"
    forcing_file = REPO_ROOT / "data/usgs-streamflow-nldas_hourly.nc"

    with nc.Dataset(forcing_file, "r") as forcing:

        def find_basin_var_idx(basin_id: str, ds: nc.Dataset) -> int:
            basins = ds.variables["basin"][:]
            basin_var_idxs = np.where(basins == basin_id)[0]
            assert len(basin_var_idxs) == 1
            return basin_var_idxs[0]

        basin_var_idx = find_basin_var_idx(basin_id, forcing)

        forcing_variable_name_mapping = {
            "total_precipitation": "atmosphere_water__liquid_equivalent_precipitation_rate",
            "temperature": "land_surface_air__temperature",
            "longwave_radiation": "land_surface_radiation~incoming~longwave__energy_flux",
            "shortwave_radiation": "land_surface_radiation~incoming~shortwave__energy_flux",
            "pressure": "land_surface_air__pressure",
            "specific_humidity": "atmosphere_air_water~vapor__relative_saturation",
            "wind_u": "land_surface_wind__x_component_of_velocity",
            "wind_v": "land_surface_wind__y_component_of_velocity",
        }

        # Regenerated after the config's static attributes were corrected to basin
        # 02064000's CAMELS values (elev_mean 192.21, slope_mean 9.95686, area_sqkm
        # 427.77). The previous array was produced with basin 12010000's attributes
        # that the config file carried at the time.
        expected_output_mm_hr = np.array(
            [
                0.2828137805237887,
                0.13929926039671825,
                0.1437470301914603,
                0.1514676301653708,
                0.15951013170062556,
                0.14394014512827535,
                0.11541240737888225,
                0.0775181564441021,
                0.04705949953481636,
                0.02527941228600472,
                0.0005993753610225028,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype="float64",
        )

        # Config and trained-model paths are relative to the repository root.
        with pushd(REPO_ROOT):
            # Create an instance of the LSTM model with BMI
            model_instance = bmi_lstm.bmi_LSTM()

            # Initialize the model with a configuration file
            model_instance.initialize(str(bmi_cfg_file))

            nts = len(expected_output_mm_hr)
            runoff_depth_m_hr = np.zeros(nts)
            for ts in range(nts):
                for forcing_name, bmi_forcing_name in forcing_variable_name_mapping.items():
                    model_instance.set_value(
                        bmi_forcing_name, forcing.variables[forcing_name][basin_var_idx, ts]
                    )
                # Update the model
                model_instance.update()

                # Retrieve and scale the runoff output
                model_instance.get_value(
                    "land_surface_water__runoff_depth", runoff_depth_m_hr[ts : ts + 1]
                )

            runoff_depth_mm_hr = runoff_depth_m_hr * 1000  # m/hr -> mm/hr
            np.testing.assert_array_almost_equal(
                runoff_depth_mm_hr, expected_output_mm_hr, decimal=6
            )


@contextlib.contextmanager
def pushd(target: Path):
    saved = os.getcwd()
    os.chdir(target)
    try:
        yield saved
    finally:
        os.chdir(saved)
