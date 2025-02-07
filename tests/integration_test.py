from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np

from lstm import bmi_lstm

REPO_ROOT = Path(__file__).parent.parent


def test_single_lstm_member_nldas_configuration():
    # "02064000", "01547700", "03015500", "01022500"
    basin_id = "02064000"
    bmi_cfg_file = REPO_ROOT / f"bmi_config_files/{basin_id}_nh_NLDAS_hourly.yml"
    forcing_file = REPO_ROOT / "data/usgs-streamflow-nldas_hourly.nc"

    forcing = nc.Dataset(forcing_file, "r")

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

    expected_output_mm_hr = np.array(
        [
            0.22876199737556302,
            0.10911937485455514,
            0.10193460220532824,
            0.10834367594168803,
            0.1097120013273214,
            0.10629000161362612,
            0.09391478024598632,
            0.07620620229002473,
            0.0590324509299075,
            0.046721716312001726,
            0.038366058420874705,
            0.03138988153673106,
            0.02457781876555787,
            0.017631574371620662,
            0.010717597050459382,
            0.003713286008854233,
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

    # Create an instance of the LSTM model with BMI
    model_instance = bmi_lstm.bmi_LSTM()

    # Initialize the model with a configuration file
    model_instance.initialize(str(bmi_cfg_file))

    forcing = nc.Dataset(forcing_file, "r")
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
    np.testing.assert_allclose(runoff_depth_mm_hr, expected_output_mm_hr)
