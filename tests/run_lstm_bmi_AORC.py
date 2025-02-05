from __future__ import annotations

from pathlib import Path
import numpy as np
from netCDF4 import Dataset
from lstm import bmi_lstm

REPO_ROOT = Path(__file__).parent.parent
bmi_cfg_file = REPO_ROOT / "bmi_config_files/01022500_nh_NLDAS_hourly.yml"
sample_data_file = REPO_ROOT / "data/usgs-streamflow-nldas_hourly.nc"

# creating an instance of an LSTM model
print("Creating an instance of an BMI_LSTM model object")
model = bmi_lstm.bmi_LSTM()

# Initializing the BMI
print("Initializing the BMI")
model.initialize(str(bmi_cfg_file))

# Get input data that matches the LSTM test runs
print("Gathering input data")
sample_data = Dataset(sample_data_file, "r")

# Now loop through the inputs, set the forcing values, and update the model
print("Set values & update model for number of timesteps = 100")
for precip, temp in zip(
    list(sample_data["total_precipitation"][3].data),
    list(sample_data["temperature"][3].data),
):
    model.set_value(
        "atmosphere_water__liquid_equivalent_precipitation_rate", np.atleast_1d(precip)
    )
    model.set_value("land_surface_air__temperature", np.atleast_1d(temp))

    print(
        "Temperature and precipitation are set to {:.2f} and {:.2f}".format(
            temp, precip
        )
    )
    model.update()

    dest_array = np.zeros(1)
    model.get_value("land_surface_water__runoff_volume_flux", dest_array)
    runoff = dest_array[0]

    print(
        " Streamflow (cms) at time {} ({}) is {:.2f}".format(
            model.get_current_time(), model.get_time_units(), runoff
        )
    )

    ts = model.get_current_time() // model.get_time_step()
    if ts > 100:
        # print('Stopping the loop')
        break

# Finalizing the BMI
print("Finalizing the BMI")
model.finalize()
