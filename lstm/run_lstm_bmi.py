
import numpy as np
import torch
from torch import nn
#import data_tools
from pathlib import Path
from netCDF4 import Dataset
# This is the BMI LSTM that we will be running
import bmi_lstm

# Define primary bmi config and input data file paths
bmi_cfg_file=Path('./../bmi_config_files/01022500_hourly_all_attributes_forcings.yml')
sample_data_file = Path('./../data/usgs-streamflow-nldas_hourly.nc')

# creating an instance of an LSTM model
print('Creating an instance of an BMI_LSTM model object')
model = bmi_lstm.bmi_LSTM()

# Initializing the BMI
print('Initializing the BMI')
model.initialize(bmi_cfg_file)

# Get input data that matches the LSTM test runs
print('Gathering input data')
sample_data = Dataset(sample_data_file, 'r')

# Now loop through the inputs, set the forcing values, and update the model
print('Set values & update model for number of timesteps = 10')
for precip, temp in zip(list(sample_data['total_precipitation'][3].data),
                        list(sample_data['temperature'][3].data)):

    model.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux',np.atleast_1d(precip))
    model.set_value('land_surface_air__temperature',np.atleast_1d(temp))

    dest_array = np.zeros(1)
    model.get_value('land_surface_air__temperature', dest_array)
    temperature = dest_array[0]
    model.get_value('atmosphere_water__time_integral_of_precipitation_mass_flux', dest_array)
    precip = dest_array[0]

    print(' Temperature and precipitation are set to {:.2f} and {:.2f}'.format(temperature, precip))
    #model.update_until(model.t+model._time_step_size)
    model.update()

    model.get_value('land_surface_water__runoff_volume_flux', dest_array)
    runoff = dest_array[0]

    print(' Streamflow (cms) at time {} ({}) is {:.2f}'.format(model.get_current_time(), model.get_time_units(), runoff))

    if model.t > 10:
        #print('Stopping the loop')
        break

# Finalizing the BMI
print('Finalizing the BMI')
model.finalize()
