
import numpy as np
import torch
from torch import nn
#import data_tools
from pathlib import Path
from netCDF4 import Dataset
# This is the BMI LSTM that we will be running
import bmi_lstm

# creating an instance of an LSTM model
print('creating an instance of an BMI_LSTM model object')
model = bmi_lstm.bmi_LSTM()

# Initializing the BMI
print('Initializing the BMI')
model.initialize(bmi_cfg_file=Path('./bmi_config_files/01022500_A.yml'))

# Get input data that matches the LSTM test runs
print('Get input data that matches the LSTM test runs')
sample_data = Dataset(Path('./data/usgs-streamflow-nldas_hourly.nc'), 'r')

# Now loop through the inputs, set the forcing values, and update the model
print('Now loop through the inputs, set the forcing values, and update the model')
for precip, temp in zip(list(sample_data['total_precipitation'][3].data),
                        list(sample_data['temperature'][3].data)):

    model.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux',precip)
    model.set_value('land_surface_air__temperature',temp)

    print('the temperature and precipitation are set to {:.2f} and {:.2f}'.format(model.get_value('land_surface_air__temperature'), 
                                                     model.get_value('atmosphere_water__time_integral_of_precipitation_mass_flux')))
    model.update()

    print('the streamflow (CFS) at time {} is {:.2f}'.format(model.get_current_time(), 
                                model.get_value('land_surface_water__runoff_volume_flux')))

    if model.t > 100:
        print('stopping the loop')
        break

# Finalizing the BMI
print('Finalizing the BMI')
model.finalize()