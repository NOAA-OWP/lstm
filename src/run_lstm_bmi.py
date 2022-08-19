
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

#dest0 = np.empty(model.get_grid_size(0), dtype=float)

# Now loop through the inputs, set the forcing values, and update the model
#print('Now loop through the inputs, set the forcing values, and update the model')
print('Set values & update model for number of timesteps = 10')
for precip, temp in zip(list(sample_data['total_precipitation'][3].data),
                        list(sample_data['temperature'][3].data)):

    model.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux',np.atleast_1d(precip))
    model.set_value('land_surface_air__temperature',np.atleast_1d(temp))

    # JG Note: 080922 if/when get_value() def is updated 
    # 
    # print(' Temperature and precipitation are set to', model.get_value('land_surface_air__temperature',dest0)[0].round(3),
    #     'and', model.get_value('atmosphere_water__time_integral_of_precipitation_mass_flux',dest0)[0].round(3)) 
    # print(' Streamflow (cms) at time', model.get_current_time(), model.get_time_units(), 'is',
    #                             model.get_value('land_surface_water__runoff_volume_flux',dest0)[0].round(3))
 
    print(' temperature and precipitation are set to {:.2f} and {:.2f}'.format(model.get_value('land_surface_air__temperature'), 
                                                     model.get_value('atmosphere_water__time_integral_of_precipitation_mass_flux')))

    print(' streamflow (cms) at time {} {} is {:.2f}'.format(model.get_current_time(), model.get_time_units(), 
                                model.get_value('land_surface_water__runoff_volume_flux')))    

    #model.update_until(model.t+model._time_step_size)
    model.update()

    if model.t > 10:
        #print('Stopping the loop')
        break

# Finalizing the BMI
print('Finalizing the BMI')
model.finalize()
