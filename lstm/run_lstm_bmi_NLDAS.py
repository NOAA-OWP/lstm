import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from netCDF4 import Dataset
from lstm import bmi_lstm  # Load module bmi_lstm (bmi_lstm.py) from lstm package.
import pickle
import xarray as xr

import sys
import os, os.path
lstm_dir = os.path.expanduser('../lstm/')
os.chdir( lstm_dir )
import pandas as pd

sample_data = Dataset('../data/usgs-streamflow-nldas_hourly.nc', 'r')
sample_basins = {sample_data['basin'][x]:x for x in range(len(list(sample_data['basin'])))}

basin_id = "01022500" # Chose from: ["01022500", "03015500", "01547700", "02064000"] 

# Create an instance of the LSTM model with BMI
model_instance = bmi_lstm.bmi_LSTM()

# Initialize the model with a configuration file
model_instance.initialize(bmi_cfg_file=Path(f'../bmi_config_files/{basin_id}_nh_NLDAS_hourly.yml'))

# Get the index of the basin
basin_index = np.where(sample_data["basin"][:] == basin_id)[0][0]  # Extracts the correct index

# Get sample data time series for precip and temp
precip_data = sample_data['total_precipitation'][basin_index, :].data  # Index by both dimensions
temp_data   = sample_data['temperature'][basin_index, :].data + 273.15  # Index by both dimensions
n_precip    = precip_data.size

print('Forcing data info:')
print('  n_precip =', n_precip)
print('  n_temp   =', temp_data.size)
print('  precip_data.dtype =', precip_data.dtype)
print('  temp_data.dtype   =', temp_data.dtype)
print('  precip:  min, max =', precip_data.min(), ',', precip_data.max() )
print('  temp:    min, max =', temp_data.min(), ',', temp_data.max() )
print()

# Store output values in an array, so we can plot it afterwards (faster)
runoff_output_ = np.zeros( n_precip )

k = 0
VERBOSE = False
print('Working, please wait...')
for k in range( n_precip ):
    precip = precip_data[k]
    temp   = temp_data[k]
    if (VERBOSE):
        print('k, precip, temp =', k, ',', precip, ',', temp)

    # Set the model forcings to those in the sample data

    model_instance.set_value('atmosphere_water__liquid_equivalent_precipitation_rate', precip)
    model_instance.set_value('land_surface_air__temperature',temp)

    # Updating the model calculates the runoff from the inputs and the model state at this time step
    model_instance.update()

    # Add the output to a list so we can plot
    dest_array = np.zeros(1)

    model_instance.get_value('land_surface_water__runoff_depth', dest_array)
    runoff_ = dest_array[0]
    # print('val =', runoff_limited)
    
    #------------------------------------------------
    # Make output unit consistant with CAMELS mm/hr
    #------------------------------------------------
    runoff_ *= 1000   # (correct factor is 1000)
    runoff_output_[ k ] = runoff_

# Calculate a metric
obs = np.array(sample_data['qobs_CAMELS_mm_per_hour'][basin_index])
sim = runoff_output_
### sim = np.array(runoff_output_list_limited)
denominator = ((obs - obs.mean())**2).sum()
numerator = ((sim - obs)**2).sum()
value = 1 - numerator / denominator
print("NSE: {:.2f}".format(1 - numerator / denominator))