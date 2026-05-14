##################################################################
# LSTM TEST EXAMPLES WITH PRECIPITATION AND AIR TEMPERATURE INPUTS
# Target NSE are provided 
##################################################################

import numpy as np
import torch
from torch import nn
#import data_tools
from pathlib import Path
from netCDF4 import Dataset
from lstm import bmi_lstm


USE_PATH = True
verbose  = False

# Two Test example basins
test_basins = ["02064000", "01022500"]

bmi_cfg_file     = f'./configs/{test_basins[0]}_nh_NLDAS_hourly.yml' # change index 0 to 1 to test the other basin

sample_data_file = './data/usgs-streamflow-nldas_hourly.nc'

# creating an instance of an LSTM model
print('Creating an instance of an BMI_LSTM model object')
model = bmi_lstm.bmi_LSTM()

# Initializing the BMI
print('Initializing the BMI')
model.initialize(bmi_cfg_file)

##############################################################
# Get input data that matches the LSTM test runs
print('Gathering input data')
sample_data = Dataset(sample_data_file, 'r')

# Sample basins
sample_basins = {sample_data['basin'][x]:x for x in range(len(list(sample_data['basin'])))}
print ("Sample basins: ", sample_basins)

# Currently selected basins
current_basin_index = sample_basins[model.cfg_bmi['basin_id']]
current_basin_gage_id = list(sample_basins.keys())[current_basin_index]
print ("Current Test Basin: ", current_basin_gage_id)


##############################################################
# Precip and temparature data
precip_data   = sample_data['total_precipitation'][current_basin_index].data
temp_data     = sample_data['temperature'][current_basin_index].data
n_timesteps   = precip_data.size
runoff_output = np.zeros(n_timesteps)
m_to_mm = 1000
C_to_K = 273.15
##############################################################


for ts in range(n_timesteps):
    precip = precip_data[ts]
    temp   = temp_data[ts] + C_to_K
    model.set_value('atmosphere_water__liquid_equivalent_precipitation_rate',np.atleast_1d(precip))
    model.set_value('land_surface_air__temperature',np.atleast_1d(temp))

    if verbose:
        print(' Temperature and precipitation are set to {:.2f} and {:.2f}'.format(temp, precip))

    model.update()

    dest_array = np.zeros(1)
    model.get_value('land_surface_water__runoff_depth', dest_array) # runoff_depht in meters
    runoff = dest_array[0]
    runoff_output[ts] = runoff * m_to_mm

    if verbose:
        print(' Streamflow (cms) at time {} ({}) is {:.2f}'.format(model.get_current_time(), model.get_time_units(), runoff))

# Finalizing the BMI
print('Finalizing the BMI')
model.finalize()

target_nse = None

if (current_basin_gage_id == "01022500"):
    target_nse = 0.58
if (current_basin_gage_id == "02064000"):
    target_nse = 0.11

    
# Calculate a metric
obs = np.array(sample_data['qobs_CAMELS_mm_per_hour'][current_basin_index])
sim = runoff_output

denominator = ((obs - obs.mean())**2).sum()
numerator   = ((sim - obs)**2).sum()
current_nse = 1 - numerator / denominator

print ("======= NSE Comparison ===========")
print("Target NSE  = {:.2f}".format(target_nse))
print("Current NSE = {:.2f}".format(current_nse))
