
import numpy as np
import pandas as pd
import torch
from torch import nn
import pickle
import matplotlib.pyplot as plt
#import data_tools
from pathlib import Path
from netCDF4 import Dataset
# This is the LSTM we actually want to use
import bmi_lstm

# creating an instance of an LSTM model
print('creating an instance of an BMI_LSTM model object')
model = bmi_lstm.bmi_LSTM()

# Initializing the BMI
print('Initializing the BMI')
model.initialize(bmi_cfg_file=Path('lstm_bmi_config.yml'))

# Get input data that matches the LSTM test runs
print('Get input data that matches the LSTM test runs')
sample_data = Dataset(Path(model.cfg_train['run_dir'] / 'test_data/usgs-streamflow-nldas_hourly.nc'), 'r')

# Now loop through the inputs, set the forcing values, and update the model
print('Now loop through the inputs, set the forcing values, and update the model')
for precip, temp in zip(list(sample_data['total_precipitation'][3].data),
                        list(sample_data['temperature'][3].data)):
    model.set_value('atmosphere_water__liquid_equivalent_precipitation_rate',precip)
    model.set_value('land_surface_air__temperature',temp)
    print('the temperature and precipitation are set to {:.2f} and {:.2f}'.format(model.temperature, model.total_precipitation))
    model.update()
    print('the streamflow (CFS) at time {} is {:.2f}'.format(model.t, model.streamflow_cfs))

    if model.t > 100:
        print('stopping the loop')
        break

# Finalizing the BMI
print('Finalizing the BMI')
model.finalize()