
import numpy as np
import torch
# from torch import nn   # this is not used
# import data_tools
from pathlib import Path
from netCDF4 import Dataset

# This is the BMI LSTM that we will be running
############  import lstm.bmi_lstm as bmi_lstm

USE_PATH = True  # (SDP; also set in bmi_lstm.py.)
run_dir = './extern/lstm_py/'  # (SDP)
# run_dir = './'
cfg_file  = run_dir + 'bmi_config_files/01022500_A.yml'
data_file = run_dir + 'data/usgs-streamflow-nldas_hourly.nc'
    
def execute():
    # creating an instance of an LSTM model
    print('Creating an instance of an BMI_LSTM model object...')
    model = bmi_LSTM()
    ### model = bmi_lstm.bmi_LSTM()
    ### model = bmi_lstm.bmi_lstm()

    # Initializing the BMI
    print('Initializing the BMI...')
    # Argument to initialize should be type string, not Path object. (SDP)
    # Better to use path inside initialize(). (SDP)
    ### model.initialize(bmi_cfg_file=Path('./bmi_config_files/01022500_A.yml'))
    model.initialize(bmi_cfg_file=cfg_file)  # (SDP)

    # Get input data that matches the LSTM test runs
    print('Get input data that matches the LSTM test runs...')

    if (USE_PATH):
        sample_data = Dataset(Path( data_file ), 'r')
    else:
        sample_data = Dataset( data_file, 'r' )  # SDP

        
    # Now loop through the inputs, set the forcing values, and update the model
    print('Loop through the inputs, set the forcing values, and update the model...')
    for precip, temp in zip(list(sample_data['total_precipitation'][3].data),
                            list(sample_data['temperature'][3].data)):
        model.set_value('atmosphere_water__liquid_equivalent_precipitation_rate',precip)
        model.set_value('land_surface_air__temperature',temp)
        #### print('  temperature and precipitation are set to {:.2f} and {:.2f}'.format(model.temperature, model.total_precipitation))
        print('  temperature and precipitation are set to {:.2f} and {:.2f}'.format(model.temperature, model.precip))
        model.update()
        print('  streamflow (CMS) at time {} is {:.2f}'.format(model.t, model.streamflow_cms))
        #### print('  streamflow (CFS) at time {} is {:.2f}'.format(model.t, model.streamflow_cfs))

        if model.t > 100:
            print('Stopping the loop...')
            break

    # Finalizing the BMI
    print('Finalizing the BMI...')
    model.finalize()
    print('Finished.')
    
