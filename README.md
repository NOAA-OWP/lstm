# Basic Model Interface (BMI) for streamflow prediction using Long Short-Term Memory (LSTM) networks
This LSTM is for use in the Next Generation National Water Model. LSTMs have been shown to be very good deep learning models for streamflow prediction. This module is available through a BMI interface that is build directly into the deep learning model class. That means this LSTM is inherently BMI enabled.

# Adaption from NeuralHydrology
This module is dependent on a trained deep learning model. The forward pass of this LSTM model (`nextgen_cuda_lstm.py`) is based on that of CudaLSTM in NeuralHydrology, but in principal can be used with any LSTM, so long as the `bmi_lstm.py` loads it in. An out-of-date version of this BMI LSTM model is also included (But will be deleted soon): `bmi_lstm_nwm3_aorc.py`.

# Data requirements
All data required for a test run of this model is available in the `./data/` directory. 
* Data for the most recent version of this BMI LSTM code are stored in a NetCDF file: `./data/sample_data/usgs-streamflow-nldas_hourly.nc`
Data required for the out-of-date model are:
* Forcing data for test period: **forcing-cat-87.txt**
* Forcing data for warmup period: **35.313-80.813.nc**
* Trained model weights: **nwmv3_normalarea_trained.pt**
* Scalers (mean and stdev. from training period): **nwmv3_normalarea_scaler.p**
* Observation values from a nearby gauge: **obs_q_02146562.csv**  

Training data is available through many sources. LSTM models are often trained on the CAMELS dataset, and those data are found on the [NCAR Wedsite](https://ral.ucar.edu/solutions/products/camels).  
Warmup data for this model was downloaded from [LDAS](https://ldas.gsfc.nasa.gov/nldas/v2/forcing).  
Static catchment attributes (longitude/latitude and elevation) was collected from Google Earth.  
Forcing data for the test period (December 2015) was provided by the NGen Framework team.  

# How to run this model
Running this model requires python and the following libraries:
* bmipy
* bottleneck
* configparser (will be removed with old model)
* data_tools (will be removed with old model)
* netcdf4 
* numpy
* pandas
* pathlib
* pickle
* Pytorch
* time
* xarray 

This LSTM model will run on any basin with the required inputs.
The first step to running the LSTM is to make sure a python environment containing pytorch, and a few others (listed above), is installed and activated. The easiest way to do that is to download the environment file from [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology/tree/master/environments)  
With the environment available, use Anaconda to [install the environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment): `conda env create -f environment_cpu.yml`. If you can load in the environments without Anaconda, that should be just fine.
The Jupyter Notebook `run-lstm-with-bmi.ipynb` and a Python script `run-lstm-with-bmi.py` have an example of running the model. The basic steps are:
0. `conda activate pytorch_environment`
1. Import required libraries (e.g., `import torch`)
2. Load in the model from the BMI file: `model = lstm.bmi_LSTM()`
3. Read in the configuration file, and this includes the model weights, etc.: `model.read_cfg_file()`
4. Now start running the BMI functions, starting with initialize: `model.initialize()`
5. The model is now available to run either one timestep at a time: `model.update()`, or many timesteps at a time: `model.update_until(model.iend)`, where model.iend is the end of the forcing file, but this can be any value less than or equal to the end of the forcing file.
6. And finally you should finalize the model instance: `model.finalize()`

# Model weights and biases
The training procedure should produce weights and biases for the LSTM model. Without these the model can still run, but will not make streamflow predictions. These are **absolutely** neccessary for running this model with NextGen. These weights and biases are trained to represent many basins, so they do not change for every basins. The model may be trained regionally, or globally, and the weights and biases need to be consistent across the appropriate basins.

# Model configuration
The LSTM model requires a configuration file for specification of forcings, weights, scalers, run options (like warmup period), run time period, static basin parameters and model time step. This configuration file needs to be generated for any specific application of the LSTM model.

# More information on NeuralHydrology
https://neuralhydrology.readthedocs.io/en/latest/index.html