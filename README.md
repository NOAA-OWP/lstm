# Basic Model Interface (BMI) for streamflow prediction using Long Short-Term Memory (LSTM) networks
This LSTM is for use in the Next Generation National Water Model. LSTMs have been shown to be very good deep learning models for streamflow prediction. This module is available through a BMI interface that is built directly into the deep learning model class. That means this LSTM is inherently BMI enabled.  

# Adaption from NeuralHydrology
This module is dependent on a trained deep learning model. The forward pass of this LSTM model (`nextgen_cuda_lstm.py`) is based on that of CudaLSTM in NeuralHydrology, but in principle can be used with any LSTM, so long as the `bmi_lstm.py` loads it in.  

# Data requirements
All data required for a test run of this model is available in `./data/sample_data/usgs-streamflow-nldas_hourly.nc`. This includes:
* Forcing data
* Observation values  

for Four USGS gauges:
* 02064000 Falling River nr Naruna, VA
* 01547700 Marsh Creek at Blanchard, PA
* 03015500 Brokenstraw Creek at Youngsville, PA
* 01022500 Narraguagus River at Cherryfield, Maine  

These are just samples, you can run on any watershed.   

# Model configuration
The LSTM model requires a configuration file for specification of forcings, weights, scalers, run options (like warmup period), run time period, static basin parameters and model time step. This configuration file needs to be generated for any specific application of the LSTM model.
This LSTM model will run on any basin with the required inputs. The place to set up the run for a specific configuration for a specific basin is in the BMI configuration file: `./bmi_config_files/*.yml`.

# Trained LSTM model
Included in this directory are two samples of trained LSTM models:
* hourly_A
* hourly_B  

These two models are trained with slightly different inputs (Basin average slope is included in only one), and with different number of hidden layers.

# System requirements to run this model
Running this model requires python and the libraries listed in the environment file: `environment.yml`.  
If you can load in the environments without Anaconda, that should be just fine. Notice that `xarray` has a specific version defined in the environment file (0.14.0). The newer versions are incompatible withe the example files, for some reason that will be fixed at some point, so for now, make sure to use this specific version of Xarray.  
It may be easiest to make sure a python environment containing pytorch, and the others listed above, are installed and activated with Anaconda using: `conda env create -f environment.yml`. This gives you a conda environment called `bmi_lstm`.  

# Running the model
The Jupyter Notebook `run-lstm-with-bmi.ipynb` and a Python script `run-lstm-with-bmi.py` have an example of running the model. The basic steps are:
0. `conda activate bmi_lstm`
1. Import required libraries (e.g., `import torch`)
2. Load in the model from the BMI file: `model = lstm.bmi_LSTM()`
3. Read in the configuration file, and this includes the model weights, etc.: `model.read_cfg_file()`
4. Now start running the BMI functions, starting with initialize: `model.initialize()`
5. The model is now available to run either one timestep at a time: `model.update()`, or many timesteps at a time: `model.update_until(model.iend)`, where model.iend is the end of the forcing file, but this can be any value less than or equal to the end of the forcing file.
6. And finally you should finalize the model instance: `model.finalize()`

# Model weights and biases
The training procedure should produce weights and biases for the LSTM model. These are stored in Pytorch files (*.pt), are kept within the training directories: `trained_neuralhydrology_models`. Without these the model can still run, but will not make streamflow predictions. These are **absolutely** necessary for running this model with NextGen. These weights and biases are trained to represent many basins, so they do not change for every basin. The model may be trained regionally, or globally, and the weights and biases need to be consistent across the appropriate basins.

# More information on training models with NeuralHydrology
https://neuralhydrology.readthedocs.io/en/latest/index.html