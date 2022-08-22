# Basic Model Interface (BMI) for streamflow prediction using Long Short-Term Memory (LSTM) networks
This Long Short-Term Memory (LSTM) network was developed for use in the [Next Generation Water Resources Modeling Framework (Nextgen)](https://github.com/NOAA-OWP/ngen). LSTMs are able to provide relatively accurate streamflow predictions when compared to other model types. This module is available through a [Basic Model Interface (BMI)](https://bmi.readthedocs.io/en/latest/).

- [Adaption from NeuralHydrology](#adaption-from-neuralhydrology)
- [Sample Data](#sample-data)
- [Configurations](#configurations)
- [Trained LSTM Model](#trained-lstm-model)
- [Dependencies](#dependencies)
- [Running BMI LSTM](#running-bmi-lstm)
- [Weights and Biases](#weights-and-biases)
- [Trained LSTM Model](#trained-lstm-model)
- [Unit Test](#unit-test)

## Adaption from NeuralHydrology
This module is dependent on a trained deep learning model. The forward pass of this LSTM model [`nextgen_cuda_lstm.py`](./src/nextgen_cuda_lstm.py) is heavily based on NeuralHydrology's [`CudaLSTM`](https://neuralhydrology.readthedocs.io/en/latest/usage/models.html#cudalstm). Other model classes can be applied but [`bmi_lstm.py`](./src/bmi_lstm.py) would need to load it in. More information about the python package NeuralHydrology can be found [here](https://neuralhydrology.readthedocs.io/en/latest/).  

## Sample Data
All data required for a test run of this model is available in the [`data/`](./data) directory. This includes:
* Forcing data: `usgs-streamflow-nldas_hourly.nc`
* Observation values: also included in `usgs-streamflow-nldas_hourly.nc`
* Static attributes: see an example configuration file for a list of these attributes in [`./bmi_config_files`](./bmi_config_files/01022500_hourly_all_attributes_forcings.yml) 

for four USGS gauges:
* 02064000 Falling River nr Naruna, VA
* 01547700 Marsh Creek at Blanchard, PA
* 03015500 Brokenstraw Creek at Youngsville, PA
* 01022500 Narraguagus River at Cherryfield, Maine  

Note that the data found in this repository are simply examples. The LSTM model can be run on any watershed, provided the necessary static attributes and dynamic forcings. The full list of attributes differs depending on the trained LSTM model chosen. Example files (`*.yml`) with the required attributes are located in the [`./bmi_config_files`](./bmi_config_files)directory. The attributes required for these configuration files can be found in the [`camels_attributes_v2.0/`](./data/camels_attributes_v2.0) data directory for catchments in the CAMELS dataset or estimated from [Addor, N., A.J. Newman, N. Mizukami, and M.P. Clark. 2017. The CAMELS data set: catchment attributes and meteorology for large-sample studies. Hydrol. Earth Syst. Sci. 21: 5293-5313. https://doi.org/10.5194/hess-21-5293-2017](https://doi.org/10.5194/hess-21-5293-2017).  

## Configurations
The LSTM model requires a configuration file for specification of forcings, weights, scalers, run options (like warmup period), runtime period, static basin parameters and model time step. This configuration file needs to be generated for any specific application of the LSTM model.

This LSTM model will run on any basin with the required inputs; however, it was trained on 500+ catchments from the CAMELS dataset [CAMELS dataset](https://ral.ucar.edu/solutions/products/camels) across the contiguous United States (CONUS) and is best suited to this CONUS region, for now. The place to set up the run for a specific configuration for a specific basin is in the BMI (`*.yml`) [configuration file](./bmi_config_files/). Ideally, the LSTM trained with all forcings and all static attributes will be used, but we've included a few example LSTMs that have limited static attributes and forcings, in the event that the total set of forcings and attributes are not available. For explanations of how the LSTM might perform with limited inputs and on ungauged basins, see [Frederik Kratzert et al., Toward Improved Predictions in Ungauged Basins: Exploiting the Power of Machine Learning, Water Resources Research](https://doi.org/10.1029/2019WR026065). To set up a specific configuration for a specific basin, change the appropriate [BMI configuration file](./bmi_config_files/). 

## Trained LSTM Model
Included in this directory are three samples of trained LSTM models:
* `hourly_all_attributes_and_forcings`: This is the model that should be used. It was trained to ingest 8 atmospheric forcings and 26 static attributes, that were chosen from the [CAMELS dataset](https://ral.ucar.edu/solutions/products/camels). If you do not have access to all these static attributes, one of the models below are available with limited static attributes, but in general would be best to use all data possible.  
* `hourly_slope_mean_precip_temp`: This model was trained to ingest only two atmospheric forcings (total precipitation and temperature) and two static attributes (basin mean slope and elevation).  
* `hourly_all_forcings_lat_lon_elev`: This model was trained to ingest eight atmospheric forcings (total precipitation, longwave radiation, shortwave radiation, pressure, specific humidity, temperature, wind in the X and Y directions) and three static attributes (basin mean elevation, latitude and longitude).  

These three models are trained with different inputs, but they all will run with the same [BMI](./src/bmi_lstm.py) and [LSTM](./src/nextgen_cuda_lstm.py) model.

## Dependencies
Running this model requires python and the libraries listed in the [environment file](./environment.yml). This example uses [Anaconda](https://www.anaconda.com), but it isn’t a requirement. You can opt to set up a python environment without it by using the libraries specified in the `environment.yml` file. If you have Anaconda, you can easily create an environment (`bmi_lstm`) with the required libraries using:  `conda env create -f environment.yml`. 

Notice that `xarray` has a specific version defined in the environment file (0.14.0) as the newer versions are incompatible with the current example files. The same goes for `llvm-openmp`, which we set to version 10.0.0 in the dependencies. On some Mac Anaconda releases, users received an error message stating `OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.` If you get this message, please make sure you have ` - llvm-openmp=10.0.0` set in your environment.yml file. More information on different solutions to resolving this issue can be found [here](https://stackoverflow.com/questions/62903775/intel-mkl-error-using-conda-and-matplotlib-library-not-loaded-rpath-libiomp5) and [here](https://stackoverflow.com/questions/62903775/intel-mkl-error-using-conda-and-matplotlib-library-not-loaded-rpath-libiomp5). 

If at any point you want to see the full list of the packages and dependencies in your activated `bmi_lstm` environment, run `conda env export > environment_<rename>.yml` replacing `<rename>` with your text of choice to avoid overwriting the original `environment.yml` file.

## Running BMI LSTM
This section goes through an example of running the LSTM with the BMI interface. These are only examples. If a user wants to run the LSTM with BMI, then these are a jumping off point. These examples were developed to provide a quick testing ground for running the LSTM with the [Nextgen framework](https://github.com/NOAA-OWP/ngen).  

Note that this code assumes the use of the `bmi_lstm` environment for Anaconda. To load this environment, enter `conda activate bmi_lstm`.   

Be aware that these scripts are examples and may require changes for your use case. For example, the Python script was developed for the trained LSTM model with limited attributes (`hourly_slope_mean_precip_temp`) and the for loop will need to be changed if running with the LSTM model that was trained with all attributes (an example of this code can be found in the [Jupyter Notebook](./notebooks/run_lstm_bmi.ipynb).

Running these examples of trained LSTM-based hydrological models require these general steps:  
1.  Retrieve atmospheric forcing data that match those included in the trained model
2.  Retrieve the catchment attributes that match those included in the trained model
3.  Create a configuration file with the key-value pairs that can be used by the BMI
4.  Run a script with the Python commands for the BMI model control functions

The [Jupyter Notebook](./notebooks/run_lstm_bmi.ipynb) and a Python script [`run_lstm_bmi.py`](./src/run_lstm_bmi.py) have an example of running the LSTM with BMI model control functions, which can be summarized as follows:    

1. `conda activate bmi_lstm`
2. Import required libraries (e.g., `import torch`)
3. Load in the model from the BMI file: `model = lstm.bmi_LSTM()`
4. Read in the configuration file, and this includes the model weights, etc.: `model.read_cfg_file()`
5. Now start running the BMI functions, starting with initialize: `model.initialize()`
6. The model is now available to run either one timestep at a time: `model.update()`, or many timesteps at a time: `model.update_until(model.iend)`, where model.iend is the end of the forcing file, but this can be any value less than or equal to the end of the forcing file.
7. And finally you should finalize the model instance: `model.finalize()`  

This repository contains an example file with weather and observed streamflow data for four catchments [here](./data/usgs-streamflow-nldas_hourly.nc). Note that the observed streamflow data isn’t necessary to run the model, but is useful for comparison purposes.

Also contained within this repository are catchment attributes for all CAMELS catchments along with two example configuration files: one for the limited data case and one for the full set of attributes.   

To run the LSTM model for another catchment, slight modifications to this code will be needed:
1.  The configuration file path when setting the `model.initialize(bmi_cfg_file='./path/to/your/config/file.yml')` function
2.  Streamflow and weather data path when defining `sample_data`. These examples shown here are stored in a NetCDF file, but the user is free to store and read the data for their use case however they please.  
3.  Check how the streamflow and weather variables are defined/passed into the model as there could be variations in headers, etc. in your data file – These are defined in a for loop.  


## Weights and Biases
The training procedure should produce weights and biases for the LSTM model. These are stored in Pytorch files (`*.pt`), are kept within the training directories: [`trained_neuralhydrology_models`](./trained_neuralhydrology_models). Without these the model can still run, but will not make streamflow predictions. These are **absolutely** necessary for running this model, including coupling, with the Nextgen framework. These weights and biases are trained to represent many basins, so they do not change for every basin. The model may be trained regionally, or globally, and the weights and biases need to be consistent across the appropriate basins. In the examples contained within this repository, we trained the models to ingest particular inputs (both static and dynamic), and the weights associated with those models cannot be interchanged.  

## Unit Test
BMI has functions that are used by a framework, or model driver, that allows interaction with models through consistent commands. The unit tests are designed to test those BMI functions (run in these examples from Python commands), to ensure that a framework, or model driver, will get the expected result when a command is called. BMI includes functions for different parts of the modeling chain, including functions to get information from the models (known as `getters`), functions to set information in the models (know as `setters`), functions to setup and run the models, etc. The unit test includes these functions, categorized below:   
- Model control functions (4)
- Model information functions (5)
- Variable information functions (6)
- Time functions (5)
- Variable getter and setter functions (5)
- Model grid functions (16)

The test script [`run_bmi_unit_test.py`](./src/run_bmi_unit_test.py) fully examines the functionality of all applicable definitions.

To run lstm-bmi unit test, from the `/src` directory, simply call `python ./run_bmi_unit_test.py` within the active conda environment `bmi_lstm`, as outlined in [Running BMI LSTM](#running-bmi-lstm).

Recall that BMI guides interoperability for model-coupling, where model components (i.e. inputs and outputs) are easily shared amongst each other. When testing outside of a true framework, we consider the behavior of BMI function definitions, rather than any expected values they produce.