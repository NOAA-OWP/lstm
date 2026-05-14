# Basic Model Interface (BMI) for streamflow prediction using Long Short-Term Memory (LSTM) networks
This Long Short-Term Memory (LSTM) network was developed for use in the [Next Generation Water Resources Modeling Framework (NextGen)](https://github.com/NOAA-OWP/ngen). LSTMs are able to provide relatively accurate streamflow predictions when compared to other model types. This module is available through a [Basic Model Interface (BMI)](https://bmi.readthedocs.io/en/latest/).

- [Adaption from NeuralHydrology](#adaption-from-neuralhydrology)
- [Sample Data](#sample-data)
- [Configurations](#configurations)
- [Trained LSTM Model](#trained-lstm-model)
- [Running BMI LSTM](#running-bmi-lstm)
- [Weights and Biases](#weights-and-biases)
- [Trained LSTM Model](#trained-lstm-model)
- [Unit Test](#unit-test)

## Adaption from NeuralHydrology
This module is dependent on a trained deep learning model. The forward pass of this LSTM model [`nextgen_cuda_lstm.py`](./lstm/nextgen_cuda_lstm.py) is heavily based on NeuralHydrology's [`CudaLSTM`](https://neuralhydrology.readthedocs.io/en/latest/usage/models.html#cudalstm). Other model classes can be applied but [`bmi_lstm.py`](./lstm/bmi_lstm.py) would need to load it in. More information about the python package NeuralHydrology can be found [here](https://neuralhydrology.readthedocs.io/en/latest/).  

## Sample Data

### NLDAS sample data
Sample data required for a test run of this model is available in the [`data`](./data) directory. This includes:
* Forcing data: `usgs-streamflow-nldas_hourly.nc`
* Observation values: also included in `usgs-streamflow-nldas_hourly.nc`
* Static attributes: see an example configuration file for a list of these attributes in [`./bmi_config_files`](./bmi_config_files/02064000_nh_NLDAS_hourly.yml) 

for four USGS gauges:
* 02064000 Falling River nr Naruna, VA
* 01547700 Marsh Creek at Blanchard, PA
* 03015500 Brokenstraw Creek at Youngsville, PA
* 01022500 Narraguagus River at Cherryfield, Maine  

Note that the data found in this repository are simply examples. The LSTM model can be run on any watershed, provided the necessary static attributes and dynamic forcings. The full list of attributes differs depending on the trained LSTM model chosen. Example files (`*.yml`) with the required attributes are located in the [`./bmi_config_files`](./bmi_config_files)directory. The attributes required for these configuration files can be found in the [`camels_attributes_v2.0/`](./data/camels_attributes_v2.0) data directory for catchments in the CAMELS dataset or estimated from [Addor, N., A.J. Newman, N. Mizukami, and M.P. Clark. 2017. The CAMELS data set: catchment attributes and meteorology for large-sample studies. Hydrol. Earth Syst. Sci. 21: 5293-5313. https://doi.org/10.5194/hess-21-5293-2017](https://doi.org/10.5194/hess-21-5293-2017).  

### AORC Sample Data
To run a sample with AORC, you can clone this repository that has data from several camples basins: [https://github.com/NWC-CUAHSI-Summer-Institute/CAMELS_data_sample](https://github.com/NWC-CUAHSI-Summer-Institute/CAMELS_data_sample). You'll need to change the paths in the sample AORC notebook.

## Configurations
The LSTM model requires a configuration file for specification of forcings, weights, scalers, run options (like warmup period), runtime period, static basin parameters and model time step. This configuration file needs to be generated for any specific application of the LSTM model.

This LSTM model will run on any basin with the required inputs; however, it was trained on 500+ catchments from the [CAMELS dataset](https://ral.ucar.edu/solutions/products/camels) across the contiguous United States (CONUS) and is best suited to this CONUS region, for now. The place to set up the run for a specific configuration for a specific basin is in the BMI (`*.yml`) [configuration file](./bmi_config_files/). Ideally, the LSTM trained with all forcings and all static attributes will be used, but we've included a few example LSTMs that have limited static attributes and forcings, in the event that the total set of forcings and attributes are not available. For explanations of how the LSTM might perform with limited inputs and on ungauged basins, see [Frederik Kratzert et al., Toward Improved Predictions in Ungauged Basins: Exploiting the Power of Machine Learning, Water Resources Research](https://doi.org/10.1029/2019WR026065). To set up a specific configuration for a specific basin, change the appropriate [BMI configuration file](./bmi_config_files/). 

## Trained LSTM Model
Included in this directory are three samples of trained LSTM models:
* `hourly_all_attributes_and_forcings`: This is the model that should be used. It was trained to ingest 8 atmospheric forcings and 26 static attributes, that were chosen from the [CAMELS dataset](https://ral.ucar.edu/solutions/products/camels). If you do not have access to all these static attributes, one of the models below are available with limited static attributes, but in general would be best to use all data possible.  
* `hourly_slope_mean_precip_temp`: This model was trained to ingest only two atmospheric forcings (total precipitation and temperature) and two static attributes (basin mean slope and elevation).  
* `hourly_all_forcings_lat_lon_elev`: This model was trained to ingest eight atmospheric forcings (total precipitation, longwave radiation, shortwave radiation, pressure, specific humidity, temperature, wind in the X and Y directions) and three static attributes (basin mean elevation, latitude and longitude).  

These three models are trained with different inputs, but they all will run with the same [BMI](./lstm/bmi_lstm.py) and [LSTM](./lstm/nextgen_cuda_lstm.py) model.

## Running BMI LSTM
Instructions for running the LSTM with the BMI interface are available in the INSTALL guide.


## Weights and Biases
The training procedure should produce weights and biases for the LSTM model. These are stored in Pytorch files (`*.pt`), are kept within the training directories: [`trained_neuralhydrology_models`](./trained_neuralhydrology_models). Without these the model can still run, but will not make streamflow predictions. These are **absolutely** necessary for running this model, including coupling, with the NextGen framework. These weights and biases are trained to represent many basins, so they do not change for every basin. The model may be trained regionally, or globally, and the weights and biases need to be consistent across the appropriate basins. In the examples contained within this repository, we trained the models to ingest particular inputs (both static and dynamic), and the weights associated with those models cannot be interchanged.  

## Unit Test
BMI has functions that are used by a framework, or model driver, that allows interaction with models through consistent commands. The unit tests are designed to test those BMI functions (run in these examples from Python commands), to ensure that a framework, or model driver, will get the expected result when a command is called. BMI includes functions for different parts of the modeling chain, including functions to get information from the models (known as `getters`), functions to set information in the models (know as `setters`), functions to setup and run the models, etc. The unit test includes these functions, categorized below:   
- Model control functions (4)
- Model information functions (5)
- Variable information functions (6)
- Time functions (5)
- Variable getter and setter functions (5)
- Model grid functions (16)

The test script [`run_bmi_unit_test.py`](./lstm/run_bmi_unit_test.py) fully examines the functionality of all applicable definitions.

To run lstm-bmi unit test, from the parent directory, simply call `python ./lstm/run_bmi_unit_test.py` within the active conda environment `bmi_lstm`, as outlined in [Running BMI LSTM](#running-bmi-lstm).

Recall that BMI guides interoperability for model-coupling, where model components (i.e. inputs and outputs) are easily shared amongst each other. When testing outside of a true framework, we consider the behavior of BMI function definitions, rather than any expected values they produce.
