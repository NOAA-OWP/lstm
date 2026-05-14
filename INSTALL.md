# Installation instructions

## Dependencies
All Python dependencies are listed in [pyproject.toml](./pyproject.toml). LSTM-BMI is compatible with Windows, Linux, and Mac systems.


## Build Python Virtual Environment for LSTM
Running this model requires a few Python libraries with specific versions listed in the [file](./pyproject.toml). Use the following instructions to build a virtual Python environment
- mkdir ~/.bmi_lstm
- python -m venv ~/.bmi_lstm
- source ~/.bmi_lstm/bin/activate
- cd <path_to_lstm_dir>
- pip install -e .

## Running BMI LSTM
This section goes through an example of running the LSTM with the BMI interface. These are only examples. If a user wants to run the LSTM with BMI, then these are a jumping off point. These examples were developed to provide a quick testing ground for running the LSTM with the [NextGen framework](https://github.com/NOAA-OWP/ngen).  See the [`doc/`](./doc) folder for more information regarding running this module within `NextGen` as well as the `ngen_files/README.txt` found [here](./ngen_files.README.txt).

Note that this code assumes the use of the `bmi_lstm` environment for Anaconda. To load this environment, enter `conda activate bmi_lstm`.  Install the library, `pip install lstm` and execute `python -m lstm`.  See [PACKAGE.md](./PACKAGE.md) for more information about running lstm as a Python library. 

Be aware that these scripts are examples and may require changes for your use case. For example, the Python script was developed for the trained LSTM model with limited attributes (`hourly_slope_mean_precip_temp`) and the for loop will need to be changed if running with the LSTM model that was trained with all attributes (an example of this code can be found in the [Jupyter Notebook](./notebooks/run_lstm_with_bmi.ipynb).

Running these examples of trained LSTM-based hydrological models require these general steps:  
1.  Retrieve atmospheric forcing data that match those included in the trained model
2.  Retrieve the catchment attributes that match those included in the trained model
3.  Create a configuration file with the key-value pairs that can be used by the BMI
4.  Run a script with the Python commands for the BMI model control functions

The [Jupyter Notebook](./notebooks/run_lstm_with_bmi.ipynb) and a Python script [`run_lstm_bmi.py`](./lstm/lstm/run_lstm_bmi.py) have an example of running the LSTM with BMI model control functions, which can be summarized as follows:    

1. `conda activate bmi_lstm`
2. Import required libraries (e.g., `import torch`)
3. Load in the model from the BMI file: `model = lstm.bmi_LSTM()`
4. Read in the configuration file, and this includes the model weights, etc.: `model.read_cfg_file()`
5. Now start running the BMI functions, starting with initialize: `model.initialize()`
6. The model is now available to run either one timestep at a time: `model.update()`, or many timesteps at a time: `model.update_until(model.iend)`, where model.iend is the end of the forcing file, but this can be any value less than or equal to the end of the forcing file.
7. And finally, you should finalize the model instance: `model.finalize()`  

This repository contains an example file with weather and observed streamflow data for four catchments [here](./data/usgs-streamflow-nldas_hourly.nc). Note that the observed streamflow data isn’t necessary to run the model, but is useful for comparison purposes.

Also contained within this repository are catchment attributes for all CAMELS catchments along with two example configuration files: one for the limited data case and one for the full set of attributes.   

To run the LSTM model for another catchment, slight modifications to this code will be needed:
1.  The configuration file path when setting the `model.initialize(bmi_cfg_file='./path/to/your/config/file.yml')` function
2.  Streamflow and weather data path when defining `sample_data`. These examples shown here are stored in a NetCDF file, but the user is free to store and read the data for their use case however they please.  
3.  Check how the streamflow and weather variables are defined/passed into the model as there could be variations in headers, etc. in your data file – These are defined in a for loop.  
