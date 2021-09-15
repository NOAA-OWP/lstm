# Need these for BMI
from bmipy import Bmi
import time
#import data_tools
# Basic utilities
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
# Here is the LSTM model we want to run
import nextgen_cuda_lstm
# Configuration file functionality
import yaml
# LSTM here is based on PyTorch
import torch
from torch import nn

class bmi_LSTM(Bmi):

    def __init__(self):
        """Create a Bmi LSTM model that is ready for initialization."""
        super(bmi_LSTM, self).__init__()
        self._values = {}
        self._var_units = {}
        self._var_loc = {}
        self._start_time = 0.0
        self._end_time = np.finfo("d").max
        self._time_units = "s"

    #----------------------------------------------
    # Required, static attributes of the model
    #----------------------------------------------
    _att_map = {
        'model_name':         'LSTM for Next Generation NWM',
        'version':            '1.0',
        'author_name':        'Jonathan Martin Frame',
        'grid_type':          'none',
        'time_step_type':     'donno',
        'step_method':        'none',
        'time_units':         '1 hour' }

    #---------------------------------------------
    # Input variable names (CSDMS standard names)
    #---------------------------------------------
    _input_var_names = [
        'land_surface_radiation~incoming~longwave__energy_flux',
        'land_surface_air__pressure',
        'atmosphere_air_water~vapor__relative_saturation',
        'atmosphere_water__liquid_equivalent_precipitation_rate',
        'land_surface_radiation~incoming~shortwave__energy_flux',
        'land_surface_air__temperature',
        'land_surface_wind__x_component_of_velocity',
        'land_surface_wind__y_component_of_velocity']

    #---------------------------------------------
    # Output variable names (CSDMS standard names)
    #---------------------------------------------
    _output_var_names = ['land_surface_water__runoff_volume_flux']

    #------------------------------------------------------
    # Create a Python dictionary that maps CSDMS Standard
    # Names to the model's internal variable names.
    # This is going to get long, 
    #     since the input variable names could come from any forcing...
    #------------------------------------------------------
    _var_name_map_long_first = {
                                'land_surface_water__runoff_volume_flux':'land_surface_water__runoff_volume_flux',
                                #--------------   Dynamic inputs --------------------------------
                                'atmosphere_water__time_integral_of_precipitation_mass_flux':'total_precipitation',
                                'land_surface_radiation~incoming~longwave__energy_flux':'longwave_radiation',
                                'land_surface_radiation~incoming~shortwave__energy_flux':'shortwave_radiation',
                                'atmosphere_air_water~vapor__relative_saturation':'specific_humidity',
                                'land_surface_air__pressure':'pressure',
                                'land_surface_air__temperature':'temperature',
                                'land_surface_wind__x_component_of_velocity':'wind_u',
                                'land_surface_wind__y_component_of_velocity':'wind_v',
                                #--------------   STATIC Attributes -----------------------------
                                'basin__area':'area_gages2',
                                'ratio__mean_potential_evapotranspiration__mean_precipitation':'aridity',
                                'basin__carbonate_rocks_area_fraction':'carbonate_rocks_frac',
                                'soil_clay__volume_fraction':'clay_frac',
                                'basin__mean_of_elevation':'elev_mean',
                                'land_vegetation__forest_area_fraction':'frac_forest',
                                'atmosphere_water__precipitation_falling_as_snow_fraction':'frac_snow',
                                'bedrock__permeability':'geol_permeability',
                                'land_vegetation__max_monthly_mean_of_green_vegetation_fraction':'gvf_max',
                                'land_vegetation__diff__max_min_monthly_mean_of_green_vegetation_fraction':'gvf_diff',
                                'atmospher_water__mean_duration_of_high_precipitation_events':'high_prec_dur',
                                'atmospher_water__frequency_of_high_precipitation_events':'high_prec_freq',
                                'land_vegetation__diff_max_min_monthly_mean_of_leaf-area_index':'lai_diff',
                                'land_vegetation__max_monthly_mean_of_leaf-area_index':'lai_max',
                                'atmosphere_water__low_precipitation_duration':'low_prec_dur',
                                'atmosphere_water__precipitation_frequency':'low_prec_freq',
                                'maximum_water_content':'max_water_content',
                                'atmospher_water__daily_mean_of_liquid_equivalent_precipitation_rate':'p_mean',
                                'land_surface_water__daily_mean_of_potential_evaporation_flux':'pet_mean',
                                'basin__mean_of_slope':'slope_mean',
                                'soil__saturated_hydraulic_conductivity':'soil_conductivity',
                                'soil_bedrock_top__depth__pelletier':'soil_depth_pelletier',
                                'soil_bedrock_top__depth__statsgo':'soil_depth_statsgo',
                                'soil__porosity':'soil_porosity',
                                'soil_sand__volume_fraction':'sand_frac',
                                'soil_silt__volume_fraction':'silt_frac'
                                 }

    #------------------------------------------------------
    # Create a Python dictionary that maps CSDMS Standard
    # Names to the units of each model variable.
    #------------------------------------------------------
    _var_units_map = {
        'land_surface_water__runoff_volume_flux':'mm',
        #--------------------------------------------------
         'land_surface_radiation~incoming~longwave__energy_flux':'W m-2',
         'land_surface_air__pressure':'Pa',
         'atmosphere_air_water~vapor__relative_saturation':'kg kg-1',
         'atmosphere_water__liquid_equivalent_precipitation_rate':'kg m-2',
         'land_surface_radiation~incoming~shortwave__energy_flux':'W m-2',
         'land_surface_air__temperature':'K',
         'land_surface_wind__x_component_of_velocity':'m s-1',
         'land_surface_wind__y_component_of_velocity':'m s-1'}


    #------------------------------------------------------
    # A list of static attributes. Not all these need to be used.
    #------------------------------------------------------
    #   These attributes can be anaything, but usually come from the CAMELS attributes:
    #   Nans Addor Andrew J. Newman, Naoki Mizukami, and Martyn P. Clark
    #   The CAMELS data set: catchment attributes and meteorology for large-sample studies
    #   https://doi.org/10.5194/hess-21-5293-2017
    _static_attributes_list = ['area_gages2','aridity','carbonate_rocks_frac','clay_frac',
                               'elev_mean','frac_forest','frac_snow','geol_permeability',
                               'gvf_max','gvf_diff','high_prec_dur','high_prec_freq','lai_diff',
                               'lai_max','low_prec_dur','low_prec_freq','max_water_content',
                               'p_mean','pet_mean','slope_mean','soil_conductivity',
                               'soil_depth_pelletier','soil_depth_statsgo','soil_porosity',
                               'sand_frac','silt_frac']

    #------------------------------------------------------------
    #------------------------------------------------------------
    # BMI: Model Control Functions
    #------------------------------------------------------------ 
    #------------------------------------------------------------

    #-------------------------------------------------------------------
    def initialize( self, bmi_cfg_file=None ):
        
        # -------------- Read in the BMI configuration -------------------------#
        # This will direct all the next moves.
        if bmi_cfg_file is not None:

            with bmi_cfg_file.open('r') as fp:
                cfg = yaml.safe_load(fp)
            self.cfg_bmi = self._parse_config(cfg)
        else:
            print("Error: No configuration provided, nothing to do...")
    
        self._var_name_map_short_first = {self._var_name_map_long_first[long_name]:long_name for long_name in self._var_name_map_long_first.keys()}
        
        # ------------- Load in the configuration file for the specific LSTM --#
        # This will include all the details about how the model was trained
        # Inputs, outputs, hyper-parameters, scalers, weights, etc. etc.
        self.get_training_configurations()
        self.get_scaler_values()
        
        # ------------- Initialize an LSTM model ------------------------------#
        self.lstm = nextgen_cuda_lstm.Nextgen_CudaLSTM(input_size=self.input_size, 
                                                       hidden_layer_size=self.hidden_layer_size, 
                                                       output_size=self.output_size, 
                                                       batch_size=1, 
                                                       seq_length=1)

        # ------------ Load in the trained weights ----------------------------#
        # Save the default model weights. We need to make sure we have the same keys.
        default_state_dict = self.lstm.state_dict()

        # Trained model weights from Neuralhydrology.
        trained_model_file = self.cfg_train['run_dir'] / 'model_epoch{}.pt'.format(str(self.cfg_train['epochs']).zfill(3))
        trained_state_dict = torch.load(trained_model_file, map_location=torch.device('cpu'))

        # Changing the name of the head weights, since different in NH
        trained_state_dict['head.weight'] = trained_state_dict.pop('head.net.0.weight')
        trained_state_dict['head.bias'] = trained_state_dict.pop('head.net.0.bias')
        trained_state_dict = {x:trained_state_dict[x] for x in default_state_dict.keys()}

        # Load in the trained weights.
        self.lstm.load_state_dict(trained_state_dict)

        # ------------- Initialize the values for the input to the LSTM  -----#
        self.set_static_attributes()
        self.initialize_forcings()
        self.set_values_dictionary()
        
        if self.cfg_bmi['initial_state'] == 'zero':
            self.h_t = torch.zeros(1, self.batch_size, self.hidden_layer_size).float()
            self.c_t = torch.zeros(1, self.batch_size, self.hidden_layer_size).float()

        self.t = 0

        # ----------- The output is area normalized, this is needed to un-normalize it
        #                         mm->m                             km2 -> m2          hour->s    
        self.output_factor_cms =  (1/1000) * (self.cfg_bmi['area_sqkm'] * 1000*1000) * (1/3600)

    #------------------------------------------------------------ 
    def update(self):
        with torch.no_grad():

            self.create_scaled_input_tensor()

            self.lstm_output, self.h_t, self.c_t = self.lstm.forward(self.input_tensor, self.h_t, self.c_t)
            
            self.scale_output()
            
            self.t += 1
    
    #------------------------------------------------------------ 
    def update_until(self, last_update):
        first_update=self.t
        for t in range(first_update, last_update):
            self.update()
    #------------------------------------------------------------    
    def finalize( self ):
        """Finalize model."""
        self._model = None
    
    #------------------------------------------------------------
    #------------------------------------------------------------
    # LSTM: SETUP Functions
    #------------------------------------------------------------
    #------------------------------------------------------------

    #-------------------------------------------------------------------
    def get_training_configurations(self):
        if self.cfg_bmi['train_cfg_file'] is not None:
            with self.cfg_bmi['train_cfg_file'].open('r') as fp:
                cfg = yaml.safe_load(fp)
            self.cfg_train = self._parse_config(cfg)

        # Collect the LSTM model architecture details from the configuration file
        self.input_size        = len(self.cfg_train['dynamic_inputs']) + len(self.cfg_train['static_attributes'])
        self.hidden_layer_size = self.cfg_train['hidden_size']
        self.output_size       = len(self.cfg_train['target_variables']) 

        # WARNING: This implimentation of the LSTM can only handle a batch size of 1
        self.batch_size        = 1 #self.cfg_train['batch_size']

        # Including a list of the model input names.
        self.all_lstm_inputs = []
        self.all_lstm_inputs.extend(self.cfg_train['dynamic_inputs'])
        self.all_lstm_inputs.extend(self.cfg_train['static_attributes'])
        self.all_lstm_input_values_dict = {x:0 for x in self.all_lstm_inputs} # sometimes we need to reference values with strings
        
        # Scaler data from the training set. This is used to normalize the data (input and output).
        with open(self.cfg_train['run_dir'] / 'train_data' / 'train_data_scaler.p', 'rb') as fb:
            self.train_data_scaler = pickle.load(fb)

    #------------------------------------------------------------ 
    def get_scaler_values(self):

        """Mean and standard deviation for the inputs and LSTM outputs""" 

        self.out_mean = self.train_data_scaler['xarray_feature_center'][self.cfg_train['target_variables'][0]].values
        self.out_std = self.train_data_scaler['xarray_feature_scale'][self.cfg_train['target_variables'][0]].values

        self.input_mean = []
        self.input_mean.extend([self.train_data_scaler['xarray_feature_center'][x].values for x in self.cfg_train['dynamic_inputs']])
        self.input_mean.extend([self.train_data_scaler['attribute_means'][x] for x in self.cfg_train['static_attributes']])
        self.input_mean = np.array(self.input_mean)

        self.input_std = []
        self.input_std.extend([self.train_data_scaler['xarray_feature_scale'][x].values for x in self.cfg_train['dynamic_inputs']])
        self.input_std.extend([self.train_data_scaler['attribute_stds'][x] for x in self.cfg_train['static_attributes']]) 
        self.input_std = np.array(self.input_std)

    #------------------------------------------------------------ 
    def create_scaled_input_tensor(self):
        self.set_values_dictionary()
        self.input_array = np.array([self._values[self._var_name_map_short_first[x]] for x in self.all_lstm_inputs])
        self.input_array_scaled = (self.input_array - self.input_mean) / self.input_std 
        self.input_tensor = torch.tensor(self.input_array_scaled)
        
    #------------------------------------------------------------ 
    def scale_output(self):
        if self.cfg_train['target_variables'][0] == 'qobs_mm_per_hour':
            self.surface_runoff_mm = (self.lstm_output[0,0,0].numpy().tolist() * self.out_std + self.out_mean)
        elif self.cfg_train['target_variables'][0] == 'QObs(mm/d)':
            self.surface_runoff_mm = (self.lstm_output[0,0,0].numpy().tolist() * self.out_std + self.out_mean) * (1/24)
        self.streamflow_cms = self.surface_runoff_mm * self.output_factor_cms
        self.streamflow_cfs = self.streamflow_cms * (1/35.314)

    #-------------------------------------------------------------------
    def read_initial_states(self):
        h_t = np.genfromtxt(self.h_t_init_file, skip_header=1, delimiter=",")[:,1]
        self.h_t = torch.tensor(h_t).view(1,1,-1)
        c_t = np.genfromtxt(self.c_t_init_file, skip_header=1, delimiter=",")[:,1]
        self.c_t = torch.tensor(c_t).view(1,1,-1)

    #---------------------------------------------------------------------------- 
    def set_static_attributes(self):
        """ Get the static attributes from the configuration file
        """
        for attribute in self._static_attributes_list:
            if attribute in self.cfg_train['static_attributes']:
                
                # This is probably the right way to do it,
                setattr(self, attribute, self.cfg_bmi[attribute])
                
                # and this is just in case.
                self.all_lstm_input_values_dict[attribute] = self.cfg_bmi[attribute]
    
    #---------------------------------------------------------------------------- 
    def initialize_forcings(self):
        for forcing_name in self.cfg_train['dynamic_inputs']:
            setattr(self, forcing_name, 0)

    #------------------------------------------------------------ 
    def set_values_dictionary(self):
        """
            This is a dictionary of all the input values
            This is useful for creating the input array 
        """
        self._values = {self._var_name_map_short_first[x]:self.all_lstm_input_values_dict[x] for x in self.all_lstm_inputs}
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # BMI: Model Information Functions
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    
    def get_attribute(self, att_name):
    
        try:
            return self._att_map[ att_name.lower() ]
        except:
            print(' ERROR: Could not find attribute: ' + att_name)

    #--------------------------------------------------------
    # Note: These are currently variables needed from other
    #       components vs. those read from files or GUI.
    #--------------------------------------------------------   
    def get_input_var_names(self):

        return self._input_var_names

    def get_output_var_names(self):
 
        return self._output_var_names

    #-------------------------------------------------------------------
    # BMI: Variable Information Functions
    #-------------------------------------------------------------------
    #def get_value(self, var_name, dest):
    def get_value(self, var_name):
        """Copy of values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        Returns
        -------
        array_like
            Copy of values.
        """
        #dest[:] = self.get_value_ptr(var_name).flatten()
        return self.get_value_ptr(var_name)
        #return dest

    #-------------------------------------------------------------------
    def get_value_ptr(self, var_name):
        """Reference to values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        array_like
            Value array.
        """
        return self._values[var_name]

    #-------------------------------------------------------------------
    def get_var_name(self, long_var_name):
                              
        return self._var_name_map_long_first[ long_var_name ]

    #-------------------------------------------------------------------
    def get_var_units(self, long_var_name):

        return self._var_units_map[ long_var_name ]
                                                             
    #-------------------------------------------------------------------
    def get_var_type(self, long_var_name):

        return str( type(self.get_value( long_var_name )) )

    #-------------------------------------------------------------------
    def get_var_rank(self, long_var_name):

        return np.int16(0)

    #-------------------------------------------------------------------
    def get_start_time( self ):
    
        return 0.0

    #-------------------------------------------------------------------
    def get_end_time( self ):

        return (self.n_steps * self.dt)


    #-------------------------------------------------------------------
    def get_current_time( self ):

        return self.time

    #-------------------------------------------------------------------
    def get_time_step( self ):

        return self.dt

    #-------------------------------------------------------------------
    def get_time_units( self ):

        return self.get_attribute( 'time_units' ) 
       
    #-------------------------------------------------------------------
    def set_value(self, long_var_name, value):
        """Set model values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
              Array of new values.
        """ 
        var_name = self.get_var_name( long_var_name )
        setattr( self, var_name, value )

        # jmframe: this next line is basically a duplicate. 
        # I guess we should stick with the attribute names instead of a dictionary approach. 
        self.all_lstm_input_values_dict[var_name] = value

    #------------------------------------------------------------ 
    def get_component_name(self):
        """Name of the component."""
        return self._name

    #------------------------------------------------------------ 
    def get_input_item_count(self):
        """Get names of input variables."""
        return len(self._input_var_names)

    #------------------------------------------------------------ 
    def get_output_item_count(self):
        """Get names of output variables."""
        return len(self._output_var_names)

    #------------------------------------------------------------ 
    def set_value_at_indices(self, name, inds, src):
        """Set model values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        indices : array_like
            Array of indices.
        """
        val = self.get_value_ptr(name)
        val.flat[inds] = src

    #------------------------------------------------------------ 
    def get_var_nbytes(self, var_name):
        """Get units of variable.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        int
            Size of data array in bytes.
        """
        return self.get_value_ptr(var_name).nbytes

    #------------------------------------------------------------ 
    def get_value_at_indices(self, var_name, dest, indices):
        """Get values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        indices : array_like
            Array of indices.
        Returns
        -------
        array_like
            Values at indices.
        """
        dest[:] = self.get_value_ptr(var_name).take(indices)
        return dest

    #------------------------------------------------------------ 
    def get_grid_edge_count(self, grid):
        raise NotImplementedError("get_grid_edge_count")

    #------------------------------------------------------------ 
    def get_grid_edge_nodes(self, grid, edge_nodes):
        raise NotImplementedError("get_grid_edge_nodes")

    #------------------------------------------------------------ 
    def get_grid_face_count(self, grid):
        raise NotImplementedError("get_grid_face_count")
    
    #------------------------------------------------------------ 
    def get_grid_face_edges(self, grid, face_edges):
        raise NotImplementedError("get_grid_face_edges")

    #------------------------------------------------------------ 
    def get_grid_face_nodes(self, grid, face_nodes):
        raise NotImplementedError("get_grid_face_nodes")
    
    def get_grid_node_count(self, grid):
        raise NotImplementedError("get_grid_node_count")

    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        raise NotImplementedError("get_grid_nodes_per_face") 
    
    def get_grid_origin(self, grid_id, origin):
        raise NotImplementedError("get_grid_origin") 

    def get_grid_rank(self, grid_id):
        raise NotImplementedError("get_grid_rank") 

    def get_grid_shape(self, grid_id, shape):
        raise NotImplementedError("get_grid_shape") 

    def get_grid_size(self, grid_id):
        raise NotImplementedError("get_grid_size") 

    def get_grid_spacing(self, grid_id, spacing):
        raise NotImplementedError("get_grid_spacing") 

    def get_grid_type(self):
        raise NotImplementedError("get_grid_type") 

    def get_grid_x(self):
        raise NotImplementedError("get_grid_x") 

    def get_grid_y(self):
        raise NotImplementedError("get_grid_y") 

    def get_grid_z(self):
        raise NotImplementedError("get_grid_z") 

    def get_var_grid(self):
        raise NotImplementedError("get_var_grid") 

    def get_var_itemsize(self, name):
        return np.dtype(self.get_var_type(name)).itemsize

    def get_var_location(self, name):
        return self._var_loc[name]


    #-- Random utility functions

    def _parse_config(self, cfg):
        for key, val in cfg.items():
            # convert all path strings to PosixPath objects
            if any([key.endswith(x) for x in ['_dir', '_path', '_file', '_files']]):
                if (val is not None) and (val != "None"):
                    if isinstance(val, list):
                        temp_list = []
                        for element in val:
                            temp_list.append(Path(element))
                        cfg[key] = temp_list
                    else:
                        cfg[key] = Path(val)
                else:
                    cfg[key] = None

            # convert Dates to pandas Datetime indexs
            elif key.endswith('_date'):
                if isinstance(val, list):
                    temp_list = []
                    for elem in val:
                        temp_list.append(pd.to_datetime(elem, format='%d/%m/%Y'))
                    cfg[key] = temp_list
                else:
                    cfg[key] = pd.to_datetime(val, format='%d/%m/%Y')

            else:
                pass

        # Add more config parsing if necessary
        return cfg