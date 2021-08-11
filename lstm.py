# LSTM here is based on PyTorch
import torch
from torch import nn
# Need these for BMI
import time
import data_tools
#import configparser
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from neuralhydrology.utils.config import Config

class bmi_LSTM(nn.Module):

    def __init__(self):
        super(bmi_LSTM, self).__init__()
        print('thank you for choosing LSTM')

    #--------------------------------------------------------------------------------------------------
    # This is the LSTM model. Based on the simple "CudaLSTM" in NeuralHydrology
    # Only meant for forward predictions, this is not for training. Do training in NeuralHydrology
    #--------------------------------------------------------------------------------------------------
    def initialize_lstm(self):
        self.lstm = nn.LSTM(self.input_size, self.hidden_layer_size)
        self.head = nn.Linear(self.hidden_layer_size, self.output_size)
        self._values = {}

    def forward(self, input_layer, h_t, c_t):
        h_t = h_t.float()
        c_t = c_t.float()
        input_layer = input_layer.float()
        input_view = input_layer.view(self.seq_length, self.batch_size, self.input_size)
        output, (h_t, c_t) = self.lstm(input_view, (h_t,c_t))
        prediction = self.head(output)
        return prediction, h_t, c_t

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
    # LWDOWN,PSFC,Q2D,RAINRATE,SWDOWN,T2D,U2D,V2D
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
    _var_name_map = { 
         'total_precipitation':'atmosphere_water__liquid_equivalent_precipitation_rate',
         'temperature':'land_surface_air__temperature'}

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

    #-------------------------------------------------------------------
    # BMI: Model Information Functions
    #-------------------------------------------------------------------
    def get_attribute(self, att_name):
    
        try:
            return self._att_map[ att_name.lower() ]
        except:
            print('###################################################')
            print(' ERROR: Could not find attribute: ' + att_name)
            print('###################################################')
            print()

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
                              
        return self._var_name_map[ long_var_name ]

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
    def set_value(self, var_name, src):
        """Set model values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
              Array of new values.
        """ 
        val = self.get_value_ptr(var_name)
        val[:] = src
 
    #------------------------------------------------------------
    # BMI: Model Control Functions
    #------------------------------------------------------------ 

    #-------------------------------------------------------------------
    def initialize( self, bmi_cfg_file=None ):
        
        # First read in the BMI configuration. This will direct all the next moves.
        if bmi_cfg_file is not None:
            self.cfg_bmi = Config._read_and_parse_config(bmi_cfg_file)
        else:
            print("Error: No configuration provided, nothing to do...")
    
        # ----    print some stuff for troubleshooting    ---- #
        if self.cfg_bmi['verbose'] > 1:
            print("Initializing LSTM")

        # Now load in the configuration file for the specific LSTM
        # This will include all the details about how the model was trained
        # Inputs, outputs, hyper-parameters, etc.
        if self.cfg_bmi['train_cfg_file'] is not None:
            self.cfg_train = Config._read_and_parse_config(self.cfg_bmi['train_cfg_file'])

        # Collect the LSTM model architecture details from the configuration file
        self.input_size        = len(self.cfg_train['dynamic_inputs']) + len(self.cfg_train['static_attributes'])
        self.hidden_layer_size = self.cfg_train['hidden_size']
        self.output_size       = len(self.cfg_train['target_variables']) 

        # ----    print some stuff for troubleshooting    ---- #
        if self.cfg_bmi['verbose'] >=5:
            print('LSTM model architecture')
            print('input size', type(self.input_size), self.input_size)
            print('hidden layer size', type(self.hidden_layer_size), self.hidden_layer_size)
            print('output size', type(self.output_size), self.output_size)
        
        # Now we need to initialize an LSTM model.
        self.initialize_lstm()

        # load in the model specific values (scalers, weights, etc.)

        # Scaler data from the training set. This is used to normalize the data (input and output).
        with open(self.cfg_train['run_dir'] / 'train_data' / 'train_data_scaler.p', 'rb') as fb:
            self.train_data_scaler = pickle.load(fb)

        # Save the default model weights. We need to make sure we have the same keys.
        default_state_dict = self.state_dict()

        # Trained model weights from Neuralhydrology.
        trained_model_file = self.cfg_train['run_dir'] / 'model_epoch{}.pt'.format(str(self.cfg_train['epochs']).zfill(3))
        trained_state_dict = torch.load(trained_model_file, map_location=torch.device('cpu'))

        # Changing the name of the head weights, since different in NH
        trained_state_dict['head.weight'] = trained_state_dict.pop('head.net.0.weight')
        trained_state_dict['head.bias'] = trained_state_dict.pop('head.net.0.bias')
        trained_state_dict = {x:trained_state_dict[x] for x in default_state_dict.keys()}

        # Load in the trained weights.
        self.load_state_dict(trained_state_dict)

        # ----    print some stuff for troubleshooting    ---- #
        if self.cfg_bmi['verbose'] >=5:
            print('Training data scalers')
            print(self.train_data_scaler)
            print('pytorch_weights_dict')
            print(pytorch_weights_dict)

    #------------------------------------------------------------ 
    def update(self):
        with torch.no_grad():
            print('updating for t: ', self.t)
            start_this_seq = self.t - self.seq_length
            print('start_this_seq', start_this_seq)
            self.input_layer = self.input_tensor[start_this_seq:self.t, :]
            lstm_output, self.h_t, self.c_t = self.forward(self.input_layer, self.h_t, self.c_t)
            self.streamflow = (lstm_output[0,0,0].numpy().tolist() * self.obs_std + self.obs_mean) * self.output_factor
            self.output_list.append(self.streamflow)
            self.t += 1
            print('for time: {} lstm output: {}'.format(self.t,self.streamflow))
    
    #------------------------------------------------------------ 
    def update_until(self, last_update):
        first_update=self.t
        for t in range(first_update, last_update):
            self.update()
    #------------------------------------------------------------    
    def finalize( self ):
        return 0

    #-------------------------------------------------------------------
    def convert_precipitation_units(self, precip, conversion_factor):     
        return precip * conversion_factor

    #-------------------------------------------------------------------
    def do_warmup(self):
        self.h_t = torch.zeros(1, self.batch_size, self.hidden_layer_size).float()
        self.c_t = torch.zeros(1, self.batch_size, self.hidden_layer_size).float()
        for n_warmup_loops in range(10):
            #for t in range(self.seq_length, self.warmup_tensor.shape[0]):
            for t in range(self.seq_length+1, self.warmup_tensor.shape[0]):
                with torch.no_grad():
                    input_layer = self.warmup_tensor[t-self.seq_length:t, :]
                    output, self.h_t, self.c_t = self.forward(input_layer, self.h_t, self.c_t)
                    h_t = self.h_t.transpose(0,1)
                    c_t = self.c_t.transpose(0,1)
                    if t == self.warmup_tensor.shape[0] - 1:
                        h_t_np = h_t[0,0,:].numpy()
                        h_t_df = pd.DataFrame(h_t_np)
                        h_t_df.to_csv(self.h_t_init_file)
                        c_t_np = c_t[0,0,:].numpy()
                        c_t_df = pd.DataFrame(c_t_np)
                        c_t_df.to_csv(self.c_t_init_file)
        h_t = self.h_t.transpose(0,1)
        c_t = self.c_t.transpose(0,1)
        h_t_np = h_t[0,0,:].numpy()
        h_t_df = pd.DataFrame(h_t_np)
        h_t_df.to_csv(self.h_t_init_file)
        c_t_np = c_t[0,0,:].numpy()
        c_t_df = pd.DataFrame(c_t_np)
        c_t_df.to_csv(self.c_t_init_file)

    #-------------------------------------------------------------------
    def read_initial_states(self):
        h_t = np.genfromtxt(self.h_t_init_file, skip_header=1, delimiter=",")[:,1]
        self.h_t = torch.tensor(h_t).view(1,1,-1)
        c_t = np.genfromtxt(self.c_t_init_file, skip_header=1, delimiter=",")[:,1]
        self.c_t = torch.tensor(c_t).view(1,1,-1)

    #-------------------------------------------------------------------
    def split_wind_components(self, single_wind_speed):
        U = single_wind_speed/2
        V = single_wind_speed/2
        return U, V

    #-------------------------------------------------------------------
    def load_forcing(self):
        with open(self.forcing_file, 'r') as f:
            df = pd.read_csv(f)
 #       df = df.rename(columns={'precip_rate':'RAINRATE', 'SPFH_2maboveground':'Q2D', 'TMP_2maboveground':'T2D', 
        df = df.rename(columns={'APCP_surface':'RAINRATE', 'SPFH_2maboveground':'Q2D', 'TMP_2maboveground':'T2D', 
                           'DLWRF_surface':'LWDOWN',  'DSWRF_surface':'SWDOWN',  'PRES_surface':'PSFC',
                           'UGRD_10maboveground':'U2D', 'VGRD_10maboveground':'V2D'})

        df['area_sqkm'] = [self.area_sqkm for i in range(df.shape[0])]
        df['lat'] = [self.lat for i in range(df.shape[0])]
        df['lon'] = [self.lon for i in range(df.shape[0])]
        
        # The precipitation rate units for the training set were obviously different than these forcings. Guessing it is a m -> mm conversion.
     #   df['RAINRATE'] = self.convert_precipitation_units(df['RAINRATE'], 1000)

        df = df.drop(['time'], axis=1) #cat-87.csv has no observation data

        df = df.loc[:,['RAINRATE', 'Q2D', 'T2D', 'LWDOWN',  'SWDOWN',  'PSFC',  'U2D', 'V2D', 'area_sqkm', 'lat', 'lon']]

        self.output_factor = df['area_sqkm'][0] * 35.315 # from m3/s to ft3/s

        self.steps_in_forcing = df.shape[0]

        self.forcings = pd.concat([self.nldas.iloc[(self.nldas.shape[0] - self.nwarm):,:], df])

    #-------------------------------------------------------------------
    def load_nldas_for_warmup(self):
        
        nldas = data_tools.load_hourly_nldas_forcings(self.warmup_forcing_file, self.lat, self.lon, self.area_sqkm)

        nldas['U2D'], nldas['V2D'] = self.split_wind_components(nldas['Wind'])

        self.nldas = nldas.loc['2015-01-01':'2015-11-30', ['RAINRATE', 'Q2D', 'T2D', 'LWDOWN',  'SWDOWN',  'PSFC',  
                                                           'U2D', 'V2D', 'area_sqkm', 'lat', 'lon']]

    #-------------------------------------------------------------------
    def load_observations(self):

        with open(self.observation_file, 'r') as f:
            obs = pd.read_csv(f)

        obs = pd.Series(data=list(obs[self.observation_column]), index=pd.to_datetime(obs.datetime)).resample('60T').mean()
        
        self.obs = obs.loc[self.test_date_start:self.test_date_end]

    #-------------------------------------------------------------------
    def load_scalers(self):

        with open(self.scaler_file, 'rb') as fb:
            self.scalers = pickle.load(fb)

    #-------------------------------------------------------------------
    def calc_metrics(self):
        diff_sum2 = 0
        diff_sum_mean2 = 0
        obs_mean = np.nanmean(np.array(self.obs))
        count_samples = 0
        for j, k in zip(self.output_list, self.obs):
            if np.isnan(k):
                continue
            count_samples += 1
            mod_diff = j-k
            mean_diff = k-obs_mean
            diff_sum2 += np.power((mod_diff),2)
            diff_sum_mean2 += np.power((mean_diff),2)
        nse = 1-(diff_sum2/diff_sum_mean2)
        print('Nash-Suttcliffe Efficiency', nse)
        print('on {} samples'.format(count_samples))

    #------------------------------------------------------------ 
    def run_model( self):
        for self.t in range(self.istart, self.input_tensor.shape[0]):
            with torch.no_grad():
                self.input_layer = self.input_tensor[self.t-self.seq_length:self.t, :]
                lstm_output, self.h_t, self.c_t = self.forward(self.input_layer, self.h_t, self.c_t)
                output = (lstm_output[0,0,0].numpy().tolist() * self.obs_std + self.obs_mean) * self.output_factor
                self.output_list.append(output)
                print(output)
        print('output stats')
        print('mean', np.mean(self.output_list))
        print('min', np.min(self.output_list))
        print('max', np.max(self.output_list))
        print('observation stats')
        print('mean', np.nanmean(self.obs))
        print('min', np.nanmin(self.obs))
        print('max', np.nanmax(self.obs))
        print('length obs', len(self.obs))
        print('length output', len(self.output_list))

    #---------------------------------------------------------------- 
    def run_unit_tests(self):
        #------------------------------------------------------------ 
        if self.get_output_var_names()[0] == 'land_surface_water__runoff_volume_flux':
            print('Unit test passed: get_output_var_names')
        else:
            print('Unit test failed: get_output_var_names')
        #------------------------------------------------------------ 
        if self.get_var_name('atmosphere_water__liquid_equivalent_precipitation_rate') == 'RAINRATE':
            print('Unit test passed: get_var_name')
        else:
            print('Unit test failed: get_var_name')
        #------------------------------------------------------------ 
        if self.get_var_units('atmosphere_water__liquid_equivalent_precipitation_rate') == 'kg m-2':
            if self.get_var_units("land_surface_water__runoff_volume_flux") == 'mm':
                print('Unit test passed: get_var_units')
            else:
                print('Unit test failed: get_var_units on land_surface_water')
        else:
            print('Unit test failed: get_var_units on atmospheric_water')
        #------------------------------------------------------------ 
        if self.get_var_rank("land_surface_water__runoff_volume_flux") == 0:
            print('Unit test passed: get_var_rank')
        else:
            print('Unit test failed: get_var_rank')
        #------------------------------------------------------------ 
        #------------------------------------------------------------ 
        #------------------------------------------------------------ 
        #------------------------------------------------------------ 
        #------------------------------------------------------------ 
        #------------------------------------------------------------ 















