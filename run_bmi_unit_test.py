"""Run BMI Unit Testing.
Author: jgarrett
Date: 08/31/2021"""

import os
import sys
import numpy as np

#import torch
#from torch import nn
from pathlib import Path
from netCDF4 import Dataset
import bmi_lstm # This is the BMI LSTM that we will be running


# setup a "success counter" for number of passing and failing bmi functions
# keep track of function def fails (vs function call)
pass_count = 0
fail_count = 0
var_name_counter = 0
fail_list = []

def bmi_except(fstring):
    """Prints message and updates counter and list

    Parameters
    ----------
    fstring : str
        Name of failing BMI function 
    """
    
    global fail_count, fail_list, var_name_counter
    print("**BMI ERROR** in " + fstring)
    if (var_name_counter == 0):
        fail_count += 1
        fail_list.append(fstring)

bmi=bmi_lstm.bmi_LSTM()

print("\nBEGIN BMI UNIT TEST\n*******************\n");

# Define config path
#cfg_file=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bmi_config_files','01022500_A.yml'))
cfg_file=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '.', 'bmi_config_files','01022500_A.yml'))
#cfg_file=Path('./bmi_config_files/01022500_A.yml')

if os.path.exists(cfg_file):
    print(" configuration found: " + cfg_file)
else:
    print(" no configuration found, exiting...")
    sys.exit()

# initialize()
try: 
    #
    # Why doesn't this work???
    # bmi.initialize(cfg_file)
    bmi.initialize(bmi_cfg_file=Path('./bmi_config_files/01022500_A.yml'))
    print(" initializing...");
    pass_count += 1
except:
    bmi_except('initialize()')

# ---------- JG NOTE: from org run-lstm-bmi ----------
# # Get input data that matches the LSTM test runs
# print('Get input data that matches the LSTM test runs')
# sample_data = Dataset(Path('./data/usgs-streamflow-nldas_hourly.nc'), 'r')

# # Now loop through the inputs, set the forcing values, and update the model
# print('Now loop through the inputs, set the forcing values, and update the model')
# for precip, temp in zip(list(sample_data['total_precipitation'][3].data),
#                         list(sample_data['temperature'][3].data)):
#     model.set_value('atmosphere_water__liquid_equivalent_precipitation_rate',precip)
#     model.set_value('land_surface_air__temperature',temp)
#     print('the temperature and precipitation are set to {:.2f} and {:.2f}'.format(model.temperature, model.total_precipitation))
#     model.update()
#     print('the streamflow (CFS) at time {} is {:.2f}'.format(model.t, model.streamflow_cfs))

#     if model.t > 100:
#         print('stopping the loop')
#         break

print("\nMODEL INFORMATION\n*****************")

# get_component_name()
try:
    print (" component name: " + bmi.get_component_name())
    pass_count += 1
except:
    bmi_except('get_component_name()')

# get_input_item_count()
try:
    print (" input item count: " + str(bmi.get_input_item_count()))
    pass_count += 1
except:
    bmi_except('get_input_item_count()')

# get_output_item_count()
try:
    print (" output item count: " + str(bmi.get_output_item_count()))
    pass_count += 1
except:
    bmi_except('get_output_item_count()')

# get_input_var_names()
try:    
    # only print statement if names exist
    test_get_input_var_names =  bmi.get_input_var_names()
    if len(test_get_input_var_names) > 0:
        print (" input var names: ")
        for var_in in test_get_input_var_names:
            print ("  " + var_in)
    pass_count += 1
except:
    bmi_except('get_input_var_names()')

# get_input_var_names()
try:    
    # only print statement if out var list not null
    test_get_output_var_names =  bmi.get_output_var_names()
    if len(test_get_output_var_names) > 0:
        print (" output var names: ")
        for var_out in test_get_output_var_names:
            print ("  " + var_out)
    pass_count += 1
except:
    bmi_except('get_output_item_count()')
    

# setup array for get_value_*
# dest = np.empty(bmi.get_grid_size(0), dtype=float)

print ("\nGET AND SET VALUES\n******************")

# Get input data that matches the LSTM test runs
sample_data = Dataset(Path('./data/usgs-streamflow-nldas_hourly.nc'), 'r')

# for var_name in (bmi.get_output_var_names() + bmi.get_input_var_names()[3:6:2]):
# lets just do inputs
for var_name in (bmi.get_input_var_names()[3:6:2]):     
    print (" " + var_name + ":" )

    # get_value_ptr()
    try:
        print ("  get value ptr: " + str(bmi.get_value_ptr(var_name)))
        if var_name_counter == 0: 
            pass_count += 1
    except:
        bmi_except('get_value_ptr()')

    # get_value()
    try:
        #dest = np.empty(bmi.get_grid_size(0), dtype=float)
        print ("  get value: " + str(bmi.get_value(var_name)))
        if var_name_counter == 0: 
            pass_count += 1
    except:
        bmi_except('get_value()')

    # get_value_at_indices()    
    try: 
        dest0 = np.empty(bmi.get_grid_size(0), dtype=float)
        print ("  get value at indices: " + str(bmi.get_value_at_indices(var_name, dest0, [0])))
        if var_name_counter == 0: 
            pass_count += 1
    except: 
        bmi_except('get_value_at_indices()')

    
    # NOTE 09.10.2021:
    #   - SETTER functions are "passing" but cannot confirm proper behavior here
    #   as new get_value* not returning the same value as what was just set.
    #   - Two possibilities:
    #       1. Definitions themselves are faulty
    #       2. The way I am calling set_value*() is wrong    
    #   - Confident that get_value* functions are correct
    #   - See end of file for current console output

    # set_value()
    try:
        if var_name =='atmosphere_water__liquid_equivalent_precipitation_rate':
            for precip in list(sample_data['total_precipitation'][3].data):
                bmi.set_value(var_name,precip)
            print('set value" {:.2f}'.format(model.total_precipitation))
        
        if var_name =='land_surface_air__temperature':
            for temp in list(sample_data['temperature'][3].data):
                bmi.set_value(var_name,temp)
            print('set value" {:.2f}'.format(model.temperature))

        if var_name_counter == 0: 
            pass_count += 1
    except:
        bmi_except('set_value()')

    # set_value_at_indices()    
    # try:
    #     bmi.set_value_at_indices(var_name,[0], [-9])
    #     print ("  set value at indices: -9")
    #     dest2 = np.empty(bmi.get_grid_size(0), dtype=float)
    #     print ("  new value at indices: " + str(bmi.get_value_at_indices(var_name, dest2, [0])))
    #     #print ("  new value: " + str(bmi.get_value_ptr(var_name)))         
    #     if var_name_counter == 0: 
    #         pass_count += 1
    # except:
    #     bmi_except('set_value_at_indices()')
    

    var_name_counter += 1
        # update()
    try:
        bmi.update()
        print (" \nupdating...");
        pass_count += 1
        if bmi.t > 10:
            break
        # go ahead and print time to show iteration
        print (" current time: " + str(bmi.get_current_time()))
    except:
        bmi_except('update()')

# set back to zero
var_name_counter = 0


# update_until()
try:
    bmi.update_until(100)
    print (" \nupdating untill...");
    pass_count += 1
    # go ahead and print time to show iteration
    print (" current time: " + str(bmi.get_current_time()))
except:
    bmi_except('update_until()')          

print("\nVARIABLE INFORMATION\n********************")

for var_name in (bmi.get_output_var_names() + bmi.get_input_var_names()[3:6:2]):  
    print (" " + var_name + ":")

    # get_var_units()
    try: 
        print ("  units: " + bmi.get_var_units(var_name))
        if var_name_counter == 0:
            pass_count += 1
    except:
        bmi_except('get_var_units()')
    
    # get_var_itemsize()
    bmi.get_var_itemsize(var_name)
    try:
        print ("  itemsize: " + str(bmi.get_var_itemsize(var_name)))
        if var_name_counter == 0:
            pass_count += 1
    except:
        bmi_except('get_var_itemsize()')

    # get_var_type()
    try:
        print ("  type: " + bmi.get_var_type(var_name))
        if var_name_counter == 0:
            pass_count += 1
    except:
        bmi_except('get_var_type()')

    # get_var_nbytes()
    try:
        print ("  nbytes: " + str(bmi.get_var_nbytes(var_name)))
        if var_name_counter == 0:
            pass_count += 1
    except:
        bmi_except('get_var_nbytes()')

    # get_var_grid
    try:
        print ("  grid id: " + str(bmi.get_var_grid(var_name)))
        if var_name_counter == 0:
            pass_count += 1
    except:
        bmi_except('get_var_grid()')

    # get_var_location
    try:
        print ("  location: " + bmi.get_var_location(var_name))
        if var_name_counter == 0:
            pass_count += 1
    except:
        bmi_except('get_var_location()')

    var_name_counter += 1

#reset back to zero
var_name_counter = 0

print("\nGRID INFORMATION\n****************")
grid_id = 0 # there is only 1
print (" grid id: " + str(grid_id))

# get_grid_rank()
try:
    print ("  rank: " + str(bmi.get_grid_rank(grid_id)))
    pass_count += 1
except:
    bmi_except('get_grid_rank()')

# get_grid_size()
try:    
    print ("  size: " + str(bmi.get_grid_size(grid_id)))
    pass_count += 1
except:
    bmi_except('get_grid_size()')

# get_grid_type()    
try:
    print ("  type: " + bmi.get_grid_type(grid_id))
    pass_count += 1
except:
    bmi_except('get_grid_type()')    

print("\nTIME INFORMATION\n****************")

# get_start_time()
try:
    print (" start time: " + str(bmi.get_start_time()))
    pass_count += 1
except:
    bmi_except('get_start_time()')

# get_end_time()
try:
    print (" end time: " + str(bmi.get_end_time()))
    pass_count += 1
except:
    bmi_except('get_end_time()')

# get_current_time()
try:
    print (" current time: " + str(bmi.get_current_time()))
    pass_count += 1
except:
    bmi_except('get_current_time()')

# get_time_step()
try:
    print (" time step: " + str(bmi.get_time_step()))
    pass_count += 1
except:
    bmi_except('get_time_step()')

# get_time_units()
try:
    print (" time units: " + bmi.get_time_units())
    pass_count += 1
except:
    bmi_except('get_time_units()')


# finalize()
try:
    bmi.finalize()
    print (" \nfinalizing...")
    pass_count += 1
except:
    bmi_except('finalize()')

# lastly - print test summary
print (" Total BMI function PASS: " + str(pass_count))
print (" Total BMI function FAIL: " + str(fail_count))
for ff in fail_list:
    print ("  " + ff)
#print (str(var_name_counter))



    