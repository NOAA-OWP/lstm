import numpy as np
import pandas as pd
import torch
from torch import nn
import pickle
import matplotlib as plt
import data_tools
import bmi_lstm
from pathlib import Path

model = bmi_lstm.bmi_LSTM()

model.initialize(bmi_cfg_file=Path('lstm_bmi_config.yml'))

model.update()

runoff_output_list = []

for i in range(1,10):
#    model.set_value('land_surface_air__temperature',i)
#    model.set_value('atmosphere_water__liquid_equivalent_precipitation_rate',i)
#    print(model.all_lstm_inputs)
#    print(model.input_array)
#    print(model.input_array_scaled)
#    print(model.input_tensor)
    model.update()

print('self.out_mean', model.out_mean * model.output_factor)