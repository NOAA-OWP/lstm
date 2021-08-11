import numpy as np
import pandas as pd
import torch
from torch import nn
import pickle
import matplotlib as plt
import data_tools
import lstm
from pathlib import Path

model = lstm.bmi_LSTM()

model.initialize(bmi_cfg_file=Path('lstm_bmi_config.yml'))

#model.set_value(1,'atmosphere_water__liquid_equivalent_precipitation_rate')

#model.update()
