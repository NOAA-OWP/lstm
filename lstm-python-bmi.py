import numpy as np
import pandas as pd
import torch
from torch import nn
import pickle
import matplotlib as plt
import data_tools
import lstm

model = lstm.bmi_LSTM()
model.read_cfg_file('lstm-info.cfg')
model.initialize()
model.do_warmup()
model.read_initial_states()
model.run_model()
model.calc_metrics()