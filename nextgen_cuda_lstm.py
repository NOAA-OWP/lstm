# LSTM here is based on PyTorch
import torch
from torch import nn
#--------------------------------------------------------------------------------------------------
# This is the LSTM model. Based on the simple "CudaLSTM" in NeuralHydrology
# Only meant for forward predictions, this is not for training. Do training in NeuralHydrology
#--------------------------------------------------------------------------------------------------
class Nextgen_CudaLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, batch_size, seq_length):
        super(Nextgen_CudaLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.seq_length = seq_length
        self.output_size = output_size
        self.batch_size = batch_size # We shouldn't neeed to do a higher batch size.
        self.lstm = nn.LSTM(self.input_size, self.hidden_layer_size)
        self.head = nn.Linear(self.hidden_layer_size, self.output_size)

    def forward(self, input_layer, h_t, c_t):
        h_t = h_t.float()
        c_t = c_t.float()
        input_layer = input_layer.float()
        input_view = input_layer.view(self.seq_length, self.batch_size, self.input_size)
        output, (h_t, c_t) = self.lstm(input_view, (h_t,c_t))
        prediction = self.head(output)
        return prediction, h_t, c_t