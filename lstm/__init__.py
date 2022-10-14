# Without line below, get error message from NextGen:
# NGen Framework 0.1.0
# Building Nexus collection
# Building Catchment collection
# libc++abi.dylib: terminating with uncaught exception of type std::runtime_error: AttributeError: module 'lstm' has no attribute 'bmi_lstm'
# zsh: abort      ./cmake_build/ngen ./data/catchment_data_lstm_test.geojson "cat-67"  "nex-65"

from .bmi_lstm import bmi_LSTM
