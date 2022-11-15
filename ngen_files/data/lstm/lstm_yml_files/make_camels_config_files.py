import pandas as pd
import numpy as np

# Loop through camels basins
  # extract the camels attributes
  # Write a configuration file

with open("../data/camels_basin_list_516.txt", "r") as f:
    basin_list = pd.read_csv(f, header=None)

basin_attributes = {}

for attribute_type in ['clim', 'geol', 'hydro', 'name', 'soil', 'topo', 'vege']:
    with open("../data/camels_attributes_v2.0/camels_{}.txt".format(attribute_type), "r") as f:
        basin_attributes[attribute_type] = pd.read_csv(f, sep=";")
    basin_attributes[attribute_type] = basin_attributes[attribute_type].set_index("gauge_id")
    print(basin_attributes[attribute_type].loc[1022500, :])

