import os
import yaml
import pandas as pd

def do_the_config_generation(template_path, output_config_path):
    # Load CAMELS basin attributes
    basin_attributes = {}
    for attribute_type in ['clim', 'geol', 'hydro', 'name', 'soil', 'topo', 'vege']:
        attr_path = f"../data/camels_attributes_v2.0/camels_{attribute_type}.txt"
        basin_attributes[attribute_type] = pd.read_csv(attr_path, sep=";").set_index("gauge_id")

    # Load template YAML
    with open(template_path, "r") as f:
        config_template = yaml.safe_load(f)

    # Extract required attributes for the basin
    try:
        name_info = basin_attributes["name"].loc[basin_id_int]
        topo_info = basin_attributes["topo"].loc[basin_id_int]

        # Update YAML structure with extracted values
        config_template["basin_id"] = basin_id_int
        config_template["basin_name"] = name_info["gauge_name"]
        config_template["area_sqkm"] = float(topo_info["area_gages2"])
        config_template["lat"] = float(topo_info["gauge_lat"])
        config_template["lon"] = float(topo_info["gauge_lon"])
        config_template["elev_mean"] = float(topo_info["elev_mean"])
        config_template["slope_mean"] = float(topo_info["slope_mean"])

        # Save the modified YAML file
        with open(output_config_path, "w") as yaml_file:
            yaml.dump(config_template, yaml_file, default_flow_style=False)

        print(f"Generated {output_config_path}")

    except KeyError as e:
        print(f"Error: Missing attribute {e} for basin {basin_id_int}")

basin_ids = [
    "01013500",
    "01333000",
    "02046000",
    "03010655",
    "03439000",
    "04015330",
    "05057200",
    "05291000",
    "06221400",
    "07057500",
    "07291000",
    "08023080",
    "08267500",
    "09035900",
    "09386900",
    "10234500",
    "10259000",
    "12010000"
]
# for the sample AORC basins from https://github.com/NWC-CUAHSI-Summer-Institute/CAMELS_data_sample
for basin_id_str in basin_ids:
    basin_id_int = int(basin_id_str)  # Ensure it's a string for lookup
    template_path = "05057200_nh_AORC_hourly_ensemble.yml"  # Path to the template YAML
    output_config_path = f"{basin_id_str}_nh_AORC_hourly_ensemble.yml"  # Output file name
    do_the_config_generation(template_path, output_config_path)

# For the Sample NLDAS hourly data in NeuralHydrology: https://github.com/neuralhydrology/neuralhydrology/tree/master/test/test_data/camels_us/hourly
basin_ids = ["03015500", "01547700", "02064000"]
for basin_id_str in basin_ids:
    template_path = "01022500_nh_NLDAS_hourly.yml"  # Path to the template YAML
    output_config_path = f"{basin_id_str}_nh_NLDAS_hourly.yml"  # Output file name
    do_the_config_generation(template_path, output_config_path)