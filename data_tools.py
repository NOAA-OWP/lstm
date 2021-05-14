import pickle
import numpy as np
import pandas as pd
import xarray
from pathlib import Path

def nwmv3_dynamic_data(basin: str) -> pd.DataFrame:
    data_dir = '/glade/scratch/jframe/neuralhydrology/data/'
    # Load dynamic inputs
    with open(data_dir+"full_period.pickle", 'rb') as f:
        start_end_date = pickle.load(f)
    start_end_date = start_end_date[basin]
    start_date = start_end_date['start_dates']
    end_date = start_end_date['end_dates']

    with open(data_dir+'basin_ave_forcing/'+basin+'.pickle', 'rb') as f:
        basin_forcing = pickle.load(f)

    #basin_forcing = basin_forcing.set_index('TIME')
    basin_forcing = basin_forcing.rename_axis('date')

    with open(data_dir+'obs_q/'+basin+'.csv', 'r') as f:
        basin_q = pd.read_csv(f)
    basin_q = basin_q.set_index('POSIXct')
    basin_q = basin_q.rename_axis('date')

    df = basin_forcing.join(basin_q['obs'])

    # replace invalid discharge values by NaNs
    qobs_cols = [col for col in df.columns if 'obs' in col.lower()]
    for col in qobs_cols:
        df.loc[df[col] < 0, col] = np.nan
    df.loc[np.isnan(df['obs']),'obs'] = np.nan
    
    return df

def nwmv3_static_data() -> pd.DataFrame:
    data_dir = '/glade/work/jframe/data/nwmv3/'
    
    # Load attributes
    hi = pd.read_csv(data_dir+'meta/domainMeta_HI.csv')
    pr = pd.read_csv(data_dir+'meta/domainMeta_PR.csv')
    nosnol = pd.read_csv(data_dir+'meta/nosnowy_large_basin.csv')
    nosnos = pd.read_csv(data_dir+'meta/nosnowy_small_basin.csv')
    snol1 = pd.read_csv(data_dir+'meta/snowy_large_basin_1.csv')
    snol2 = pd.read_csv(data_dir+'meta/snowy_large_basin_2.csv')
    snos = pd.read_csv(data_dir+'meta/snowy_small_basin.csv')
    df = pd.concat([hi, pr, nosnol, nosnos, snol1, snol2, snos])
    df = df.set_index('site_no')

    # For some reason these areas aren't in the tables. I looked them up online.
    df.loc['50038100','area_sqkm'] = 525.768
    df.loc['50051800','area_sqkm'] = 106.42261

    return df

def load_hourly_nldas_forcings(file_path, lat, lon, area_sqkm) -> pd.DataFrame:
    """Load the hourly forcing data for a cell of the NLDAS data set.
    
    Parameters
    ----------
    data_dir : Path
        Path to the NLDAS directory. 
        The main directory (`data_dir`) should contain the folder nldas which again has to contain per year subfolders, 
        e.g. path/to/main_dir/nldas/yyyy/
    cell : str
        This is the lat/lon values in the format  "lat-lon.nc": xx.xxx-xx.xxx.nc
        The lat/lon values are of the NLDAS grid cell. They are available from one of the NLDAS time files of the CONUS grid.
        The lat/lon values are stored in a single dimension array
        Some of the lat/lon values are maksed out, and those cells are not saved.
        The files in the year subfolder are all the valid cells. 
    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the forcing data.
    """
    # Put all the data in a dictionary, for easy access.
    name_switch_nldas = {
        'LWRadAtm': 'LWDOWN',
        'pptrate': 'RAINRATE',
        'SWRadAtm': 'SWDOWN',
        'airtemp': 'T2D',
        'windspd': 'Wind',
        'spechum': 'Q2D',
        'airpres': 'PSFC'
        }

    # Read in forcing data from netcdf
    forcing_nc = xarray.open_dataset(file_path)
    new_index = pd.date_range(forcing_nc['time'][0].values,
                              forcing_nc['time'][-1].values + pd.to_timedelta('30min'),
                              freq='30T')
    forcing_nc = forcing_nc.interp(time=new_index).ffill(dim='time')

    # Files should have these variables for forcing data
    forcing_variables = ['airpres', 'airtemp', 'pptrate', 'spechum', 'windspd', 'LWRadAtm', 'SWRadAtm']

    df = forcing_nc.to_dataframe().droplevel('hru')[forcing_variables]

    # Create a dataframe with the dictionary, and make sure date is the index.
    df = df.rename(columns=name_switch_nldas)

    # index name required to be 'data'
    df.index.name = 'date'

    df['lat'] = [lat for i in range(df.shape[0])]
    df['lon'] = [lon for i in range(df.shape[0])]
    df['area_sqkm'] = [area_sqkm for i in range(df.shape[0])]

    return df
