"""
Functions to set up a SHETRAN catchment

"""
import os
import sys
import itertools
import datetime
import pickle

import numpy as np
import pandas as pd
import xarray as xr

# -----------------------------------------------------------------------------
# Global / input variables

root = 'S:/'

# Open master input file (static fields)
static_inputs_path = root + 'SHETRAN_GB_2021/inputs/SHETRANGB_data.p'
static_inputs = open(static_inputs_path, 'rb')
ds = pickle.load(static_inputs)

# Helper dictionary of details of static fields
# #- keys are names used for output and values are lists of variable name in
# master static dataset alongside output number format
static_field_details = {
    'DEM': ['surface_altitude', '%.2f'],
    'MinDEM': ['surface_altitude_min', '%.2f'],
    'Lake': ['lake_presence', '%d'],
    'LandCover': ['land_cover_lccs', '%d'],
    'Soil': ['soil_type', '%d'],
}

# Climate input folders
prcp_input_folder = root + 'CEH-GEAR downloads/'
tas_input_folder = root + 'CHESS_T/'
pet_input_folder = root + 'CHESS/'

# -----------------------------------------------------------------------------

def read_ascii_raster(file_path, data_type=np.int, return_metadata=True):
    """Read ascii raster into numpy array, optionally returning headers."""
    headers = []
    dc = {}
    with open(file_path, 'r') as fh:
        for i in range(6):
            line = fh.readline()
            headers.append(line.rstrip())
            key, val = line.rstrip().split()
            dc[key] = val
    ncols = int(dc['ncols'])
    nrows = int(dc['nrows'])
    xll = float(dc['xllcorner'])
    yll = float(dc['yllcorner'])
    cellsize = float(dc['cellsize'])
    nodata = float(dc['NODATA_value'])
    
    arr = np.loadtxt(file_path, dtype=data_type, skiprows=6)
    
    headers = '\n'.join(headers)
    headers = headers.rstrip()
    
    if return_metadata:
        return(arr, ncols, nrows, xll, yll, cellsize, headers)
    else:
        return(arr)

def create_static_maps(xll, yll, ncols, nrows, cellsize, output_folder, headers, catch, mask, nodata=-9999):
    """Write ascii rasters for DEM, minimum DEM, lake map, vegetation type map
    and soil map.
    """
    xur = xll + (ncols * cellsize) - 1
    yur = yll + (nrows * cellsize) - 1
    
    catch_data = ds.sel(y=slice(yur, yll), x=slice(xll, xur))
    
    # Save each variable to ascii raster
    for array_name, array_details in static_field_details.items():
        array = catch_data[array_details[0]].values
        
        # Renumber soil types so consecutive from one
        if array_name == 'Soil':
            orig_soil_types = np.unique(array[mask != nodata]).tolist()
            new_soil_types = range(1, len(orig_soil_types)+1)
            for orig_type, new_type in zip(orig_soil_types, new_soil_types):
                array[array == orig_type] = new_type
        
        array[mask == nodata] = nodata
        output_path = output_folder + catch + '_' + array_name + '.asc'
        np.savetxt(
            output_path, array, fmt=array_details[1], header=headers, comments=''
        )
        
        # Get vegetation and soil type arrays for library file construction
        if array_name == 'LandCover':
            veg = array
        if array_name == 'Soil':
            soil = array
    
    # Also save mask
    np.savetxt(
        output_folder + catch + '_Mask.asc', mask, fmt='%d', header=headers, 
        comments=''
    )
    
    return(veg, soil, orig_soil_types, new_soil_types)

def get_catchment_coords_ids(xll, yll, urx, ury, cellsize, mask):
    """Find coordinates of cells in catchment and assign IDs."""
    x = np.arange(xll, urx+1, cellsize)
    y = np.arange(yll, ury+1, cellsize)
    y[::-1].sort()
    xx, yy = np.meshgrid(x, y)
    xx_cat = xx[mask==0]
    yy_cat = yy[mask==0]
    cat_coords = []
    for xv, yv in zip(xx_cat, yy_cat):
        cat_coords.append((yv, xv))
    
    cell_ids = np.zeros(xx.shape, dtype=np.int) - 9999
    id = 1
    for i in range(len(y)):
        for j in range(len(x)):
            if mask[i,j] == 0:
                cell_ids[i,j] = id
                id += 1
    
    return(cat_coords, cell_ids)

def make_series(
    input_files, input_folder, xll, yll, urx, ury, variable, startime, endtime,
    cat_coords, cell_ids, series_output_path, write_cell_id_map=False, 
    map_output_path=None, map_hdrs=None
    ):
    """Make and save climate time series for an individual variable."""    
    # Read time series for full grid of cells into df
    dfs = []
    for input_file in input_files:
        input_path = os.path.join(input_folder, input_file)
        
        ds = xr.open_dataset(input_path)
        if variable == 'rainfall_amount':
            ds_sel = ds.sel(y=slice(ury, yll), x=slice(xll, urx))
        else:
            ds_sel = ds.drop(['lat', 'lon'])
            ds_sel = ds_sel.sel(y=slice(yll, ury), x=slice(xll, urx))
        df = ds_sel[variable].to_dataframe()
        df = df.unstack(level=['y', 'x'])
        
        y_coords = list(df.columns.levels[1])
        y_coords.sort(reverse=True)
        x_coords = list(df.columns.levels[2])
        x_coords.sort(reverse=False)
        
        df_ord = df.loc[:, list(itertools.product([variable], y_coords, x_coords))]
        dfs.append(df_ord)
        
        ds.close()
    
    # Subset on time period and cells in catchment
    df = pd.concat(dfs).sort_index().loc[startime:endtime]
    tmp = np.asarray(df.columns[:])
    all_coords = [(y,x) for _, y, x in tmp]
    cat_indices = []
    ind = 0
    for all_pair in all_coords:
        if all_pair in cat_coords:
            cat_indices.append(ind)
        ind += 1
    df = df.iloc[:, cat_indices]
    
    # Convert from degK to degC if temperature
    if variable == 'tas':
        df -= 273.15

    # Write outputs
    headers = np.unique(cell_ids)
    headers = headers[headers >= 1]
    df.to_csv(series_output_path, index=False, float_format='%.2f', header=headers)
    if write_cell_id_map:
        np.savetxt(map_output_path, cell_ids, fmt='%d', header=map_hdrs, comments='')

def get_date_components(date_string, fmt='%Y-%m-%d'):
    date = datetime.datetime.strptime(date_string, fmt)
    return(date.year, date.month, date.day)

def create_climate_files(startime, endtime, mask_path, catch, output_folder):
    """Create climate time series."""
    start_year, _, _ = get_date_components(startime)
    end_year, _, _ = get_date_components(endtime)
    
    # Precipitation input folders and file lists
    prcp_input_files = [str(y) + '.nc' for y in range(start_year, end_year+1)]
    
    # Temperature input folders and file lists
    tmp = sorted(os.listdir(tas_input_folder))
    tas_input_files = []
    for fn in tmp:
        if int(fn.split('_')[-1][:4]) in range(start_year, end_year+1):
            tas_input_files.append(fn)
    
    # PET input folders and file lists
    tmp = sorted(os.listdir(pet_input_folder))
    pet_input_files = []
    for fn in tmp:
        if int(fn.split('_')[-1][:4]) in range(start_year, end_year+1):
            pet_input_files.append(fn)
    
    # Read catchment mask
    mask, ncols, nrows, xll, yll, cellsize, hdrs = read_ascii_raster(
        mask_path, data_type=np.int, return_metadata=True
    )
    
    # ---
    # Precipitation
    
    # Figure out coordinates of upper right
    urx = xll + (ncols-1)*cellsize
    ury = yll + (nrows-1)*cellsize
    
    # Get coordinates and IDs of cells inside catchment
    cat_coords, cell_ids = get_catchment_coords_ids(xll, yll, urx, ury, cellsize, mask)
    
    # Make precipitation time series and cell ID map
    series_output_path = output_folder + catch + '_Precip.csv'
    map_output_path = output_folder + catch + '_Cells.asc'
    if not os.path.exists(series_output_path):
        make_series(
            prcp_input_files, prcp_input_folder, xll, yll, urx, ury, 'rainfall_amount', 
            startime, endtime, cat_coords, cell_ids, series_output_path, 
            write_cell_id_map=True, map_output_path=map_output_path, map_hdrs=hdrs
        )
    
    # ---
    # Temperature
    
    # Cell centre ll coords
    xll_cc = xll + 500.0
    yll_cc = yll + 500.0
    
    # Figure out coordinates of upper right
    urx_cc = xll_cc + (ncols-1)*cellsize
    ury_cc = yll_cc + (nrows-1)*cellsize
    
    # Get coordinates and IDs of cells inside catchment
    cat_coords_cc = []
    for yv, xv in cat_coords:
        cat_coords_cc.append((yv+500.0, xv+500.0))
    
    # Make temperature time series
    series_output_path = output_folder + catch + '_Temp.csv'
    if not os.path.exists(series_output_path):
        make_series(
            tas_input_files, tas_input_folder, xll_cc, yll_cc, urx_cc, ury_cc, 'tas', 
            startime, endtime, cat_coords_cc, cell_ids, series_output_path
        )
    
    # ---
    # PET
    
    # Make PET time series
    series_output_path = output_folder + catch + '_PET.csv'
    if not os.path.exists(series_output_path):
        make_series(
            pet_input_files, pet_input_folder, xll_cc, yll_cc, urx_cc, ury_cc, 'pet', 
            startime, endtime, cat_coords_cc, cell_ids, series_output_path
        )

def get_veg_string(veg):
    """Get string containing vegetation details for the library file."""
    veg_vals = [int(v) for v in np.unique(veg[veg!=-9999])]
    strickler_dict = {1: 0.6, 2: 3, 3: 0.5, 4: 1, 5: 0.25, 6: 2, 7: 5}
    # extract the veg properties from the metadata
    ##veg_props = ds.land_cover_lccs.attrs["land_cover_key"][ds.land_cover_lccs.attrs["land_cover_key"]["Veg Type #"].isin(veg_vals)]
    veg_props = ds.land_cover_lccs.attrs["land_cover_key"].loc[ds.land_cover_lccs.attrs["land_cover_key"]["Veg Type #"].isin(veg_vals)].copy()
    veg_props["strickler"] = [strickler_dict[item] for item in veg_props["Veg Type #"]]
    # write the subset of properties out to a string
    veg_string = veg_props.to_csv(header=False, index=False)
    ##veg_string = "<VegetationDetail>" + veg_string[:-1].replace("\n", "</VegetationDetail>\n<VegetationDetail>") + "</VegetationDetail>\n"
    tmp = []
    for line in veg_string[:-1].split('\n'):
        tmp.append('<VegetationDetail>' + line.rstrip() + '</VegetationDetail>')
    veg_string = '\n'.join(tmp)
    return(veg_string)

def get_soil_strings(soil, orig_soil_types, new_soil_types):
    """Get the unique soil columns out for the library file."""
    ##soil_vals = [int(v) for v in np.unique(soil[soil!=-9999])]
    
    orig_soil_types = [int(v) for v in orig_soil_types]
    new_soil_types = [int(v) for v in new_soil_types]

    # Find the attributes of those columns
    ##soil_props = ds.soil_type.attrs["soil_key"][ds.soil_type.attrs["soil_key"]["Soil Category"].isin(soil_vals)]
    soil_props = ds.soil_type.attrs["soil_key"].loc[ds.soil_type.attrs["soil_key"]["Soil Category"].isin(orig_soil_types)].copy()
    
    for orig_type, new_type in zip(orig_soil_types, new_soil_types):
        soil_props.loc[soil_props['Soil Category'] == orig_type, 'tmp0'] = new_type
    soil_props['Soil Category'] = soil_props['tmp0'].values
    soil_props['Soil Category'] = soil_props['Soil Category'].astype(np.int)

    # Rename the soil types for the new format of shetran
    soil_props["New_Soil_Type"] = soil_props["Soil Type"].copy()
    aquifer_types = ["NoGroundwater", "LowProductivityAquifer", "ModeratelyProductiveAquifer", "HighlyProductiveAquifer"]
    ##soil_props.New_Soil_Type.loc[(~soil_props.New_Soil_Type.isin(aquifer_types)) & soil_props["Soil Layer"] == 1] = "Top_" + soil_props["Soil Type"]
    ##soil_props.New_Soil_Type.loc[(~soil_props.New_Soil_Type.isin(aquifer_types)) | soil_props["Soil Layer"] != 1] = "Sub_" + soil_props["Soil Type"]
    soil_props['tmp1'] = np.where(
        (~soil_props['Soil Type'].isin(aquifer_types)),
        'Top_' + soil_props['Soil Type'],
        soil_props['Soil Type']
    )
    soil_props['tmp2'] = np.where(
        (~soil_props['Soil Type'].isin(aquifer_types)),
        'Sub_' + soil_props['Soil Type'],
        soil_props['Soil Type']
    )
    soil_props['New_Soil_Type'] = np.where(
        soil_props['Soil Layer'] == 1, soil_props['tmp1'], soil_props['tmp2']
    )

    # Assign a new soil code to the unique soil types
    soil_codes = soil_props.New_Soil_Type.unique()
    soil_codes_dict = dict(zip(soil_codes, [i+1 for i in range(len(soil_codes))]))
    soil_props["Soil_Type_Code"] = [soil_codes_dict[item] for item in soil_props.New_Soil_Type]

    # Select the relevant information for the library file
    soil_types = soil_props.loc[:, ["Soil_Type_Code", "New_Soil_Type", "Saturated Water Content", "Residual Water Content", "Saturated Conductivity (m/day)", "vanGenuchten- alpha (cm-1)", "vanGenuchten-n"]]
    soil_types.drop_duplicates(inplace=True)

    soil_cols = soil_props.loc[:, ["Soil Category", "Soil Layer", "Soil_Type_Code", "Depth at base of layer (m)"]]

    # Write the subset of properties out to a string
    soil_types_string = soil_types.to_csv(header=False, index=False)
    soil_cols_string = soil_cols.to_csv(header=False, index=False)

    ##soil_types_string = "<SoilProperty>" + soil_types_string[:-1].replace("\n", "</SoilProperty>\n<SoilProperty>") + "</SoilProperty>\n"
    ##soil_cols_string = "<SoilDetail>" + soil_cols_string[:-1].replace("\n", "</SoilDetail>\n<SoilDetail>") + "</SoilDetail>\n"
    
    tmp = []
    for line in soil_types_string[:-1].split('\n'):
        tmp.append('<SoilProperty>' + line.rstrip() + '</SoilProperty>')
    soil_types_string = '\n'.join(tmp)
    
    tmp = []
    for line in soil_cols_string[:-1].split('\n'):
        tmp.append('<SoilDetail>' + line.rstrip() + '</SoilDetail>')
    soil_cols_string = '\n'.join(tmp)
    
    return(soil_types_string, soil_cols_string)

def create_library_file(
    output_folder, catch, veg_string, soil_types_string, soil_cols_string,
    startime, endtime, prcp_timestep=24, pet_timestep=24
    ):
    """Create library file."""
    start_year, start_month, start_day = get_date_components(startime)
    end_year, end_month, end_day = get_date_components(endtime)
    
    output_list = [
        '<?xml version=1.0?><ShetranInput>',
        '<ProjectFile>{}_ProjectFile</ProjectFile>'.format(catch),
        '<CatchmentName>{}</CatchmentName>'.format(catch),
        '<DEMMeanFileName>{}_DEM.asc</DEMMeanFileName>'.format(catch),
        '<DEMminFileName>{}_MinDEM.asc</DEMMinFileName>'.format(catch),
        '<MaskFileName>{}_Mask.asc</MaskFileName>'.format(catch),
        '<VegMap>{}_LandCover.asc</VegMap>'.format(catch),
        '<SoilMap>{}_Soil.asc</SoilMap>'.format(catch),
        '<LakeMap>{}_Lake.asc</LakeMap>'.format(catch),
        '<PrecipMap>{}_Cells.asc</PrecipMap>'.format(catch),
        '<PeMap>{}_Cells.asc</PeMap>'.format(catch),
        '<VegetationDetails>',
        '<VegetationDetail>Veg Type #, Vegetation Type, Canopy storage capacity (mm), Leaf area index, Maximum rooting depth(m), AE/PE at field capacity,Strickler overland flow coefficient</VegetationDetail>',
        veg_string,
        '</VegetationDetails>',
        '<SoilProperties>',
        '<SoilProperty>Soil Number,Soil Type, Saturated Water Content, Residual Water Content, Saturated Conductivity (m/day), vanGenuchten- alpha (cm-1), vanGenuchten-n</SoilProperty> Avoid spaces in the Soil type names',
        soil_types_string,
        '</SoilProperties>',
        '<SoilDetails>',
        '<SoilDetail>Soil Category, Soil Layer, Soil Type, Depth at base of layer (m)</SoilDetail>',
        soil_cols_string,
        '</SoilDetails>',
        '<InitialConditions>0</InitialConditions>',
        '<PrecipitationTimeSeriesData>{}_Precip.csv</PrecipitationTimeSeriesData>'.format(catch),
        '<PrecipitationTimeStep>{}</PrecipitationTimeStep>'.format(prcp_timestep),
        '<EvaporationTimeSeriesData>{}_PET.csv</EvaporationTimeSeriesData>'.format(catch),
        '<EvaporationTimeStep>{}</EvaporationTimeStep>'.format(pet_timestep),
        '<MaxTempTimeSeriesData>{}_Temp.csv</MaxTempTimeSeriesData>'.format(catch),
        '<MinTempTimeSeriesData>{}_Temp.csv</MinTempTimeSeriesData>'.format(catch),
        '<StartDay>{}</StartDay>'.format(start_day, '02'),
        '<StartMonth>{}</StartMonth>'.format(start_month, '02'),
        '<StartYear>{}</StartYear>'.format(start_year),
        '<EndDay>{}</EndDay>'.format(end_day, '02'),
        '<EndMonth>{}</EndMonth>'.format(end_month, '02'),
        '<EndYear>{}</EndYear>'.format(end_year),
        '<RiverGridSquaresAccumulated>2</RiverGridSquaresAccumulated> Number of upstream grid squares needed to produce a river channel. A larger number will have fewer river channels',
        '<DropFromGridToChannelDepth>2</DropFromGridToChannelDepth> The standard and minimum value is 2 if there are numerical problems with error 1060 this can be increased',
        '<MinimumDropBetweenChannels>0.5</MinimumDropBetweenChannels> This depends on the grid size and how steep the catchment is. A value of 1 is a sensible starting point but more gently sloping catchments it can be reduced.'
        '<RegularTimestep>1.0</RegularTimestep> This is the standard Shetran timestep it is autmatically reduced in rain. The standard value is 1 hour. The maximum allowed value is 2 hours',
        '<IncreasingTimestep>0.05</IncreasingTimestep> speed of increase in timestep after rainfall back to the standard timestep. The standard value is 0.05. If if there are numerical problems with error 1060 it can be reduced to 0.01 but the simulation will take longer.',
        '<SimulatedDischargeTimestep>24.0</SimulatedDischargeTimestep> This should be the same as the measured discharge',
        '<SnowmeltDegreeDayFactor>0.0002</SnowmeltDegreeDayFactor> Units  = mm s-1 C-1',
        '</ShetranInput>',
    ]
    output_string = '\n'.join(output_list)
    
    f = open(output_folder + catch + "_LibraryFile.xml", "w")
    f.write(output_string)
    f.close()

def process_catchment(catch, mask_path, startime, endtime, output_subfolder, q=None):
    """Create all files needed to run shetran-prepare."""
    print(catch)
    
    if not os.path.isdir(output_subfolder):
        os.mkdir(output_subfolder)
    
    try:
    
        # Read mask
        mask, ncols, nrows, xll, yll, cellsize, headers = read_ascii_raster(
            mask_path, data_type=np.int, return_metadata=True
        )
        
        # Create static maps and return veg (land cover) and soil arrays/info
        veg, soil, orig_soil_types, new_soil_types = create_static_maps(
            xll, yll, ncols, nrows, cellsize, output_subfolder, headers, catch, mask
        )
        
        # Create climate time series files (and cell ID map)
        create_climate_files(startime, endtime, mask_path, catch, output_subfolder)
        
        # Get strings of vegetation and soil properties/details for library file
        veg_string = get_veg_string(veg)
        soil_types_string, soil_cols_string = get_soil_strings(
            soil, orig_soil_types, new_soil_types
        )
        
        # Create library file
        create_library_file(
            output_subfolder, catch, veg_string, soil_types_string, soil_cols_string,
            startime, endtime
        )
        
        #sys.exit()
    
    except:
        pass













