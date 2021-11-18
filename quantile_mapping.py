import os
import sys
import shutil

import numpy as np
from scipy.stats.mstats import mquantiles
from scipy.interpolate import interp1d

# -----------------------------------------------------------------------------

##scenario = sys.argv[1]

root = '/media/david/civg01/CONVEX/' # 'S:/' # 
historical_folder = root + 'SHETRAN_GB_2021/historical_210930/'
scenario_folder = root + 'SHETRAN_GB_2021/ukcp18rcm_210927/'
camels_metadata_path = root + 'SHETRAN_GB_2021/inputs/CAMELS_GB_topographic_attributes.csv'

log_path = scenario_folder + 'bc_log.csv'
log_append = True

scenarios = ['control', 'future'] # [scenario] # 
models = ['01', '04', '05', '06','07' , '08', '09', '10', '11', '12', '13', '15']
variables = ['Precip', 'PET', 'Temp']
variables_standard = {'Precip': 'pr', 'PET': 'pet', 'Temp': 'tas'}

use_multiprocessing = True # False # 
nprocs = 28 # 32

# -----------------------------------------------------------------------------

def process_catchment(catch, scenario, model, variable): # , q=None
    print(scenario, model, variables_standard[variable], catch)
    
    # Copy pre-correction files that do not need to be changed
    dst_folder = scenario_folder + scenario + '_bc/' + model + '/' + catch + '/'
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)
    if variable == 'Precip':
        # - cells map needs to come from historical, not pre-correction
        src = historical_folder + catch + '/' + catch + '_Cells.asc'
        dst = dst_folder + catch + '_Cells.asc'
        shutil.copy(src, dst)
        # - rest of files
        src_folder = scenario_folder + scenario + '/' + model + '/' + catch + '/'
        files_to_copy = [
            catch + "_DEM.asc",
            catch + "_Lake.asc",
            catch + "_LandCover.asc",
            catch + "_LibraryFile.xml",
            catch + "_Mask.asc",
            catch + "_MinDEM.asc",
            catch + "_Soil.asc"
        ]
        for file in files_to_copy:
            shutil.copy(src_folder + file, dst_folder + file)
    
    # Read catchment mask
    mask_path = historical_folder + catch + '/' + catch + '_Mask.asc'
    mask, ncols, nrows, xll, yll, cellsize, hdrs = read_ascii_raster(
        mask_path, data_type=np.int, return_metadata=True
    )
    
    # Read historical and scenario cell ID files
    historical_cells_path = (
        historical_folder + catch + '/' + catch + '_Cells.asc'
    )
    scenario_cells_path = (
        scenario_folder + 'control/' + model + '/' + catch + '/' + catch + 
        '_Cells.asc'
    )
    historical_cells = read_ascii_raster(historical_cells_path, return_metadata=False)
    scenario_cells = read_ascii_raster(scenario_cells_path, return_metadata=False)
    
    # Read control pre-correction time series file
    subfolder = (
            scenario_folder + 'control/' + model + '/' + catch + '/'
    )
    uncorr_control_path = subfolder + catch + '_' + variable + '.csv'
    uncorr_control_series = read_series(uncorr_control_path)
    
    # If correction of scenario other than control, read uncorrected series
    if scenario == 'future':
        subfolder = (
                scenario_folder + 'future/' + model + '/' + catch + '/'
        )
        uncorr_future_path = subfolder + catch + '_' + variable + '.csv'
        uncorr_future_series = read_series(uncorr_future_path)
    
    # Read historical time series file
    historical_subfolder = historical_folder + catch + '/'
    historical_path = historical_subfolder + catch + '_' + variable + '.csv'
    historical_series = read_series(historical_path)
    
    # Cell-wise quantile mapping
    corr_series = {}
    for yi in range(nrows):
        for xi in range(ncols):
            m = mask[yi,xi]
            if m == 0:
                h_cell = historical_cells[yi,xi]
                s_cell = scenario_cells[yi,xi]
                
                h_series = np.asarray(historical_series[h_cell])
                uc_series = np.asarray(uncorr_control_series[s_cell])
                
                nbins = uc_series.shape[0] # 100
                
                if scenario == 'control':
                    c_series = eqm(
                        h_series, uc_series, uc_series, nbins=nbins, extrapolate='constant'
                    )
                elif scenario == 'future':
                    uf_series = np.asarray(uncorr_future_series[s_cell])
                    c_series = eqm(
                        h_series, uc_series, uf_series, nbins=nbins, extrapolate='constant'
                    )
                
                if variable in ['Precip', 'PET']:
                    c_series[c_series < 0.0] = 0.0
                
                corr_series[h_cell] = c_series
    
    # Write to output file
    subfolder = (
        scenario_folder + scenario + '_bc/' + model + '/' + catch + '/'
    )
    #if not os.path.exists(subfolder):
    #    os.mkdir(subfolder)
    output_path = subfolder + catch + '_' + variable + '.csv'
    hdrs = sorted(corr_series.keys())
    hdrs = ','.join(str(h) for h in hdrs)
    series_len = len(corr_series[1])
    ncells = len(corr_series.keys())
    with open(output_path, 'w') as fho:
        fho.write(hdrs + '\n')
        for t in range(series_len):
            output_line = []
            for ci in range(ncells):
                output_line.append(corr_series[ci+1][t])
            output_line = ','.join('{:.2f}'.format(val) for val in output_line)
            fho.write(output_line + '\n')

def process_catchment_mp(catch, scenario, model, variable, q):
    try:
        process_catchment(catch, scenario, model, variable)
        output_line = [scenario, model, variable, catch, 'Y']
    except:
        output_line = [scenario, model, variable, catch, 'N']
    output_line = ','.join(output_line)
    q.put(output_line)

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

def read_series(file_path):
    dc = {}
    with open(file_path, 'r') as fhi:
        hdrs = fhi.readline()
        hdrs = hdrs.rstrip().split(',')
        hdrs = [int(h) for h in hdrs]
        for cid in hdrs:
            dc[cid] = []
        for line in fhi:
            line = line.rstrip().split(',')
            cid = 1
            for val in line:
                dc[cid].append(float(val))
                cid += 1
    return dc

def eqm(obs, p, s, nbins=10, extrapolate=None):
    """Empirical quantile mapping.
    
    Based on: https://svn.oss.deltares.nl/repos/openearthtools/trunk/python/applications/hydrotools/hydrotools/statistics/bias_correction.py
    
    Args:
        obs: observed climate data for the training period
        p: simulated climate by the model for the same variable obs for the 
            training ("control") period
        s: simulated climate for the variables used in p, but considering the 
            test/projection ("scenario") period
        nbins: number of quantile bins
        extrapolate: None or 'constant', indicating the extrapolation method to
            be applied to correct values in 's' that are out of the range of 
            lowest and highest quantile of 'p'
    
    """
    binmid = np.arange((1./nbins)*0.5, 1., 1./nbins)
    
    qo = mquantiles(obs[np.isfinite(obs)], prob=binmid)
    qp = mquantiles(p[np.isfinite(p)], prob=binmid)
    
    p2o = interp1d(qp, qo, kind='linear', bounds_error=False)
    
    c = p2o(s)
    
    if extrapolate is None:
        c[s > np.max(qp)] = qo[-1]
        c[s < np.min(qp)] = qo[0]
    elif extrapolate == 'constant':
        c[s > np.max(qp)] = s[s > np.max(qp)] + qo[-1] - qp[-1]
        c[s < np.min(qp)] = s[s < np.min(qp)] + qo[0] - qp[0]
    
    return c

def log_status(log_path, q):
    with open(log_path, 'a') as fh:
        if not log_append:
            fh.write('Scenario,Model,Variable,Catchment,Flag\n')
        while True:
            msg = q.get()
            if msg == 'kill':
                break
            fh.write(msg + '\n')
            fh.flush()

def process_mp(cases_to_processs):
    
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(nprocs)
    
    logger = pool.apply_async(log_status, (log_path, q))
    
    jobs = []
    
    for case in cases_to_process:
        scenario, model, variable, catch = case
        job = pool.apply_async(process_catchment_mp, (catch, scenario, model, variable, q))
        jobs.append(job)

    for job in jobs: 
        job.get()
    
    q.put('kill')
    pool.close()
    pool.join()

# -----------------------------------------------------------------------------

catchments = []
with open(camels_metadata_path, 'r') as fh:
    fh.readline()
    for line in fh:
        catchments.append(line.rstrip().split(',')[0])

# TESTING
#catchments = ['10002']
#scenario = 'control' # 'future' # 
#variable = 'Precip'
#model = '01'

# Make initial log file based on cases processed so far
# - THEN COMMENT OUT
## models = ['01', '04', '05']
## with open(log_path, 'a') as fh_log:
##     fh_log.write('Scenario,Model,Variable,Catchment,Flag\n')
##     for scenario in scenarios:
##         for model in models:
##             for variable in variables:
##                 for catch in catchments:
##                     print(scenario, model, variable, catch)
##                     case_folder = scenario_folder + '/' + scenario + '_bc/' + model + '/' + catch + '/'
##                     series_path = case_folder + catch + '_' + variable + '.csv'
##                     if os.path.exists(series_path):
##                         flag = 'Y'
##                     else:
##                         flag = 'N'
##                     output_line = [scenario, model, variable, catch, flag]
##                     output_line = ','.join(output_line)
##                     fh_log.write(output_line + '\n')
## sys.exit()

processed = []
if log_append:
    with open(log_path, 'r') as fh_log:
        fh_log.readline()
        for line in fh_log:
            line = line.rstrip().split(',')
            ##case = line[:-1]
            case = line
            case = '_'.join(case)
            processed.append(case)

cases_to_process = []
for scenario in scenarios:
    for model in models:
        for variable in variables:
            for catch in catchments:
                case = '_'.join([scenario, model, variable, catch, 'Y'])
                if case not in processed:
                    cases_to_process.append([scenario, model, variable, catch])

if __name__ == "__main__":
    
    if use_multiprocessing:
        import multiprocessing as mp
        process_mp(cases_to_process)
    else:
        with open(log_path, 'a') as fh_log:
            if not log_append:
                fh_log.write('Scenario,Model,Variable,Catchment,Flag\n')
            
            for case in cases_to_process:
                scenario, model, variable, catch = case
                
                try:
                    process_catchment(catch, scenario, model, variable)
                    output_line = [scenario, model, variable, catch, 'Y']
                    output_line = ','.join(output_line)
                    fh_log.write(output_line + '\n')
                except:
                    output_line = [scenario, model, variable, catch, 'N']
                    output_line = ','.join(output_line)
                    fh_log.write(output_line + '\n')







