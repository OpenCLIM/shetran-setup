import os
import sys
import numpy as np
import xarray as xr

# -----------------------------------------------------------------------------

ukcp18_folder = '/media/david/civg01/CONVEX/UKCP18_UK_12km/' # "S:/UKCP18_UK_12km/"
variables = ["hurs", "rls", "rss", "sfcWind", "tasmax", "tasmin"]
models = ['01', '04', '05', '06','07' , '08', '09', '10', '11', '12', '13', '15']
periods = [
    '19801201-19901130', '19901201-20001130', '20001201-20101130', 
    '20401201-20501130', '20501201-20601130', '20601201-20701130'
]
output_folder = '/media/david/civg01/CONVEX/SHETRAN_GB_2021/ukcp18_regional_pet/' # 'S:/SHETRAN_GB_2021/ukcp18_regional_pet/'

# -----------------------------------------------------------------------------

def calculateET(nssr, nslr, Tmax, Tmin, rh, uz):
    Rn = (nssr + nslr)/11.6 #divide by 11.6 to convert from Wm-2 to MJm-2
    G = 0 
    T = (Tmax + Tmin)/2
    
    es = ((0.6108* 2.718281828**(17.27*Tmax/(Tmax + 237.3))) + (0.6108* 2.718281828**(17.27*Tmin/(Tmin + 237.3))))/2
    ea = rh/100*es


    DELTA = (4098*es)/((T+237.3)**2)

    gamma = 0.000665* 101.3

    #maybe need this to convert to wind speed at 2m height

    #uz = uz * 0.514444444 # converting from knots to m/s
    ##u2 = uz*4.87/(math.log(67.8*10 - 5.42))
    u2 = uz*4.87/(np.log(67.8*10 - 5.42))


    ET0 = (0.408*DELTA*(Rn - G) + gamma*(900/(T + 273))*u2*(es-ea))/(DELTA+(gamma*(1+(0.34*u2))))
    
    # !! LIMIT TO ZERO !!
    ET0.values[0,:,:,:][ET0.values[0,:,:,:] < 0.0] = 0.0

    return(ET0)

# -----------------------------------------------------------------------------

dc = {}

for model in models:
    for period in periods:
        print(model, period)
        
        # Read input files
        del dc
        dc = {}
        for variable in variables:
            input_path = (
                ukcp18_folder + model + '/'+ variable + '/day/latest/'
                + variable + '_rcp85_land-rcm_uk_12km_' + model + '_day_'
                + period + '.nc'
            )
            dc[variable] = xr.open_dataset(input_path)
        
        # TESTING
        # pet = calculateET(
        #     dc['rss'].rss.sel(projection_y_coordinate=slice(-20000, 20000), projection_x_coordinate=slice(-20000, 20000)), 
        #     dc['rls'].rls.sel(projection_y_coordinate=slice(-20000, 20000), projection_x_coordinate=slice(-20000, 20000)), 
        #     dc['tasmax'].tasmax.sel(projection_y_coordinate=slice(-20000, 20000), projection_x_coordinate=slice(-20000, 20000)),
        #     dc['tasmin'].tasmin.sel(projection_y_coordinate=slice(-20000, 20000), projection_x_coordinate=slice(-20000, 20000)),
        #     dc['hurs'].hurs.sel(projection_y_coordinate=slice(-20000, 20000), projection_x_coordinate=slice(-20000, 20000)),
        #     dc['sfcWind'].sfcWind.sel(projection_y_coordinate=slice(-20000, 20000), projection_x_coordinate=slice(-20000, 20000)),
        # )
        # tmp1 = dc['hurs'].sel(projection_y_coordinate=slice(-20000, 20000), projection_x_coordinate=slice(-20000, 20000))
        
        # Calculate PET
        pet = calculateET(
            dc['rss'].rss, 
            dc['rls'].rls, 
            dc['tasmax'].tasmax,
            dc['tasmin'].tasmin,
            dc['hurs'].hurs,
            dc['sfcWind'].sfcWind,
        )        
        
        # Create dataset
        pet_ds = dc['hurs'].rename_vars({'hurs': 'pet'})
        pet_ds['pet'] = pet # pet_ds.assign({'pet': pet})
        pet_ds.pet.attrs['standard_name'] = 'potential_evapotranspiration'
        pet_ds.pet.attrs['long_name'] = 'Potential evapotranspiration'
        pet_ds.pet.attrs['units'] = 'mm/day'
        pet_ds.pet.attrs['description'] = 'FAO56 Penman-Monteith potential (reference) evapotranspiration (ET0)'
        pet_ds.pet.attrs['label_units'] = 'mm/day'
        pet_ds.pet.attrs['plot_label'] = 'Potential evapotranspiration'
        
        # Write output
        output_subfolder = output_folder + model + '/pet/day/latest/'
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        output_path = (
            output_subfolder + 'pet_rcp85_land-rcm_uk_12km_' + model + '_day_'
            + period + '.nc'
        )
        pet_ds.to_netcdf(output_path)
        
        #sys.exit()
        
        # Close input files
        for variable in variables:
            dc[variable].close()






