# Copyright 2018 GRID-Geneva. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

# Import necessary stuff
import sys
import rasterio
from osgeo import gdal

import numpy as np
import pandas as pd
import xarray as xr

from datetime import datetime
from numpy.lib.stride_tricks import as_strided
from pyproj import Transformer
from itertools import product as iterprod

from utils.data_cube_utilities.dc_utilities import clear_attrs


def create_slc_clean_mask(slc, valid_cats = [4, 5, 6, 7, 11]):
    """
    Description:
      Create a clean mask from a list of valid categories,
    Input:
      slc (xarray) - slc from dc_preproc product (generated with sen2cor)
    Args:
      slc: xarray data array to extract clean categories from.
      valid_cats: array of ints representing what category should be considered valid.
      * category selected by default
      ###################################
      # slc categories:                 #
      #   0 - no data                   #
      #   1 - saturated or defective    #
      #   2 - dark area pixels          #
      #   3 - cloud_shadows             #
      #   4 * vegetation                #
      #   5 * not vegetated             #
      #   6 * water                     #
      #   7 * unclassified              #
      #   8 - cloud medium probability  #
      #   9 - cloud high probability    #
      #  10 - thin cirrus               #
      #  11 * snow                      #
      ###################################
    Output:
      clean_mask (boolean numpy array)
    """

    return xr.apply_ufunc(np.isin, slc, valid_cats).values


# Return unique values and count
def unik_count(vals):
    bc = vals.flatten()
    bc = np.bincount(bc)
    unik = np.nonzero(bc)[0]
    cnt = bc[unik] * 100
    return (unik, cnt)


# Return bit length
def bit_length(int_type):
    length = 0
    while (int_type):
        int_type >>= 1
        length += 1
    return(length)


def ls_qa_clean(dc_qa, valid_bits = [1, 2, 4]):
    """
    Description:
      create a clean mask of a Landsat Collection 1 dataset using pixel_qa band and a list of valid bits
    Input:
      dc_qa: pixel_qa band of a Landast Collection 1 xarray.DataArray
    Args:
      valid_bits: array of ints representing which bit should be considered as valid (default: clear, water, snow)
      #############################################
      # BITS : CATEGORIES                         #
      #    0 : Fill                               #
      #    1 : Clear                              #
      #    2 : Water                              #
      #    3 : Cloud shadow                       #
      #    4 : Snow                               #
      #    5 : Cloud                              #
      #   10 : Terrain occlusion (Landsat 8 only) #
      #############################################
    Output:
      clean_mask (boolean numpy array)
    """

    # Check submitted input
    if str(type(dc_qa)) != "<class 'xarray.core.dataarray.DataArray'>":
        sys.exit("SCRIPT INTERRUPTED: dc_qa should be an xarray.DataArray")
    if dc_qa.name != "pixel_qa":
        sys.exit("SCRIPT INTERRUPTED: dc_qa name  should be pixel_qa")

    # List and count all dc_qa unique values
    dc_qas, dc_cnt = unik_count(dc_qa.values)
    # Return bit encoding
    bit_len = bit_length(max(dc_qas))

    # First keep only low confidence cloud (and cirrus)
    ok_qas = []
    ko_qas = []

    if bit_len == 8: # Landsat 5 and 7
        for v in sorted(dc_qas):
            b = str(bin(v))[2:].zfill(bit_len)[::-1]
            if b[6] == '1' and b[7] == '0':
                ok_qas.append(v)
            else:
                ko_qas.append(v)

    if bit_len >= 10: # Landsat 8 (>= as sometimes pixel_qa become 11 bit !!!)
        for v in sorted(dc_qas):
            b = str(bin(v))[2:].zfill(bit_len)[::-1]
            if b[6] == '1' and b[7] == '0' and b[8] == '1' and b[9] == '0':
                ok_qas.append(v)
            else:
                ko_qas.append(v)

    # Second keep only valid_bits
    data_qas = []
    nodata_qas = []
    for v in sorted(ok_qas):
        b = str(bin(v))[2:].zfill(bit_len)[::-1]
        for c in valid_bits:
            if b[c] == '1':
                data_qas.append(v)
                break

    return xr.apply_ufunc(np.isin, dc_qa, data_qas, dask = 'allowed').values


def load_multi_clean(dc, products, measurements, valid_cats = [], **kwargs):
    """
    Description:
      Create a clean dataset (multi-product or not) using cleaning "autor's recommended ways"
      - ls_qa_clean
      - create_slc_clean_mask
      Scene without any data removed, sorted by ascending time 
    Input:
      dc:           datacube.api.core.Datacube
                    The Datacube instance to load data with.
    Args:
      products:     list of products
      valid_cats:   array of ints representing what category should be considered valid
                    * category selected by default
      # SENTINEL 2 ################################
      #   0 - no data                             #
      #   1 - saturated or defective              #
      #   2 - dark area pixels                    #
      #   3 - cloud_shadows                       #
      #   4 * vegetation                          #
      #   5 * not vegetated                       #
      #   6 * water                               #
      #   7 * unclassified                        #
      #   8 - cloud medium probability            #
      #   9 - cloud high probability              #
      #  10 - thin cirrus                         #
      #  11 * snow                                #
      #############################################
      # LANDSAT 5, 7 and 8 ########################
      #    0 : Fill                               #
      #    1 * Clear                              #
      #    2 * Water                              #
      #    3 : Cloud shadow                       #
      #    4 * Snow                               #
      #    5 : Cloud                              #
      #   10 : Terrain occlusion (Landsat 8 only) #
      #############################################
      any other default argument from dc.load (time, lon, lat, output_crs, resolution, resampling,...)
      
    Output:
      cleaned dataset and clean_mask sorted by ascending time
    Authors:
      Bruno Chatenoux (UNEP/GRID-Geneva, 15.03.2022)
    """
    
    # Check submitted input
    # Convert product string into list
    if isinstance(products, str):
        products = products.split()
        
    # Get common measurements
    common_measurements = []
    measurement_list = dc.list_measurements(with_pandas=False)
    for product in products:
        measurements_for_product = filter(lambda x: x['product'] == product, measurement_list)
        common_measurements.append(set(map(lambda x: x['name'], measurements_for_product)))
    common_measurements = list(set.intersection(*map(set, common_measurements)))
    assert len(common_measurements) > 0, \
           '! No common measurements found'
    
    # Check requested measurements are in common measurements
    assert all([item in common_measurements for item in measurements]), \
           f"""
           All requested measures are not available for each product
           Only {common_measurements} are available
           """
    
    # Add quality measurement for Landsat or Sentinel 2 products
    # using the first product as they shouldn't be mixed
    if products[0][:2] == 'ls' and 'pixel_qa' not in measurements:
        measurements.append('pixel_qa')
    elif products[0][:2] == 's2' and 'slc' not in measurements:
        measurements.append('slc')
    
    # Load and combine dataset
    ds_out = None
    for product in products:
        # load product dataset
        ds_tmp = dc.load(product = product, measurements = measurements, **kwargs)        
        
        if len(ds_tmp.variables) == 0: continue # skip the current iteration if empty

        # clean product dataset
        if products[0][:2] == 'ls':
            if len(valid_cats) == 0: valid_cats = [1, 2, 4]
            clean_mask_tmp = ls_qa_clean(ds_tmp.pixel_qa, valid_cats)
        elif products[0][:2] == 's2':
            if len(valid_cats) == 0: valid_cats = [4, 5, 6, 7, 11]
            clean_mask_tmp = create_slc_clean_mask(ds_tmp.slc, valid_cats)
        ds_tmp = ds_tmp.where(clean_mask_tmp)
        
        # remove time without any data
        ds_tmp = ds_tmp.dropna('time', how='all')
        
        # initiate or append to dataset to return
        if ds_out is None:
            ds_out = ds_tmp.copy(deep=True)
        else:
            ds_out = xr.concat([ds_out, ds_tmp], dim = 'time')
        del ds_tmp

    if ds_out is not None:
        # sort dataset by ascending time
        ds_out = ds_out.sortby('time')
        return (ds_out, ~np.isnan(ds_out[measurements[0]].values))
    else:
        return (0, 0)


# source: https://stackoverflow.com/questions/32846846/quick-way-to-upsample-numpy-array-by-nearest-neighbor-tiling
def tile_array(a, x0, x1, x2):
    t, r, c = a.shape                                    # number of rows/columns
    ts, rs, cs = a.strides                                # row/column strides 
    x = as_strided(a, (t, x0, r, x1, c, x2), (ts, 0, rs, 0, cs, 0)) # view a as larger 4D array
    return x.reshape(t*x0, r*x1, c*x2)                      # create new 2D array

def updown_sample(ds_l, ds_s, resampl):
    """
    Description:
      Up or down sample a "large" resolution xarray.Dataset (so far Landsat products) and a "small" resolution
      xarray.Dataset (so far Sentinel 2 product) and combine them into a single xarray.Dataset.
      "large" resolution must be a multiple of "small" resolution and geographical extent must be adequate.
      Xarray.Dataset need to be cleaned as mask band will be removed from the output
      To enforce this requirement usage of load_lss2_clean function (without the resampl option) is
      highly recommended.

    Args:
      ds_l:         'large' resolution xarray.Dataset
      ds_s:         'small' resolution xarray.Dataset
      resampl:      'up' to upsample
                    'down_mean' to downsample using mean values
                    'down_median' to downsample using median values
      
    Output:
      Upsampled and combined dataset and clean_mask sorted by ascending time.
    Authors:
      Bruno Chatenoux (UNEP/GRID-Geneva, 11.12.2019)
    """
    
    # check resampl options
    resampl_opts = ['up', 'down_mean', 'down_median']
    assert (resampl in resampl_opts) or (resampl == ''), \
           '\nif used, resample option must be %s' % resampl_opts
    
    # check ds ratio
    ratiox = len(ds_s.longitude.values) / len(ds_l.longitude.values)
    ratioy = len(ds_s.latitude.values) / len(ds_l.latitude.values)
    assert (ratiox == 3), \
           '\nthe ratio of the number of columns should be 3 (Landsat/Sentinel 2 only so far) !'
    assert (ratioy == 3), \
           '\nthe ratio of the number of rows should be 3 (Landsat/Seentinel 2 only so far) !'

    # check ds resolutions
    resx_l = (ds_l.longitude.values.max() - ds_l.longitude.values.min()) / (len(ds_l.longitude.values) - 1)
    resy_l = (ds_l.latitude.values.max() - ds_l.latitude.values.min()) / (len(ds_l.latitude.values) - 1)
    resx_s = (ds_s.longitude.values.max() - ds_s.longitude.values.min()) / (len(ds_s.longitude.values) - 1)
    resy_s = (ds_s.latitude.values.max() - ds_s.latitude.values.min()) / (len(ds_s.latitude.values) - 1)
    # in reason of proper float storage issue, compare resolution with a 0.1% accuracy
    assert ((abs(resx_s - resx_l / ratiox) / resx_s * 100) < 0.1), \
           '\nthe column resolution is not a mutiple of %i !' % (ratiox)
    assert ((abs(resy_s - resy_l / ratioy) / resy_s * 100) < 0.1), \
           '\nthe row resolution is not a mutiple of %i !' % (ratioy)
    
    # check spacing of ds top left pixel center with a 0.1%
    assert ((abs(ds_l.longitude.values.min() - ds_s.longitude.values.min()) - resx_s) < resx_s * 0.001), \
           '\nthe longitudinal extent of both dataset do not overlay properly !' + \
           '\nuse load_lss2_clean function to fix this issue'
    assert ((abs(ds_l.latitude.values.min() - ds_s.latitude.values.min()) - resy_s) < resy_s * 0.001), \
           '\nthe latitudinal extent of both dataset do not overlay properly !' + \
           '\nuse load_lss2_clean function to fix this issue'
    
    # check vars (without mask band as they will no be combined)
    vars_l = [ele for ele in sorted(list(ds_l.data_vars)) if ele not in ['pixel_qa', 'slc']]
    vars_s = [ele for ele in sorted(list(ds_s.data_vars)) if ele not in ['pixel_qa', 'slc']]
    assert (vars_l == vars_s), \
           '\nmeasurements in dataset are not identical'
    
    # upsample "large" dataset (using temporary array)
    for index, var in enumerate(vars_l):
        if resampl == 'up':
            arr_l = tile_array(ds_l[var].values, 1, int(ratiox), int(ratioy))
            da_l = xr.DataArray(arr_l, dims=['time', 'latitude', 'longitude'])
            da_l = da_l.assign_coords(time = ds_l.time,
                                        latitude = ds_s.latitude,
                                        longitude = ds_s.longitude)
            # combine s and l
            da = xr.concat([ds_s[var], da_l], dim = 'time')
        elif resampl[:5] == 'down_':
            # source: https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy/42463491#42463491
            # 4x faster than skimage way (who has an issue with median function in the case of large stdev !)
            t, lat, lon = ds_s[var].values.shape
            nlat = lat // ratiox
            nlon = lon // ratioy
            if resampl == 'down_median':
                arr_s = np.nanmedian(ds_s[var].values[:1*t, :int(nlat*ratioy), :int(nlon*ratiox)]. \
                        reshape(1, t, int(nlat), int(ratioy), int(nlon), int(ratiox)), axis=(0, 3, 5))
            elif resampl == 'down_mean':
                arr_s = np.nanmean(ds_s[var].values[:1*t, :int(nlat*ratioy), :int(nlon*ratiox)]. \
                        reshape(1, t, int(nlat), int(ratioy), int(nlon), int(ratiox)), axis=(0, 3, 5))
            da_s = xr.DataArray(arr_s, dims=['time', 'latitude', 'longitude'])
            da_s = da_s.assign_coords(time = ds_s.time,
                                      latitude = ds_l.latitude,
                                      longitude = ds_l.longitude)
            # combine l and s
            da = xr.concat([ds_l[var], da_s], dim = 'time')
        
        if index == 0:   
            ds = da.to_dataset(name = var)
        else:
            ds = ds.merge(da.to_dataset(name = var))

    # Sort dataset by ascending time
    ds = ds.sortby('time')
    
    return ds

def load_lss2_clean(dc, products, time, lon, lat, measurements,
                   resampl = '', valid_cats = [[],[]]):
    """
    Description:
      Create a clean dataset mixing Landsat and Sentinel 2 products (respectively with prefixs 'ls' and
      's2')
      and using cleaning "autor's recommended ways":
      - ls_qa_clean
      - create_slc_clean_mask
      Sorted by ascending time
      If resample option is activated ('up' or 'down_mean', 'down_median') up/downsampling is performed
      and products output combined into a single 'lss2' prefix
      This function works as load_multi_clean function, but with a mix of Landsat and Sentinel 2 products
      the resampl option was added (to optionally combine products output)

    Input:
      dc:           datacube.api.core.Datacube
                    The Datacube instance to load data with.
    Args:
      products:     list of products
      time:         pair (list) of minimum and maximum date
      lon:          pair (list) of minimum and maximum longitude
      lat:          pair (list) of minimum and maximum longitude
      measurements: list of measurements (without mask band, landsat and Sentinel 2 products prefix
                    shouls be 'ls or 's2)
      resampl:      (OPTIONAL) Up/Downsample ('up', 'down_mean', 'down_median' ) products and combine
                    their output
      valid_cats:   (OPTIONAL) list of list of ints representing what category should be considered valid
                    first Landsat categories, then Sentinel 2 categories
                    * meand category by default
      # SENTINEL 2 ################################
      #   0 - no data                             #
      #   1 - saturated or defective              #
      #   2 - dark area pixels                    #
      #   3 - cloud_shadows                       #
      #   4 * vegetation                          #
      #   5 * not vegetated                       #
      #   6 * water                               #
      #   7 * unclassified                        #
      #   8 - cloud medium probability            #
      #   9 - cloud high probability              #
      #  10 - thin cirrus                         #
      #  11 * snow                                #
      #############################################
      # LANDSAT 5, 7 and 8 ########################
      #    0 : Fill                               #
      #    1 * Clear                              #
      #    2 * Water                              #
      #    3 : Cloud shadow                       #
      #    4 * Snow                               #
      #    5 : Cloud                              #
      #   10 : Terrain occlusion (Landsat 8 only) #
      #############################################
    Output:
      cleaned dataset and clean_mask sorted by ascending time stored in dictionnaries,
      if no up/downsampling is performed dictionnaries contains the two Landsat and Sentinel 2 output
      products
    Authors:
      Bruno Chatenoux (UNEP/GRID-Geneva, 11.12.2019)
    """
    
    # intersect measurements with common measurements
    measurement_list = dc.list_measurements(with_pandas=False)
    for index, product in enumerate(products):
        measurements_for_product = filter(lambda x: x['product'] == product, measurement_list)
        valid_measurements_name_array = set(map(lambda x: x['name'], measurements_for_product))
        if index == 0:
            common_measurements = sorted(valid_measurements_name_array)
        else:
            common_measurements = sorted(set(common_measurements).intersection(valid_measurements_name_array))
    measurements = sorted(set(measurements).intersection(common_measurements))
    
    # dictionary sensor -> mask band (Higher resolution first !)
    dict_sensmask = {'ls':'pixel_qa',
                     's2': 'slc'}
    
    resampl_opts = ['up', 'down_mean', 'down_median']
    
    # check mix Landsat and Sentinel 2
    sensors = []
    for product in products:
        if product[:2] not in sensors:
            sensors.append(product[:2])
    assert (sorted(set(sensors)) == sorted(set(dict_sensmask.keys()))), \
           '\nA mix of Landsat and Sentinel 2 products is required !\nYou should use load_multi_clean function'
    
    assert (len(valid_cats) == 2), \
           '\nvalid_cats argument must be a list of list (read the doc for more details)'
    
    assert (resampl in resampl_opts) or (resampl == ''), \
           '\nif used, resample option must be %s' % resampl_opts
    
    dict_dsc = {}
    dict_cm = {}
    
    # Process first Landsat and then Sentinel 2 (based on dict_sensmask order)
    for index, sensor in enumerate(dict_sensmask.keys()):
        # fix Sentinel 2 geographical extent based on Landsat dataset
        if index == 1:
            resx = (dsc.longitude.values.max() - dsc.longitude.values.min()) / len(dsc.longitude.values)
            resy = (dsc.latitude.values.max() - dsc.latitude.values.min()) / len(dsc.latitude.values)
            lon = (dsc.longitude.values.min() - resx / 3, dsc.longitude.values.max() + resx / 3)
            lat = (dsc.latitude.values.min() - resy / 3, dsc.latitude.values.max() + resy / 3)
        
        dsc, cm = load_multi_clean(dc = dc,
                                  products = [prod for prod in products if prod[:2] == sensor] ,
                                  time = time,
                                  lon = lon,
                                  lat = lat,
                                  measurements = measurements + [dict_sensmask[sensor]], # append mask band
                                  valid_cats = valid_cats[index])
        dict_dsc[sensor] = dsc
        dict_cm[sensor] = cm
    
    if resampl in resampl_opts :
        dsc = updown_sample(dict_dsc['ls'], dict_dsc['s2'], resampl)
        dict_dsc = {}
        dict_cm = {}
        dict_dsc['lss2'] = dsc
        dict_cm['lss2'] = ~np.isnan(dsc[measurements[0]].values)
    
    return dict_dsc, dict_cm


def _get_transform_from_xr(dataset):
    """Create a geotransform from an xarray dataset.
    """

    cols = len(dataset.longitude)
    rows = len(dataset.latitude)
    pixelWidth = abs(dataset.longitude[-1] - dataset.longitude[0]) / (cols - 1)
    pixelHeight = abs(dataset.latitude[-1] - dataset.latitude[0]) / (rows - 1)

    from rasterio.transform import from_bounds
    geotransform = from_bounds(dataset.longitude[0] - pixelWidth / 2, dataset.latitude[-1] - pixelHeight / 2,
                               dataset.longitude[-1] + pixelWidth / 2, dataset.latitude[0] + pixelHeight / 2,
                               cols, rows)
    return geotransform


def write_geotiff_from_xr(tif_path, dataset, bands = None, no_data = -9999,
                          crs = None, compr = ''):
    """
    Write a geotiff from an xarray dataset
    Modified for SDC:
    - fixed pixel shift bug
    - original band name added to band numbers
    - compression option added

    Args:
        tif_path: path for the tif to be written to.
        dataset: xarray dataset
        bands: (OPTIONAL) list of strings representing the bands in the order
        they should be written, or all <dataset> bands by default.
        no_data: (OPTIONAL) nodata value for the dataset (-9999 by default)
        crs: (OPTIONAL) requested crs (in the case the info is not available in <dataset>
        compr: (OPTIONAL) compression option (None by default), could be e.g. 'DEFLATE' or 'LZW'

    """
    # Check CRS information is correctly provided
    try:
        ds_crs = dataset.crs
        if crs is None:
            crs = ds_crs
        elif crs != ds_crs:
            crs = None # as a direct assert returns an error and switch to except
    except:
        assert crs is not None, \
               '<dataset> do not contains crs attribute, you have to fill <crs>!'
    # assert outside of try as it returns an error and switch to except
    assert crs is not None, \
           '<crs> differ from <dataset> crs, simply keep <crs> empty!'
    
    # Check band information
    if bands is None:
        bands = list(dataset.data_vars)
    assert isinstance(bands, list), "Bands must a list of strings"
    assert len(bands) > 0 and isinstance(bands[0], str), "You must supply at least one band."
    
    # Create the geotiff
    with rasterio.open(
            tif_path,
            'w',
            driver='GTiff',
            height=dataset.dims['latitude'],
            width=dataset.dims['longitude'],
            count=len(bands),
            dtype=dataset[bands[0]].dtype,
            crs=crs,
            transform=_get_transform_from_xr(dataset),
            nodata=no_data,
            compress=compr) as dst:
        for index, band in enumerate(bands):
            dst.write(dataset[band].values, index + 1)
        dst.close()
    
    # set band names
    ds = gdal.Open(tif_path, gdal.GA_Update)
    for index, band in enumerate(bands):
        rb = ds.GetRasterBand(index + 1)
        rb.SetDescription(band)
    del ds
    

def new_get_query_metadata(dc, product, quick = False):
    """
    Gets a descriptor based on a request.

    Args:
        dc: The Datacube instance to load data with.
        product (string): The name of the product associated with the desired dataset.
        quick (boolean): Attempt to quickly get metadata from a small dataset, and process
                         the full dataset if not possible. tile_count will not be evaluated
                         with this option.

    Returns:
        scene_metadata (dict): Dictionary containing a variety of data that can later be
                               accessed.
    """
    todo = True
    if quick:
        limit = 10
        ds = dc.load(product, measurements = [], limit = limit)
        if len(ds.time) == limit:
            todo = False
            tile_count = 'not calculated with quick option'
    if todo:
        ds = dc.load(product, measurements = [])
        tile_count = ds.time.size
    
    if len(set(ds.dims).intersection(['x', 'y'])) >= 1:
        ds = ds.rename({'x': 'longitude', 'y': 'latitude'})
    
    resx = (max(ds.longitude.values) - min(ds.longitude.values)) / (len(ds.longitude) - 1)
    resy = (max(ds.latitude.values) - min(ds.latitude.values)) / (len(ds.latitude) - 1)
    minx = min(ds.longitude.values) - resx / 2
    miny = min(ds.latitude.values) - resy / 2
    maxx = max(ds.longitude.values) + resx / 2
    maxy = max(ds.latitude.values) + resy / 2
    
    return {'lon_extents': (minx, maxx),
            'lat_extents': (miny, maxy),
            'lon_res': resx,
            'lat_res': resy,
            'crs': ds.crs,
            'time_extents': (ds.time[0].values.astype('M8[ms]').tolist(),
                             ds.time[-1].values.astype('M8[ms]').tolist()),
            'tile_count': tile_count,
            'pixel_count': len(ds.longitude) * len(ds.latitude)}
    
def summarize_products_extents(dc, products):
    """
    Returns the maximum extent (in space and time) of a given list of products.
    Args:
        dc: The Datacube instance to load data with
        products (list): List of products to get metadata from.

    Returns:
        scene_metadata (dict): Dictionary of min and max extents.
    """
    miny, maxy = 1E27, -1E27
    minx, maxx = 1E27, -1E27
    start_date, end_date = datetime.strptime('2050-12-31', '%Y-%m-%d'), datetime.strptime('1970-01-01', '%Y-%m-%d')
    for product in products:
        mt = new_get_query_metadata(dc, product)
        miny = mt['lat_extents'][0] if mt['lat_extents'][0] < miny else miny
        maxy = mt['lat_extents'][1] if mt['lat_extents'][1] > maxy else maxy
        minx = mt['lon_extents'][0] if mt['lon_extents'][0] < minx else minx
        maxx = mt['lon_extents'][1] if mt['lon_extents'][1] > maxx else maxx
        start_date = mt['time_extents'][0] if mt['time_extents'][0] < start_date else start_date
        end_date = mt['time_extents'][1] if mt['time_extents'][1] > end_date else end_date
    
    return {'lat_extents': (miny, maxy),
            'lon_extents': (minx, maxx),
            'time_extents': (start_date, end_date)}


def get_products_attributes(dc, qry, cols = ['name', 'crs', 'resolution']):
    """
    Description:
      Get products attributes using a query (WITHOUT "", e.g. products['name'].str.startswith('SPOT'))
    Input:
      dc:           datacube.api.core.Datacube
                    The Datacube instance to load data with.
    Args:
      qry:          query string, e.g.:
                    "products['name'].str.startswith('SPOT')"
                    "products['name'].str.match('^SPOT.*$')" should give the same result as startswith example
                    "products['name'].str.match('^SPOT.*_PAN_scene$')"
                    
      cols:         (OPTIONAL) list of column names to get (you can view the column available by running 'dc.list_products().columns')
    Output:
      pandas.Dataframe
    Authors:
      Bruno Chatenoux (UNEP/GRID-Geneva, 5.11.2020)
    """
    products = dc.list_products()
    prod_df = products[eval(qry)][cols].reset_index().drop(['id'], axis=1)
    prod_df['measurements'] = prod_df.apply(lambda row: sorted(map(lambda x: x['name'],
                                                                   filter(lambda x: x['product'] == row['name'],
                                                                          dc.list_measurements(with_pandas=False)))), axis=1)
    return(prod_df)

def time_list(ds):
    time_list = []
    for i in range(len(ds.time)):
        time_list.append(i)
    return time_list

# source: https://stackoverflow.com/questions/57856010/automatically-optimizing-pandas-dtypes
def optimize_types(dataframe):
    """
    The function takes in a dataframe and returns the same dataframe with optimized data types
    
    :param dataframe: the dataframe to optimize
    :return: the dataframe with the optimized types.
    """
    np_types = [np.int8 ,np.int16 ,np.int32, np.int64,
               np.uint8 ,np.uint16, np.uint32, np.uint64]
    np_types = [np_type.__name__ for np_type in np_types]
    type_df = pd.DataFrame(data=np_types, columns=['class_type'])

    type_df['min_value'] = type_df['class_type'].apply(lambda row: np.iinfo(row).min)
    type_df['max_value'] = type_df['class_type'].apply(lambda row: np.iinfo(row).max)
    type_df['range'] = type_df['max_value'] - type_df['min_value']
    type_df.sort_values(by='range', inplace=True)
    for col in dataframe.loc[:, dataframe.dtypes <= np.integer]:
        col_min = dataframe[col].min()
        col_max = dataframe[col].max()
        temp = type_df[(type_df['min_value'] <= col_min) & (type_df['max_value'] >= col_max)]
        optimized_class = temp.loc[temp['range'].idxmin(), 'class_type']
        dataframe[col] = dataframe[col].astype(optimized_class)
    return dataframe

def df_point_append_values(df, df_lon, df_lat, df_crs, ds, pts_shift = 0):
    """
    The function takes a csv dataframe, a xarray.Dataset, the names of the longitude and latitude
    columns in the csv dataframe, the crs of the csv dataframe, and the shift value (in degrees) to be
    added to the csv dataframe longitude and latitude columns (in csv dataframe coordinates units)
    
    :param df: the dataframe to which you want to append the xarray values
    :param df_lon: the name of the longitude column in your csv file
    :param df_lat: the name of the latitude column in your csv file
    :param df_crs: the coordinate reference system of the csv file
    :param ds: the xarray dataset to get values from
    :param pts_shift: the number of pixels to shift the csv points up and right , defaults to 0 (optional,
    this option is usefull when for example points location correspond to the exact corner of the xarray
    dataset (positive value are appropriate to lower-left corner)
    :return: A dataframe with the same number of rows as the input dataframe, and one column for each of
    the variables in the xarray.Dataset.
    """
    # get the real bbox of the dataset (as by default extent is given to the center of corner pixels)
    ds_min_lon = float(ds.longitude.min())
    ds_max_lon = float(ds.longitude.max())
    ds_min_lat = float(ds.latitude.min())
    ds_max_lat = float(ds.latitude.max())

    # get the resolution of a pixel
    resx = (ds_max_lon - ds_min_lon) / (len(ds.longitude.values) - 1)
    resy = (ds_max_lat - ds_min_lat) / (len(ds.latitude.values) - 1)
    # extend by half a pixel
    ds_min_lon = ds_min_lon - (resx / 2)
    ds_max_lon = ds_max_lon + (resx / 2)
    ds_min_lat = ds_min_lat - (resy / 2)
    ds_max_lat = ds_max_lat + (resy / 2)
    
    ds_crs = int(ds.crs.split(':')[1])

    # reproject real bbox corners from ds to csv CRS
    # source: https://hatarilabs.com/ih-en/how-to-translate-coordinate-systems-for-xy-point-data-tables-with-python-pandas-and-pyproj
    transformer = Transformer.from_crs(f"epsg:{ds_crs}", f"epsg:{df_crs}",always_xy=True)
    corners = list(iterprod([ds_min_lon, ds_max_lon], [ds_min_lat, ds_max_lat]))
    trans_corners = np.array(list(transformer.itransform(corners)))   
    
    # clip the csv dataframe with reprojected bbox
    df = df[(df[df_lon] + pts_shift >= np.min(trans_corners[:, 0])) &
            (df[df_lon] + pts_shift <= np.max(trans_corners[:, 0])) &
            (df[df_lat] + pts_shift>= np.min(trans_corners[:, 1])) &
            (df[df_lat] + pts_shift <= np.max(trans_corners[:, 1]))]
    
    # reproject csv dataframe coordinates to ds CRS
    transformer = Transformer.from_crs(f"epsg:{df_crs}", f"epsg:{ds_crs}",always_xy=True)
    points = list(zip(df[df_lon],df[df_lat]))
    trans_coords = np.array(list(transformer.itransform(points)))
    
    # append trans_coords as get_x and get_y (coordinated to be used to get pixel values in xarray.Dataset)
    pd.options.mode.chained_assignment = None # fix for "A value is trying to be set on a copy of a slice from a DataFrame."
    df['get_x'] = trans_coords[:,0]
    df['get_y'] = trans_coords[:,1]
    
    # Get values of xarray.Dataset on points coordinates and append to csv dataframe
    # get
    ds_pts = ds.sel(longitude = xr.DataArray(df.get_x, dims=["point"]),
                    latitude = xr.DataArray(df.get_y, dims=["point"]),
                    method="nearest")
    df_pts = ds_pts.to_dataframe().drop(['latitude', 'longitude'], axis = 1)
    if 'time' in df_pts.columns:
        df_pts = df_pts.drop(['time'], axis = 1)
        
    # deal with duplicated column names
    for c in list(df_pts.columns):
        if c in list(df.columns):
            df_pts.rename(columns={c: f"{c}_joined"}, inplace=True)
        
    # append
    df = df.drop(['get_x', 'get_y'], axis = 1)
    df = df.join(df_pts)
    
    return df


def indices_ts_stats(ds, sm_dict, idces_dict, stats, nanpercs = None, verbose = False):
    """
    Given a dataset, a dictionary of "seasons", a dictionary of indices, a list of statistics, and
    optionnaly a list of percentiles the function returns a dataset with the statistics of the indices
    per season.
    
    :param ds: the dataset to be analyzed
    :param sm_dict: a dictionary with seasons names as keys and lists of months as values.
     'all' values will use the full dataset.
    :param idces_dict: a dictionnary with the indices names as keys and related functions as values.
     The dataset name in the functions need to be 'ds'.
    :param stats: a list of statistical functions to be applied to the data
     numpy by default, 'range' can also be used
    :param nanpercs: list of percentiles to calculate (OPTIONAL, required if np.nanpercentile in
     <stats>)
    :param verbose: Print processing description (OPTIONAL, False by default)
    
    :return: A dataset with the requested statistics per season and indice
    """
    if np.nanpercentile in stats:
        assert 'nanpercs' in locals(), \
               "!!!  <nanpercs> is required with 'np.nanpercentile' !!!"
    
    first = True
    for s, m in sm_dict.items():
        if verbose: print(f"  └{s}")
        if m == 'all':
            ds_s = ds
        else:
            ds_s = ds.sel(time=ds.time.dt.month.isin(m))
        if (not isinstance(ds_s, xr.Dataset)) or (len(ds_s.time) == 0):
            continue
            
        for idces,form in idces_dict.items():
            if verbose: print(f"   └{idces}")
            da_idces = eval(form.replace('ds.', 'ds_s.'))
            da_idces = da_idces.where(np.isfinite(da_idces)) # replace +-Inf by nan
            
            for i in range(0, len(stats)):
                if stats[i] == 'range':
                    da_stat = xr.DataArray(np.max(a = da_idces, axis = 0) - np.min(a = da_idces, axis = 0),
                                           dims = ['latitude', 'longitude'])
                    stat_name = stats[i]
                    if verbose: print(f"     └{stat_name}")
                    ds_stat = da_stat.assign_coords(longitude = da_idces.longitude.values,
                                                    latitude = da_idces.latitude.values).to_dataset(name = f'{s}_{idces}_{stat_name}')
                    del da_stat
                    if first:
                        first = False
                        ds_stats = ds_stat
                    else:
                        ds_stats = ds_stats.merge(ds_stat)
                    del ds_stat
                elif stats[i].__name__ == 'nanpercentile':
                    for pc in nanpercs:
                        da_stat = xr.DataArray(stats[i](a = da_idces, q = pc, axis = 0),
                                               dims = ['latitude', 'longitude'])
                        stat_name = f"{stats[i].__name__}{pc:02d}"
                        if verbose: print(f"     └{stat_name}")
                        ds_stat = da_stat.assign_coords(longitude = da_idces.longitude.values,
                                                        latitude = da_idces.latitude.values).to_dataset(name = f'{s}_{idces}_{stat_name}')
                        del da_stat
                        if first:
                            first = False
                            ds_stats = ds_stat
                        else:
                            ds_stats = ds_stats.merge(ds_stat)
                    del ds_stat
                else:
                    da_stat = xr.DataArray(stats[i](a = da_idces, axis = 0),
                                           dims = ['latitude', 'longitude'])
                    stat_name = stats[i].__name__ 
                    if verbose: print(f"     └{stat_name}")
                    ds_stat = da_stat.assign_coords(longitude = da_idces.longitude.values,
                                                    latitude = da_idces.latitude.values).to_dataset(name = f'{s}_{idces}_{stat_name}')
                    del da_stat
                    if first:
                        first = False
                        ds_stats = ds_stat
                    else:
                        ds_stats = ds_stats.merge(ds_stat)
                    del ds_stat
            del da_idces
        del ds_s
    return ds_stats
