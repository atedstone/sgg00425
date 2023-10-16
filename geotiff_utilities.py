"""
Updated versions of functions in sdc_utilities.py

AJT September 2023
"""

import rasterio

def _get_transform_from_xr(dataset, xdim='x', ydim='y'):
    """Create a geotransform from an xarray dataset.
    """
    cols = len(dataset[xdim])
    rows = len(dataset[ydim])
    pixelWidth = abs(dataset[xdim][-1] - dataset[xdim][0]) / (cols - 1)
    pixelHeight = abs(dataset[ydim][-1] - dataset[ydim][0]) / (rows - 1)

    from rasterio.transform import from_bounds
    geotransform = from_bounds(dataset[xdim][0] - pixelWidth / 2, dataset[ydim][-1] - pixelHeight / 2,
                               dataset[xdim][-1] + pixelWidth / 2, dataset[ydim][0] + pixelHeight / 2,
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

    # Check georeferencing information
    if 'x' in dataset.dims:
        key_width = 'x'
        key_height = 'y'
    else:
        key_width = 'longitude'
        key_height = 'latitude'
    
    # Create the geotiff
    with rasterio.open(
            tif_path,
            'w',
            driver='GTiff',
            height=dataset.dims[key_height],
            width=dataset.dims[key_width],
            count=len(bands),
            dtype=dataset[bands[0]].dtype,
            crs=crs,
            transform=_get_transform_from_xr(dataset, xdim=key_width, ydim=key_height),
            nodata=no_data,
            compress=compr) as dst:
        for index, band in enumerate(bands):
            dst.write(dataset[band].values, index + 1)
            dst.set_band_description(index + 1, band)
        
        dst.close()