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
import os
import logging
import rasterio
import re
from PIL import Image, ImageDraw, ImageFont

from IPython.display import display, HTML
from datetime import datetime
from base64 import b64encode
from io import BytesIO
from os.path import basename
from math import ceil

import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd

from ipyleaflet import (
    Map,
    basemaps,
    basemap_to_tiles,
    ImageOverlay,
    LayersControl,
    Rectangle,
    DrawControl
)

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from skimage import exposure
from math import cos, sin, asin, sqrt, radians, floor, log10
from osgeo import ogr
from shapely.geometry import Polygon

from pyproj import Transformer
from itertools import product as iterprod

from utils.data_cube_utilities.dc_display_map import _degree_to_zoom_level


def draw_map(lat_ext = None, lon_ext = None, crs = 'EPSG:4326', draw = True):
    """
    Draw a map with a rectangle drawn on it
    
    :param lat_ext: latitude extent of the map
    :param lon_ext: longitude extent of the map
    :param draw: set it up to False to use the function as a replacement for
    display_map (optional)
    :return: The map and if draw not False, the draw control.
    """
    # check options combination
    assert not((lat_ext is None) or (lon_ext is None)), \
           'lat_ext and lon_ext are required'
    assert lat_ext[0] < lat_ext[1], 'lat_ext values are in the wrong order'
    assert lon_ext[0] < lon_ext[1], 'lon_ext values are in the wrong order'

    # reproject bbox if required
    if crs != 'EPSG:4326':
        # reproject real bbox corners from default to riorepro CRSs
        # source: https://hatarilabs.com/ih-en/how-to-translate-coordinate-systems-for-xy-point-data-tables-with-python-pandas-and-pyproj
        transformer = Transformer.from_crs(crs.lower(), 'epsg:4326',always_xy=True)
        corners = list(iterprod(lon_ext, lat_ext))
        trans_corners = np.array(list(transformer.itransform(corners)))
        lon_ext = [np.min(trans_corners[:, 0]), np.max(trans_corners[:, 0])]
        lat_ext = [np.min(trans_corners[:, 1]), np.max(trans_corners[:, 1])]
    
    # Location
    center = [np.mean(lat_ext), np.mean(lon_ext)]

    # source: https://sdc.unepgrid.ch:8080/edit/utils/data_cube_utilities/dc_display_map.py
    margin = -0.5
    zoom_bias = 0
    lat_zoom_level = _degree_to_zoom_level(margin = margin, *lat_ext ) + zoom_bias
    lon_zoom_level = _degree_to_zoom_level(margin = margin, *lon_ext) + zoom_bias
    zoom = min(lat_zoom_level, lon_zoom_level)

    m = Map(center=center, zoom=zoom, scroll_wheel_zoom = True)

    # Layers
    # http://leaflet-extras.github.io/leaflet-providers/preview/
    esri = basemap_to_tiles(basemaps.Esri.WorldImagery)
    m.add_layer(esri)
    terrain = basemap_to_tiles(basemaps.Stamen.Terrain)
    m.add_layer(terrain)
    mapnik = basemap_to_tiles(basemaps.OpenStreetMap.Mapnik)
    m.add_layer(mapnik)

    rectangle = Rectangle(bounds = ((lat_ext[0], lon_ext[0]),
                                   (lat_ext[1], lon_ext[1])),
                          color = 'red', weight = 2, fill = False)

    m.add_layer(rectangle)

    m.add_control(LayersControl())

    if draw:
        dc = DrawControl(rectangle={'shapeOptions': {'color': '#0000FF'}},
                         polygon={'shapeOptions': {'color': '#0000FF'}},
                         marker={},
                         polyline={},
                         circle={},
                         circlemarker={}
                        )
        m.add_control(dc)
        return m, dc
    else:
        return m, None

def printandlog(msg, logname = 'default.log', started = False, reset = False):
    """
    Description:
      Function to print and write in a log file any info
    -----
    Input:
      message: Message to print and log
      started: Starting time to calculate processing time
      reset: Reset the existing log if True, or append to existing log if False (default)
      logname: Name of the logfile. It is strongly advised to defined it once in the configuration section
    Output:
      Print message in page and logname after date and time
    -----
    Usage:
      printandlog('Started computing', 'any_name.log', started = start_time, reset = True)
    """
    logging.basicConfig(filename=logname,
                        level=logging.INFO,
                        format='%(asctime)s | %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    if reset:
        open(logname, 'w').close()

    if started:
        msg = '%s (done in %s)' % (msg, datetime.now() - started)

    print('%s | %s' % (datetime.now(), msg))
    logging.info(msg)
    return


def str_ds(ds):
    """
    create a string from a given xarray.Dataset by combining geographical extent and resolution
    keeping 6 digits, removing '.', adding 3 leading 0 to longitude and 2 to latitude and using
    '-' as separator.

    Parameters
    ----------
    ds: xarray.Dataset
    """
    return '{:010.6f}-{:010.6f}-{:09.6f}-{:09.6f}-{:01.6f}' \
      .format(ds.longitude.min().values, ds.longitude.max().values,
              ds.latitude.min().values, ds.latitude.max().values,
              ds.longitude.resolution).replace('.','')

def da_linreg_params(y, dim = 'time'):
    """
    Description:
      Calculation of linear regression slope on a given xarray.DataArray.
      nan "bullet proof", faster than vectorized ufunc approach.
    Input:
      y:            xarray.DataArray
      dim:          x dimension (time per fault)
    Output:
      slope and intercept
    Authors:
      Bruno Chatenoux (UNEP/GRID-Geneva, 11.6.2019)
    """
    x = y.where(np.isnan(y), y[dim]) # attribute time to pixel with values

    mean_x = x.mean(dim=dim)
    mean_y = y.mean(dim=dim)
    mean_xx = (x * x).mean(dim=dim)
    mean_xy = (x * y).mean(dim=dim)

    s = ((mean_x * mean_y) - mean_xy) / ((mean_x * mean_x) - mean_xx)

    i = mean_y - mean_x * s

    return s, i


def da_to_png64(da, cm):
    """
    The function takes in a data array and a color map, and returns a base64 encoded string
    of the image
    
    :param da: the data array
    :param cm: The colormap to use
    :return: a string of the image encoded in base64.
    
    source: https://github.com/jupyter-widgets/ipyleaflet/blob/master/examples/Numpy.ipynb
    but without reprojection:
    - seems to have an issue with bounds still in WGS84 and array reprojected
    - reprojection create more problems than solve them
    """
    arr = da.values
    arr_norm = arr - np.nanmin(arr)
    arr_norm = arr_norm / np.nanmax(arr_norm)
    arr_norm = np.where(np.isfinite(arr), arr_norm, 0)
    arr_im = Image.fromarray(np.uint8(cm(arr_norm)*255))
#     arr_im = PIL.Image.fromarray(np.uint8(cm(arr_norm)*255))
    arr_mask = np.where(np.isfinite(arr), 255, 0)
    mask = Image.fromarray(np.uint8(arr_mask), mode='L')
    im = Image.new('RGBA', arr_norm.shape[::-1], color=None)
#     mask = PIL.Image.fromarray(np.uint8(arr_mask), mode='L')
#     im = PIL.Image.new('RGBA', arr_norm.shape[::-1], color=None)
    im.paste(arr_im, mask=mask)
    f = BytesIO()
    im.save(f, 'png')
    data = b64encode(f.getvalue())
    data = data.decode('ascii')
    imgurl = 'data:image/png;base64,' + data
    return imgurl


def display_da(da, cm, tool = 'point'):
    """
    Display a colored dataarray on a map and allow the user to select a point
    or a recangular region
    
    :param da: the DataArray you want to plot
    :param cm: colormap
    :param tool: 'point' or 'rectangle', defaults to point (optional)
    :return: a map, a drawcontrol, and an imageoverlay.
    """    
    # Check inputs
    assert 'dataarray.DataArray' in str(type(da)), "da must be an xarray.DataArray"
    assert tool in ['point', 'rectangle'], "<tool> must be either point (default) or rectangle"
    
    # convert DataArray to png64
    imgurl = da_to_png64(da, cm)
    
    
    # Display
    latitude = (da.latitude.values.min(), da.latitude.values.max())
    longitude = (da.longitude.values.min(), da.longitude.values.max())
    
    margin = -0.5
    zoom_bias = 0
    lat_zoom_level = _degree_to_zoom_level(margin = margin, *latitude ) + zoom_bias
    lon_zoom_level = _degree_to_zoom_level(margin = margin, *longitude) + zoom_bias
    zoom = min(lat_zoom_level, lon_zoom_level) - 1
    center = [np.mean(latitude), np.mean(longitude)]
    m = Map(center=center, zoom=zoom)
    
    # http://leaflet-extras.github.io/leaflet-providers/preview/
    esri = basemap_to_tiles(basemaps.Esri.WorldImagery)
    m.add_layer(esri)
    
    io = ImageOverlay(name = 'DataArray', url=imgurl, bounds=[(latitude[0],longitude[0]),(latitude[1], longitude[1])])
    m.add_layer(io)
    
    if tool == 'point':
        dc = DrawControl(circlemarker={'color': 'yellow'}, polygon={}, polyline={})
    else:
        dc = DrawControl(rectangle={'color': 'yellow'}, circlemarker={}, polygon={}, polyline={})
    m.add_control(dc)
    
    m.add_control(LayersControl())
    
    return m, dc, io


def fig_aspects(sizes, max_size = 20):
    """
    The function fig_aspects(sizes, max_size = 20) takes a dictionary of the 
    form {'longitude': pxx, 'latitude': pxy} and returns a tuple of the form 
    (height, width, orient, posit) where height and width are the height and 
    width of the figure in inches, orient is the orientation of the figure, and 
    posit is the position of the axis with respect to the figure
    
    :param sizes: a dictionary with keys 'latitude' and 'longitude'
    :param max_size: the maximum size of the figure in inches, defaults to 20 (optional)
    :return: a tuple of four values: height, width, orientation, and position.
    """
    pxx = sizes['longitude']
    pxy = sizes['latitude']
    orient = 'horizontal'
    posit = 'bottom'
    width = max_size
    height = pxy * (max_size / pxx)
    if pxx * 1.01 < pxy:
        orient = 'vertical'
        posit = 'right'
        height = max_size
        width = pxx * (max_size / pxy)
    return (height, width, orient, posit)

def xtrms_format(vals):
    """
    Given a list of values, return a list of strings that represent the minimum
    and maximum values in the list formated to a number of digits needed to
    appreciate the difference
    
    :param vals: A list of values to be passed to the function
    :return: The return value is a list of strings.
    """
    min_val = min(vals)
    max_val = max(vals)
    digits = floor(log10(max_val - min_val))
    if  digits < 1:
        digits = -digits + 1
    else:
        digits = 0

    return ['{:.{prec}f}'.format(round(min_val, digits), prec = digits),
            '{:.{prec}f}'.format(round(max_val, digits), prec = digits)]

def create_scalebar(data, ax, scalebar_color):
    """
    Description:
      Compute and create an horizontal scalebar in metre to be added to the map.
    -----
    Input:
      data: xarray to be mapped
      ax: matplotlib.axes to work on
      scalebar_color (OPTIONAL): scalebar color (e.g.'orangered')
                                 https://matplotlib.org/examples/color/named_colors.html)
    Output:
      Scalebar to be added
    """
    # Convert lenght at average latitude from decimal degree into kilometer
    ave_lat = ((min(data.latitude) + max(data.latitude)) / 2).values
    
    lon_width = max(data.longitude).values - min(data.longitude).values
    
    # convert dd to metre if width is bigger than Swiss width in dd (4.7°)
    if lon_width < 5:
        lon_width = dd2m(ave_lat, min(data.longitude).values, ave_lat, max(data.longitude).values) 
    # Calculate the scalebar caracteristics (rounded value of 1/4 of lengths)
    lon_px = len(data.longitude)
    bar_len = lon_width * 0.25 # 25% of the map width
    e = pow(10, floor(log10(bar_len)))
    bar_len = round(bar_len / e) * e
    units = 'metre'
    bar_px = round(lon_px * bar_len / lon_width)
    # add the scalebar
    fontprops = fm.FontProperties(size=18)
    scalebar = AnchoredSizeBar(ax.transData,
                               bar_px, '%i %s' % (bar_len, units), 'lower right', 
                               pad=0.1,
                               color=scalebar_color,
                               frameon=False,
                               size_vertical=1,
                               label_top=True,
                               fontproperties=fontprops)
    return(scalebar)

def dd2m(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    source: https://gis.stackexchange.com/questions/61924/python-gdal-degrees-to-meters-without-reprojecting
    
    :param lat1: latitude of the first point
    :param lon1: longitude of the first point
    :param lat2: latitude of the second point
    :param lon2: longitude of the second point
    :return: The distance between the two points in metres.
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km * 1000

def oneband_fig(data, leg, title, scalebar_color= None, fig_name=None, v_min=None, v_max=None, max_size=16):   
    """
    Description:
      Create a one band (one time) figure
    -----
    Input:
      data: one time xarray.DataArray.
      leg: colormap to be applied (either standard (https://matplotlib.org/examples/color/colormaps_reference.html)
           or custom)
      title: prefix of the figure title
      scalebar_color (OPTIONAL): scalebar color (https://matplotlib.org/examples/color/named_colors.html)
      v_min (OPTIONAL, default minimum value): minimum value to display.
      v_max (OPTIONAL, default maximum value): maximum value to display.
      fig_name (OPTIONAL): file name (including extension) to save the figure (show only if not added to input).
      max_size (OPTIONAL, default 16): maximum size of the figure (either horizontal or vertical).
    Output:
      figure.
    """
    # check options combination
    assert not((v_min is not None) ^ (v_max is not None)), \
           'v_min option requires v_max option, and inverserly'
    if v_min is not None:
        assert v_min < v_max, 'v_min value must be lower than v_max'

    height, width, orient, posit = fig_aspects(data.sizes, max_size)

    plt.close('all')
    fig, ax = plt.subplots()
    fig.set_size_inches(width, height)

    if not v_min and not v_max:
        im = ax.imshow(data, interpolation='nearest', cmap=leg)
    else:
        im = ax.imshow(data, interpolation='nearest', cmap=leg, vmin = v_min, vmax = v_max)

    # add a scalebar if required
    if scalebar_color:
        ax.add_artist(create_scalebar(data, ax, scalebar_color))

    # ticks moved 1 pixel inside to guarantee they are displayed
    plt.yticks([data.shape[0] - 1, 1], xtrms_format(data.latitude.values),
               rotation='vertical', va='center')
    plt.xticks([1, data.shape[1] - 1], xtrms_format(data.longitude.values))

    plt.title(title, weight='bold', fontsize=16)

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes(posit, "2%", pad="5%")
    cbar = fig.colorbar(im, orientation=orient, cax=cax)

    fig.patch.set_alpha(1) # opaque white border
    fig.tight_layout()
    if fig_name:
        plt.savefig(fig_name, dpi=150)
        display(HTML("""<a href="{}" target="_blank" >View and download {}</a>""".format(fig_name, basename(fig_name))))
    else:
        plt.show()
    plt.close()


def composite_fig(data, bands, title, scalebar_color=None, fig_name=None, max_size=16, hist_str=None, \
                  v_min = None, v_max = None):
    """
    Description:
      Create a three band (one time) composite figure
    -----
    Input:
      data: one time xarray.Dataset containing the three bands mentionned in bands.
      bands: bands to be used in the composite (RGB order).
      title: prefix of the figure title.
      scalebar_color (OPTIONAL): scalebar color (https://matplotlib.org/examples/color/named_colors.html)
      fig_name (OPTIONAL): file name (including extension) to save the figure (show only if not added to input).      
      max_size (OPTIONAL, default 16): maximum size of the figure (either horizontal or vertical).
      hist_str (OPTIONAL): histogram stretch type (['contr','eq','ad_eq']). Cannot be used with v_min, v_max options.
      v_min (OPTIONAL, default minimum value): minimum value to display. Cannot be used with hist_str option.
      v_max (OPTIONAL, default maximum value): maximum value to display. Cannot be used with hist_str option.
    Output:
      figure.
    """

    # check options combination
    assert not((hist_str is not None) and (v_min is not None or v_max is not None)) , \
           'hist_str option cannot be used with v_min, vmax options'
    assert not((v_min is not None) ^ (v_max is not None)), \
           'v_min option requires v_max option, and inverserly'
    if v_min is not None:
        assert v_min < v_max, 'v_min value must be lower than v_max'

    # Create a copy to unlink from original dataset
    rgb = data.copy(deep = True)

    height, width, orient, posit = fig_aspects(rgb.sizes, max_size)

    rgb = np.stack([rgb[bands[0]],
                    rgb[bands[1]],
                    rgb[bands[2]]])

    # perform stretch on each band
    for b in range(3):
        # https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html
        # Contrast stretching
        if hist_str == 'contr':
            p2, p98 = np.nanpercentile(rgb[b], (2, 98))
            rgb[b] = exposure.rescale_intensity(rgb[b], in_range=(p2, p98))
        # Equalization
        if hist_str == 'eq':
            rgb[b] = exposure.equalize_hist(rgb[b])
        # Adaptive Equalization
        if hist_str == 'ad_eq':
            rgb[b] = exposure.equalize_adapthist(rgb[b], clip_limit=0.03)

    rgb = np.stack(rgb, axis = -1)

    # normalize between 0 and 1
    if v_min is None:
        rgb = (rgb - np.nanmin(rgb)) / (np.nanmax(rgb) - np.nanmin(rgb))
    else:
        rgb = (rgb - v_min) / (v_max - v_min)

    # Start plotting the figure
    plt.close('all')
    fig, ax = plt.subplots()
    fig.set_size_inches(width, height)
    im = ax.imshow(rgb, vmin = 0, vmax = 1)

    # add a scalebar if required
    if scalebar_color:
        ax.add_artist(create_scalebar(data, ax, scalebar_color))

    # ticks moved 1 pixel inside to guarantee they are displayed
    plt.yticks([rgb.shape[0] - 1, 1], xtrms_format(data.latitude.values),
              rotation='vertical', va='center')
    plt.xticks([1, rgb.shape[1] - 1], xtrms_format(data.longitude.values))

    plt.title(title, weight='bold', fontsize=16)
    
    fig.patch.set_alpha(1) # opaque white border
    fig.tight_layout()
    if fig_name:
        plt.savefig(fig_name, dpi=150)
        display(HTML("""<a href="{}" target="_blank" >View and download {}</a>""".format(fig_name, basename(fig_name))))
    else:
        plt.show()
    plt.close()
    
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

# source: https://code.activestate.com/recipes/578267-use-pil-to-make-a-contact-sheet-montage-of-images/
def make_contact_sheet(fnames, ncols = 0 ,nrows = 0 , photow = 0, photoh = 0, by = 'row',
                       title = None, font = 'LiberationSans-Bold.ttf', size = 14,
                       fig_name = None):
    """
    Description:
      Make a contact sheet from a group of images filenames.
    -----
    Input:
      fnames (OPTIONAL): a list of names of the image files
      ncols, nrows (OPTIONAL): number of columns OR rows in the contact sheet
      photow, photoh (OPTIONAL): width OR eight of the photo thumbs in pixels
      by (OPTIONAL): images displayed by row (default) or columns
      title (OPTIONAL): optional title
      font (OPTIONAL): title font (default LiberationSans bold)
      size (OPTIONAL): title font size (default 14)
      fig_name (OPTIONAL): file name (including extension) to save the figure (show only if not added to input).      
    Output:
      figure.
    """
    
    assert ncols * nrows == 0 and ncols + nrows > 0, '! You need to specify <ncols> OR <nrows>'
    assert photow * photoh == 0 and photow + photoh > 0, '! You need to specify <photow> OR <photoh>'
    assert by in ('row, col'), "! <by> can only be 'row' or 'col' !"
    
    if ncols > 0:
        nrows = ceil(len(fnames)/ncols)
    else:
        ncols = ceil(len(fnames)/nrowss)
        
    # get first photo size
    pxw, pxh = Image.open(fnames[0]).size
    if photow > 0:
        photoh = ceil(photow * pxh / pxw)
    else:
        photow = ceil(photoh * pxw / pxx)
    
    # Calculate the size of the output image, based on the
    #  photo thumb sizes, margins, and padding
    marl, marr, mart, marb = 5,5,5,5 # hardcoded margins
    padding = 1                      # hardcoded padding
    
    if title:
        try:
            font = ImageFont.truetype(font, size=size)
        except:
            print('! {} font is not availble, run !fc-list to find one !'.format(font))
            sys.exit
        mart += size
    
    marw = marl+marr
    marh = mart+ marb
    padw = (ncols-1)*padding
    padh = (nrows-1)*padding
    isize = (ncols*photow+marw+padw,nrows*photoh+marh+padh)
    
    # Create the new image. The background doesn't have to be white
    white = (255,255,255)
    inew = Image.new('RGB',isize,white)
    
    # reshape <fnames> if required
    if by == 'col':
        # append nans to get a proper fnames length
        ns = [np.nan] * (ncols * nrows - len(fnames))
        fnames += ns
        fnames = np.reshape(fnames, (ncols, nrows)).T.flatten()

    count = 0
    # Insert each thumb:
    for irow in range(nrows):
        for icol in range(ncols):
            left = marl + icol*(photow+padding)
            right = left + photow
            upper = mart + irow*(photoh+padding)
            lower = upper + photoh
            bbox = (left,upper,right,lower)
            try:
                # Read in an image and resize appropriately
                img = Image.open(fnames[count]).resize((photow,photoh))
            except:
                break
            inew.paste(img,bbox)
            count += 1
    
    if title:
        d = ImageDraw.Draw(inew)
        w, h = d.textsize(title)
        d.text(((isize[0] - w)/2, 5), title, fill='black', font = font)
    
    if fig_name:
        inew.save(fig_name)
        display(HTML("""<a href="{}" target="_blank" >View and download {}</a>""".format(fig_name, basename(fig_name))))
    else:
        return inew
