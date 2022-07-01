import numpy as np
import rasterio
from rasterio.mask import mask as Mask

from pathlib import Path
import os
import json
import matplotlib.pyplot as plt
# import matplotlib
# %matplotlib 

import fiona
from  shapely.geometry import (LinearRing, Point, Polygon, mapping) 
from rasterio.plot import show as rshow
from rasterio.transform import Affine

from scipy import interpolate
from PIL import Image

# Params 
root_path = Path('belv/preproc')
out_dir = Path('out')

#TODO: put parameters in json file
# DS1 
ds_1 = {"name": '1',
      "xlim": [-57.5, 11.],
      "ylim": [-30.5, -4], 
      "dsm_z0" : 0.,
      "inter_method": 'nearest'      
      }

# DS2
ds_2 = {"name": '2',
      "xlim": [-3.5, 44.5],
      "ylim": [-26.0, -5.0], 
      "dsm_z0" : 0. ,
      "inter_method": 'nearest'      
      }

# DS3
ds_3 = {"name": '3',
      "xlim": [-85.000, 10.000], 
      "ylim": [-47.500, -19.500], 
      "dsm_z0" : 0.,
      "inter_method": 'linear'      
      }

# DS4 - nadir
ds_4 = {"name": '4',
      "xlim": [416433.000, 416498.000],
      "ylim": [5091045.000, 5091117.000], 
      "dsm_z0" : 2000.,
      "inter_method": 'linear'      
      }

ds = ds_4


#-- Utils

def normalize_img(img, up_lim=65535.):
    img = img / up_lim
    return np.float32(img)

def convert_image_to_float(image):
    image_float = image / 256.
    return np.float32(image_float)

def convert_16bit_to_8bit(image):
    image_8bit = image / 256
    return image_8bit.astype('uint8')

def find_holes_image(image, nodata_val=65535.):
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if len(image.shape) > 2:
        channels = image.shape[2]
    else:
        channels = 1
    mask_channel = []
    for i in range(channels):
        mask_channel.append(image[:,:,i]==nodata_val)
    return mask_channel    
 
def find_holes_channel(image, nodata_val=np.NAN):
    mask = image==nodata_val
    return mask    
 
def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0.
    ):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value)

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image
    
def preproc_ortofoto(img, inter_method='nearest'):
    img_8bit = convert_16bit_to_8bit(img)
    image_gray = Image.fromarray(np.moveaxis(img_8bit, 0, -1)).convert('L')
    img = np.asarray(image_gray)
    mask = find_holes_channel(img, nodata_val=255)
    img_out = interpolate_missing_pixels(img, mask, method=inter_method)    
    return img_out

def preproc_dsm(dsm, dsm_z0=0., no_data=np.NAN):
    dsm = dsm[0,:]
    mask = find_holes_channel(dsm, no_data)
    dsm_interp = interpolate_missing_pixels(dsm, mask, method='nearest')    
    dsm_out = dsm_z0 - dsm_interp
    dsm_out = np.expand_dims(dsm_out, axis=0) 
    return dsm_out



#-- Cut orthos and DSM
ds_path = Path(str(ds['name']))
if not (root_path/ds_path/out_dir).is_dir():
    os.mkdir(root_path/ds_path/out_dir)

poly = Polygon([(ds['xlim'][0],ds['ylim'][0]), 
                (ds['xlim'][0],ds['ylim'][1]), 
                (ds['xlim'][1],ds['ylim'][1]), 
                (ds['xlim'][1],ds['ylim'][0])],
               )
schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'int'},
}
with fiona.open(root_path/ds_path/'bbox.shp', 'w', 'ESRI Shapefile', schema) as c:
    c.write({
        'geometry': mapping(poly),
        'properties': {'id': 123},
    })
with fiona.open(root_path/ds_path/'bbox.shp', "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]
    
# Images
fnames = list((root_path / ds_path).glob('DJI*.tif'))
# i = 0; fname = fnames[i]
for i, fname in enumerate(fnames):
    with rasterio.open(fname, 'r') as dataset:     
        crop_image, out_transform = Mask(dataset, shapes, crop=True)
        crop_image_out = preproc_ortofoto(crop_image, inter_method=ds["inter_method"])
        out_meta = dataset.meta
        out_meta.update({"driver": "GTiff",
                         "dtype": 'uint8',
                         "height": crop_image.shape[1],
                         "width": crop_image.shape[2],
                         "count": 1,                     
                         "transform": out_transform})
        print(out_transform)
        out_name = Path.joinpath(root_path, ds_path, out_dir, Path(fname).stem+f"_cut_{ds['name']}.tif")
        with rasterio.open(out_name, "w", **out_meta) as dest:
            dest.write(crop_image_out.astype(rasterio.uint8), indexes=1)

# DSM
fnames = list((root_path / ds_path).glob('dsm*.tif'))
for i, fname in enumerate(fnames):
    with rasterio.open(fname, 'r') as dataset:     
        crop_dsm, out_transform = Mask(dataset, shapes, crop=True)
        crop_dsm_out = preproc_dsm(crop_dsm, dsm_z0=ds['dsm_z0'], no_data=-32767.)
        out_meta = dataset.meta
        out_meta.update({"driver": "GTiff",
                         "dtype": rasterio.float32,
                         "height": crop_image.shape[1],
                         "width": crop_image.shape[2],
                         "count": 1,                     
                         "transform": out_transform})
        print(out_transform)
        # out_image[out_image == 34767] = 120.
        # out_image[out_image == 0] = 120.
        out_name = Path.joinpath(root_path, ds_path, out_dir, Path(fname).stem+f"_{ds['name']}.tif")
        with rasterio.open(out_name, "w", **out_meta) as dest:
            dest.write(crop_dsm_out.astype(rasterio.float32))


# Fix affine transformation rounding issue...         
tform = []
fout_names = list((root_path/ds_path/out_dir).glob('*.tif'))
for i, fname in enumerate(fout_names):
    with rasterio.open(fname, 'r') as dataset:   
        print(dataset.get_transform())
        tform.append(dataset.get_transform())
tform = np.array(tform)
tform_mean = tform.mean(0)
tform_std = tform.std(0)
print(f'tform std: {tform_std}')
res = np.round(tform_mean[1],2)
x, y = np.round(tform_mean[0],2), np.round(tform_mean[3],2)
transform = Affine.translation(x, y) * Affine.scale(res, -res)

for i, fname in enumerate(fout_names):
    with rasterio.open(fname, 'r+') as dataset:   
        dataset.transform = transform
        dataset.close()
    
#%%

# Fix DSM resolution
# ds_path = Path('belv/preproc/4/out' )
# fnames = ['DJI_0122_cut_4.tif','DJI_0424_cut_4.tif', 'dsm_gt_4.tif']
# fnames = [ds_path/fname for fname in fnames] 

# for i, fname in enumerate(fnames):
#     with rasterio.open(fname, 'r') as dataset:    
#         data = dataset.read(1)
#         data = data[:-1,:]
#         out_meta = dataset.meta
#         out_meta.update({"driver": "GTiff",
#                      "height": data.shape[0],
#                      "width": data.shape[1], 
#                      "count": 1,
#                      "nodata": 0})
#         with rasterio.open(fname, "w", **out_meta) as dest:
#             dest.write(data, indexes=1)


# out_image, out_transform = mask(dataset, shapes, crop=True)
# out_name = Path.joinpath(root_path, ds_path, out_dir, Path(fname).stem+'_cut.tif')
# out_meta = dataset.meta
# out_meta.update({"driver": "GTiff",
#              "height": out_image.shape[1],
#              "width": out_image.shape[2],
#              "transform": out_transform})
# print(out_transform)
# with rasterio.open(out_name, "w", **out_meta) as dest:
#     dest.write(-out_image)


#%% Tests
# interp_image = out_image.copy()
# for channel, data in enumerate(out_image):
#     mask = find_holes_channel(data)
#     interp_image[channel] = interpolate_missing_pixels(data, mask, method='nearest')

# plt.figure(0)    
# plt.imshow(np.moveaxis(normalize_img(out_image), 0, -1))

# plt.figure(1)
# plt.imshow(np.moveaxis(normalize_img(interp_image), 0, -1))

# img = convert_16bit_to_8bit(crop_image)
# img = crop_image.copy()
# img = np.moveaxis(img, 0, -1)
# image_pil = Image.fromarray(img).convert('L')
# # image_pil.show()

# img = np.asarray(image_pil)
# mask = find_holes_channel(img, nodata_val=255)
# interp_image = interpolate_missing_pixels(img, mask, method='nearest')

# fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
# ax1.imshow(img, cmap='gray')
# ax2.imshow(interp_image, cmap='gray')

# satellite = Image.open('15MAR17102414-P1BS-502980289080_01_P002.tif')
# satellite = np.asarray(satellite)



#%%

# from PIL import Image
# path = root_path / ds_path / out_dir /'DJI_0151_cut_1.tif'
# image = Image.open('belv/preproc/1/out/DJI_0151_cut_1.tif')
# img = np.asarray(image, dtype=('float32'))
# # img *= 255.
# plt.imshow(img)


