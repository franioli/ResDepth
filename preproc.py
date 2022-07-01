import numpy as np
import rasterio

from pathlib import Path
import os
import matplotlib.pyplot as plt
# import matplotlib
# %matplotlib 

from rasterio.mask import mask
import fiona
from  shapely.geometry import (LinearRing, Point, Polygon, mapping) 
from rasterio.plot import show as rshow
from rasterio.transform import Affine

def norm_img(img):
    img = img / 65535.
    return np.float32(img)


# Params 
root_path = Path('belv/preproc')
out_dir = Path('out')

# DS1 
# ds = '1'
# xlim = [-57.5, 11.]
# ylim = [-30.5, -3.5]
# X0 = 0. 

# DS2
# ds = '2'
# xlim = [-6.5, 45.5]
# ylim = [-29.5, -6.5]
# X0 = 0. 

# DS3
# ds = '3'
# xlim = [-103.000, 13.000]
# ylim = [-47.500, -19.500]
# X0 = 0. 

# DS4 - nadir
ds = '4'
xlim = [416426.000, 416504.000]
ylim = [5091039.000, 5091123.000]
X0 = 2000. 

#- 
ds_path = Path(ds)
if not (root_path/ds_path/out_dir).is_dir():
    os.mkdir(root_path/ds_path/out_dir)

poly = Polygon([(xlim[0],ylim[0]), (xlim[0],ylim[1]), (xlim[1],ylim[1]), (xlim[1],ylim[0])])
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
for i, fname in enumerate(fnames):
    with rasterio.open(fname, 'r') as dataset:     
        out_image, out_transform = mask(dataset, shapes, crop=True)
        out_name = Path.joinpath(root_path, ds_path, out_dir, Path(fname).stem+f'_cut_{ds}.tif')
        out_meta = dataset.meta
        out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
        print(out_transform)
        with rasterio.open(out_name, "w", **out_meta) as dest:
            dest.write(norm_img(out_image))

# DSM
fnames = list((root_path / ds_path).glob('dsm*.tif'))
for i, fname in enumerate(fnames):
    with rasterio.open(fname, 'r') as dataset:     
        out_image, out_transform = mask(dataset, shapes, crop=True)
        out_name = Path.joinpath(root_path, ds_path, out_dir, Path(fname).stem+f'_{ds}.tif')
        out_meta = dataset.meta
        out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
        print(out_transform)
        out_image = X0-out_image
        # out_image[out_image == 34767] = 120.
        # out_image[out_image == 0] = 120.
        with rasterio.open(out_name, "w", **out_meta) as dest:
            dest.write(out_image)


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
ds_path = Path('belv/preproc/4/out' )
fnames = ['DJI_0122_cut_4.tif','DJI_0424_cut_4.tif', 'dsm_gt_4.tif']
fnames = [ds_path/fname for fname in fnames] 

for i, fname in enumerate(fnames):
    with rasterio.open(fname, 'r') as dataset:    
        data = dataset.read(1)
        data = data[:-1,:]
        out_meta = dataset.meta
        out_meta.update({"driver": "GTiff",
                     "height": data.shape[0],
                     "width": data.shape[1], 
                     "count": 1,
                     "nodata": 0})
        with rasterio.open(fname, "w", **out_meta) as dest:
            dest.write(data, indexes=1)

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
