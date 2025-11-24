import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import verde as vd
from pathlib import Path
import os
import time
from tqdm.auto import tqdm
import geopandas as gpd
from numba import njit, prange
from numba_progress import ProgressBar

import warnings
warnings.filterwarnings("ignore")

from utilities import *

rename_dict = {
    'surface_altitude (m)' : 'surface',
    'land_ice_thickness (m)' : 'thickness',
    'bedrock_altitude (m)' : 'bed',
    'longitude (degree_east)' : 'lon',
    'latitude (degree_north)' : 'lat'
}

def collect_files(path, verbose=False):
    total = 0
    for item in os.scandir(path):
        if item.name.endswith('.csv'):
            total += 1
    
    dfs = []
    
    for i, item in tqdm(enumerate(os.scandir(path)), total=total):
        if item.name.endswith('.csv'):
            if verbose==True:
                print(item.path)
            dfs.append(pd.read_csv(item.path, header=18))

    if len(dfs)==1:
        df = dfs[0]
    else:
        df = pd.concat(dfs)
    df = df.rename(columns=rename_dict)
    df = df[['surface', 'thickness', 'bed', 'lat', 'lon']]
    df = df.replace(-9999, np.nan)
    
    return df

@njit(parallel=True)
def block_reduce_jit(xx, bins, thickness, rands, progress_proxy):
    
    counts = np.zeros(xx.shape)
    mean = np.full(xx.shape, np.nan)
    median = np.full(xx.shape, np.nan)
    stdev = np.full(xx.shape, np.nan)
    median_partial = np.full(xx.shape, np.nan)
    
    xb_uniq = np.unique(bins[:,0])
    yb_uniq = np.unique(bins[:,1])
    
    for i in prange(xb_uniq.size):
        xbin = xb_uniq[i]
        inds1 = bins[:,0]==xbin
        bins1 = bins[inds1,:]
        thick1 = thickness[inds1]
        rands1 = rands[inds1]
        for j in range(yb_uniq.size):
            ybin = yb_uniq[j]
            inds2 = bins1[:,1]==ybin
            thick2 = thick1[inds2]
            rands2 = rands1[inds2]
            
            count = thick2.size
            if count > 0:
                counts[ybin,xbin] = count
                mean[ybin,xbin] = np.mean(thick2)
                median[ybin,xbin] = np.nanquantile(thick2, 0.5)
                stdev[ybin,xbin] = np.std(thick2)
                if count > 1:
                    sample = thick2[rands2 > 0.5]
                    if sample.size==0:
                        median_partial[ybin,xbin] = np.nanquantile(thick2, 0.5)
                    else:
                        median_partial[ybin,xbin] = np.nanquantile(sample, 0.5)
                else:
                    median_partial[ybin,xbin] = thick2[0]
        progress_proxy.update(1)
        
    return (counts, mean, median, stdev, median_partial)

tic = time.time()

bm1path = Path('D:/bedmap/BEDMAP1')
bm2path = Path('D:/bedmap/BEDMAP2')
bm3path = Path('D:/bedmap/BEDMAP3')
bmgrid_path = Path('D:/bedmap/bedmap3.nc')
bmach_path = Path('D:/bedmachine/BedMachineAntarctica-v3.nc')
stream_path = Path('D:/bedmap/bm3_streamlines_pt/bm3_streamlines_pt.shp')

print('collecting files')
bm1 = collect_files(bm1path)
bm2 = collect_files(bm2path)
bm3 = collect_files(bm3path)

df = pd.concat([bm1, bm2, bm3])

del(bm1)
del(bm2)
del(bm3)

x_coords, y_coords = geo2ant(df['lat'], df['lon'])
df['x'] = x_coords
df['y'] = y_coords

msk = (df['thickness'].isna()==True) & (df['surface'].isna()==False) & (df['bed'].isna()==False)
df.loc[msk, 'thickness'] = df.loc[msk, 'surface'] - df.loc[msk, 'bed']

df = df.loc[df['thickness'].isna()==False]

# add new COLDEX data
coldex = pd.read_csv(Path('D:/bedmap/2023_Antarctica_BaslerMKB.csv'))

x_coords, y_coords = geo2ant(coldex['LAT'], coldex['LON'])
coldex['x'] = x_coords
coldex['y'] = y_coords

coldex = coldex.rename(columns={'LAT' : 'lat', 'LON' : 'lon', 'THICK' : 'thickness', 'SURFACE' : 'surface', 'BOTTOM' : 'bed'})
coldex = coldex[['surface', 'thickness', 'bed', 'lat', 'lon', 'x', 'y']]

df = pd.concat([df, coldex])

ds = xr.open_dataset(bmgrid_path)
xx, yy = np.meshgrid(ds.x, ds.y)

### Custom block reduction
xbin_edges = ds.x.values - 250
xbin_edges = np.append(xbin_edges, ds.x.values[-1]+250)

ybin_edges = ds.y.values - 250
ybin_edges = np.append(ybin_edges, ds.y.values[-1]-750)

xbins = np.digitize(df.x.values, xbin_edges)
ybins = np.digitize(df.y.values, ybin_edges)

# do block reduction
bins = np.stack([xbins, ybins]).T
thickness = df.thickness.values
rng = np.random.default_rng(0)
rands = rng.random(size=thickness.size)

print('doing block reduction')
with ProgressBar(total=np.unique(xbins).size) as progress:
    result = block_reduce_jit(xx, bins, thickness, rands, progress)

counts = result[0]
median = result[2]
stdev = result[3]

bad_msk = (stdev>300) | (np.abs(ds.ice_thickness.values-median)>300)
x_bad = xx[bad_msk]
y_bad = yy[bad_msk]

print(f'{x_bad.size:,} bad points, {x_bad.size/np.count_nonzero(counts>0)*100:.2f}% of conditioning data')

# extract results
counts = np.where(bad_msk, np.nan, result[0])
mean = np.where(bad_msk, np.nan, result[1])
median = np.where(bad_msk, np.nan, result[2])
stdev = np.where(bad_msk, np.nan, result[3])
median_sample = np.where(bad_msk, np.nan, result[4])

plt.scatter(x_bad, y_bad, s=0.5)
plt.axis('scaled')
plt.xlabel('X [km]')
plt.ylabel('Y [km]')
plt.title('Bad grid cells')
plt.savefig(Path('../figures/bad_grid_cells.png'), dpi=300, bbox_inches='tight')

thick_cond = median

# Interpolate BedMachine geoid onto Bedmap3
bmach = xr.open_dataset(bmach_path)

xx_bmach, yy_bmach = np.meshgrid(bmach.x, bmach.y)

linear = vd.KNeighbors(k=1)
linear.fit((xx_bmach, yy_bmach), bmach.geoid.values)
preds = linear.predict((xx, yy))

ds['geoid'] = (('y', 'x'), preds)
ds['thick_cond'] = (('y', 'x'), thick_cond)

ice_rock_msk = (ds.mask == 1) | (ds.mask == 2)
print(f'{np.count_nonzero(np.isnan(ds.thick_cond.values) & ice_rock_msk):,} grid cells to simulate at 500 m resolution')

# Bedmap3 streamline ice thickness
print('getting streamlines')
pts = gpd.read_file(stream_path)

coords = pts.get_coordinates()
coords = (np.rint(coords['x'].values), np.rint(coords['y'].values))
thick = pts.thick.values

stream_thick = xy_into_grid(ds.x.values, ds.y.values, coords, thick)

ds['stream_thick'] = (('y', 'x'), stream_thick)

# create trend
print('creating trend')

bed_cond = ds.surface_topography.values - ds.thick_cond.values
xx, yy = np.meshgrid(ds.x, ds.y)

cond_msk = ~np.isnan(bed_cond)
x_cond = xx[cond_msk]
y_cond = yy[cond_msk]
data_cond = bed_cond[cond_msk]

cond_coords = np.array([x_cond[::1000], y_cond[::1000]]).T
trend = spline_interp_msk(cond_coords, data_cond[::1000], xx, yy, ice_rock_msk, damping=1e-5)

res_cond = bed_cond - trend

ds['trend'] = (('y', 'x'), trend)

# save to netCDF
ds.to_netcdf(Path('../processed_data/bedmap3_mod_500.nc'))

# coarsen to 1 km resolution
print('coarsening to 1 km')
ds = ds.coarsen(x=2, y=2, boundary='trim').median()
ds['mask'] = (('y', 'x'), np.where(np.isnan(ds.mask.values), np.nan, np.rint(ds.mask.values).astype(int)))

ds['bed_topography'] = (('y', 'x'), np.where(ds.mask.values==4, ds.surface_topography.values, ds.bed_topography.values))

ice_rock_msk = (ds.mask == 1) | (ds.mask == 2)
print(f'{np.count_nonzero(np.isnan(ds.thick_cond.values) & ice_rock_msk):,} grid cells to simulate at 1 km resolution')

ds.to_netcdf(Path('../processed_data/bedmap3_mod_1000.nc'))

toc = time.time()

print(f'time elapsed: {toc-tic}')

fig, axs = plt.subplots(1, 2, figsize=(14,5), sharey=True)
ax = axs[0]
im = ax.imshow(trend, extent=(ds.x.values.min(), ds.x.values.max(), ds.y.values.min(), ds.y.values.max()))
ax.axis('scaled')
plt.colorbar(im, ax=ax, pad=0.03, aspect=40)
ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_title('Trend')

ax = axs[1]
im = ax.imshow(res_cond, extent=(ds.x.values.min(), ds.x.values.max(), ds.y.values.min(), ds.y.values.max()))
ax.axis('scaled')
plt.colorbar(im, ax=ax, pad=0.03, aspect=40)
ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_title('Residual')
plt.savefig(Path('../figures/trend_residual.png'), dpi=300, bbox_inches='tight')
plt.show()