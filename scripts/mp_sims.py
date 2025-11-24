# general
import numpy as np
from numpy.random import PCG64, SeedSequence
import xarray as xr
from tqdm.auto import tqdm
from cmcrameri import cm
import multiprocessing as mp

# plotting
import matplotlib.pyplot as plt

# io
from tqdm.auto import tqdm
import os
from pathlib import Path
import time
import numbers
import warnings

import sys
sys.path.append('..')

import gstatsim_custom as gsim

if __name__=='__main__':

    ds = xr.open_dataset(Path('../processed_data/bedmap3_mod_1000.nc'))

    # add exposed bedrock to conditioning data
    thick_cond = np.where(ds.mask.values == 4, 0, ds.thick_cond.values)
    
    bed_cond = ds.surface_topography.values - thick_cond
    ice_rock_msk = (ds.mask.values == 1) | (ds.mask.values == 4) | (ds.mask.values == 2)
    bed_cond = np.where(ice_rock_msk, bed_cond, np.nan)
    xx, yy = np.meshgrid(ds.x, ds.y)
    
    cond_msk = ~np.isnan(bed_cond)
    trend = ds.trend.values

    res_cond = bed_cond - trend

    res_norm, nst_trans = gsim.utilities.gaussian_transformation(res_cond, cond_msk)

    dsv = xr.load_dataset(Path('../processed_data/continental_variogram_1000.nc'))
    
    vario = {
        'azimuth' : dsv.azimuth.values,
        'nugget' : 0,
        'major_range' : dsv.major_range.values,
        'minor_range' : dsv.minor_range.values,
        'sill' : dsv.sill.values,
        's' : dsv.smooth.values,
        'vtype' : 'matern',
    }
    
    rng = np.random.default_rng(0)
    k = 24
    rad = 50e3

    # print('starting simulation')
    # tic = time.time()
    # sim = gsim.interpolate.sgs(xx, yy, res_cond, vario, rad, k, seed=rng, sim_mask=ice_rock_msk, rcond=1e-4)
    # toc = time.time()
    # print(f'{toc-tic} seconds')

    # bound the bed to be below the surface
    bounds = (-9999, ds.surface_topography.values - trend)

    n_sims = 4
    n_workers = 4

    params = []
    for i in range(n_sims):
        params.append([xx, yy, res_cond, vario, rad, k, 'ok', ice_rock_msk, True, None, 1e-4, bounds, None])

    tic = time.time()
    
    with mp.Pool(n_workers) as pool:
        result = pool.starmap(gsim.interpolate.sgs, params)

    for i, sim in enumerate(result):
        np.save(Path(f'../results/nonstationary_sim_1000_{i}.npy'), sim)

    toc = time.time()
    print(f'{toc-tic} seconds')

    ilow = 1000
    ihigh = 5800
    jlow = 700
    jhigh = 6200
    
    xx_trim = xx[ilow:ihigh,jlow:jhigh]
    yy_trim = yy[ilow:ihigh,jlow:jhigh]
    sim_trim = sim[ilow:ihigh,jlow:jhigh]
    trend_trim = trend[ilow:ihigh,jlow:jhigh]

    plt.figure(figsize=(13,10))
    im = plt.pcolormesh(xx_trim/1000, yy_trim/1000, sim_trim+trend_trim, cmap=cm.batlowW)
    plt.axis('scaled')
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('SGS simulation+trend, matern covariance, nonstationary, anisotropic')
    plt.colorbar(im, pad=0.03, aspect=40, shrink=0.7, label='bed elevation [meters]')
    # plt.savefig(Path('../figures/full_simulation_nonstationary_1000.png'), dpi=300, bbox_inches='tight')
    plt.show()