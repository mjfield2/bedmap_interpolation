import numpy as np
from pyproj import Transformer
from tqdm.auto import tqdm
import verde as vd

def geo2ant(lat, lon):
    """
    Converts geodetic coordinates to Antarctic Polar Stereographic

    Args:
        lat, lon : latitude and longitude arrays
    Outputs:
        x, y : coordinates in Antarctic Polar Stereographic
    """
    transformer = Transformer.from_crs(4326, 3031)
    x, y = transformer.transform(lat, lon)
    return x, y

def xy_into_grid(gridx, gridy, coords, values, fill=np.nan, quiet=False):
    """
    Place tabular data that is colocated with grid coordinates inside of grid. UPDATE THIS

    Args:
        ds : xarray.Dataset with grid coordinates
        coords : tuple of tabular coordinates, e.g., (x, y)
        values : tuple of values correspondin to coords, e.g., (var1, var2)
        fill : value to fill in grid where there are no values
        quiet : show progress bar if False
    Outputs:
        if multiple arrays in values, returns a tuple of gridded values, 
        otherwise returns single gridded array
    """
    if gridx[0] > gridx[1]:
        gridx = np.sort(gridx)
        reversex = True
    else:
        reversex = False
    if gridy[0] > gridy[1]:
        gridy = np.sort(gridy)
        reversey = True
    else:
        reversey = False
    
    values = np.array(values)
    if len(values.shape)==1:
        values = np.expand_dims(values, axis=0)
        
    arr = np.full((len(values), len(gridy), len(gridx)), fill)
    
    for i in tqdm(range(len(coords[0])), disable=quiet):
        xi = coords[0][i]
        yi = coords[1][i]
        xind = np.searchsorted(gridx, xi)
        yind = np.searchsorted(gridy, yi)
        if (xind < len(gridx)) & (yind < len(gridy)):
            if (gridx[xind] == xi) & (gridy[yind] == yi):
                if reversex == True:
                    xind = len(gridx) - xind - 1
                if reversey == True:
                    yind = len(gridy) - yind - 1
                arr[:,yind, xind] = values[:,i].squeeze()
    if arr.shape[0]>1:
        return tuple(arr)
    else:
        return arr.squeeze()

def spline_interp_msk(points, values, xx, yy, mask, damping):
    sp = vd.Spline(damping=damping)
    sp.fit((points[:,0], points[:,1]), values)
    preds = sp.predict((xx[mask], yy[mask]))
    grid = np.full(xx.shape, np.nan)
    np.place(grid, mask, preds)
    return grid
