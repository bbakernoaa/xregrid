# Accessors

XRegrid provides xarray accessors for both `DataArray` and `Dataset` objects, allowing you to perform regridding using a convenient `.regrid.to()` syntax.

## DataArray Accessor

### to

::: xregrid.accessors.RegridDataArrayAccessor.to

Regrid the DataArray to a target grid.

```python
import xarray as xr
from xregrid import create_global_grid

# Load some data
da = xr.tutorial.open_dataset("air_temperature").air

# Define target grid
target_grid = create_global_grid(res_lat=1.0, res_lon=1.0)

# Regrid using the accessor
regridded_da = da.regrid.to(target_grid, method='bilinear')
```

## Dataset Accessor

### to

::: xregrid.accessors.RegridDatasetAccessor.to

Regrid the Dataset to a target grid.

```python
import xarray as xr
from xregrid import create_global_grid

# Load some data
ds = xr.tutorial.open_dataset("air_temperature")

# Define target grid
target_grid = create_global_grid(res_lat=1.0, res_lon=1.0)

# Regrid using the accessor
regridded_ds = ds.regrid.to(target_grid, method='bilinear')
```
