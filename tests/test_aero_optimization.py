from __future__ import annotations

import numpy as np
import xarray as xr
import dask
from xregrid.utils import create_grid_like, create_global_grid


class ComputeCounter(dask.callbacks.Callback):
    def __init__(self):
        self.count = 0

    def _start(self, dsk):
        self.count += 1


def test_create_grid_like_optimization_rectilinear():
    """
    Verify that create_grid_like for rectilinear grids (1D dim coords)
    does not trigger hidden computes for Dask-backed objects.
    """
    # 1. Create a source grid
    ds_src = create_global_grid(res_lat=1.0, res_lon=1.0)
    # Add a dummy lazy variable to make the dataset "lazy"
    ds_src["dummy"] = (("lat", "lon"), dask.array.zeros((180, 360), chunks=(90, 180)))

    # In Xarray, dimension coordinates are eager, but we want to ensure
    # create_grid_like uses the indexes instead of triggering computes
    # if it were to accidentally call .values or .compute() on the coordinates.

    # 2. Call create_grid_like with a counter
    counter = ComputeCounter()
    with counter:
        ds_new = create_grid_like(ds_src, res=2.0)

    # For rectilinear grids with dimension coordinates, count should be 0
    # because we use ds.indexes which is eager in Xarray.
    assert counter.count == 0

    # 3. Verify extent
    assert ds_new.lat.size == 90
    assert ds_new.lon.size == 180
    assert np.allclose(ds_new.lat.min(), -89.0)
    assert np.allclose(ds_new.lat.max(), 89.0)


def test_create_grid_like_curvilinear_heuristic():
    """
    Verify that create_grid_like for curvilinear grids (2D coords)
    triggers a compute but hopefully just for edges/samples (verified by result).
    """
    # Create a dummy curvilinear grid
    # 0 to 10 with 11 points means points at 0, 1, 2, ..., 10.
    # Resolution is 1.0.
    lon, lat = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
    ds_src = xr.Dataset(
        coords={
            "lat": (["y", "x"], lat, {"units": "degrees_north"}),
            "lon": (["y", "x"], lon, {"units": "degrees_east"}),
        }
    )
    ds_src = ds_src.chunk({"x": 5, "y": 5})

    # Call create_grid_like
    # This should trigger compute on edges because it's 2D lazy
    ds_new = create_grid_like(ds_src, res=1.0)

    # Verify extent
    # Extent detected from centers [0, 10] with res 1.0 will be [-0.5, 10.5]
    # New grid with res 1.0: centers at 0, 1, 2, ..., 10. (11 points)
    assert ds_new.lat.size == 11
    assert ds_new.lon.size == 11
    assert np.allclose(ds_new.lat.min(), 0.0)
    assert np.allclose(ds_new.lat.max(), 10.0)


def test_create_grid_like_consistency_eager_lazy():
    """
    Verify that create_grid_like returns identical results for Eager and Lazy backends.
    """
    # Eager source
    ds_eager = create_global_grid(res_lat=10.0, res_lon=10.0)

    # Lazy source
    ds_lazy = ds_eager.chunk({"lat": 9, "lon": 18})

    # Apply
    res_eager = create_grid_like(ds_eager, res=5.0)
    res_lazy = create_grid_like(ds_lazy, res=5.0)

    xr.testing.assert_identical(res_eager, res_lazy)
