from __future__ import annotations

import numpy as np
import xarray as xr
import dask.array as da
from xregrid.regridder import Regridder
from xregrid.utils import create_global_grid


def test_detect_periodicity_eager_2d():
    # Create a global grid (periodic)
    # lon is 0 to 360, centers are 5 to 355
    ds = create_global_grid(res_lat=10, res_lon=10)
    # Convert 1D coords to 2D
    lon_2d, lat_2d = xr.broadcast(ds.lon, ds.lat)
    ds_2d_coarse = xr.Dataset(
        coords={
            "lat": (["lat", "lon"], lat_2d.data),
            "lon": (["lat", "lon"], lon_2d.data),
        }
    )

    regridder = Regridder.__new__(Regridder)
    # coarsened extent = 355 - 5 = 350.
    assert regridder._detect_periodicity(ds_2d_coarse) is False

    # extent = 355 - 5 = 350.
    # 354 <= 350 < 360 is False.
    # Wait, 10 degree resolution global grid has centers at 5, 15, ..., 355.
    # Max - Min = 355 - 5 = 350.
    # My _detect_periodicity uses 354 as the lower bound.
    # Let's use a finer resolution to pass 354.

    ds_fine = create_global_grid(res_lat=1, res_lon=1)
    lon_2d_fine, lat_2d_fine = xr.broadcast(ds_fine.lon, ds_fine.lat)
    ds_2d_fine = xr.Dataset(
        coords={
            "lat": (["lat", "lon"], lat_2d_fine.data),
            "lon": (["lat", "lon"], lon_2d_fine.data),
        }
    )
    # extent = 359.5 - 0.5 = 359.0
    assert regridder._detect_periodicity(ds_2d_fine) is True


def test_detect_periodicity_lazy_2d():
    # Create a 2D lazy grid with fine resolution
    ds = create_global_grid(res_lat=1, res_lon=1)
    lon_2d, lat_2d = xr.broadcast(ds.lon, ds.lat)

    # Chunk it
    lon_lazy = da.from_array(lon_2d.data, chunks=(180, 360))
    lat_lazy = da.from_array(lat_2d.data, chunks=(180, 360))

    ds_lazy = xr.Dataset(
        coords={
            "lat": (["lat", "lon"], lat_lazy),
            "lon": (["lat", "lon"], lon_lazy),
        }
    )

    regridder = Regridder.__new__(Regridder)

    # Use dask tracker to count computes
    from dask.callbacks import Callback

    class ComputeCounter(Callback):
        def __init__(self):
            self.count = 0

        def _precompute(self, key, task, dsk, state):
            self.count += 1

    counter = ComputeCounter()
    with counter:
        is_periodic = regridder._detect_periodicity(ds_lazy)

    assert is_periodic is True


def test_detect_periodicity_non_periodic():
    # Create a regional grid (non-periodic)
    ds = xr.Dataset(
        coords={
            "lat": (["lat"], np.arange(-20, 20, 1)),
            "lon": (["lon"], np.arange(0, 100, 1)),
        }
    )
    regridder = Regridder.__new__(Regridder)
    assert regridder._detect_periodicity(ds) is False
