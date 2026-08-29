from __future__ import annotations

import numpy as np
import xarray as xr

from xregrid.grid import _get_mesh_info
from xregrid.utils import is_lazy


def test_separable_2d_grid_eager_and_lazy():
    """
    Test 2D separable grid detection in _get_mesh_info for both Eager and Lazy data.

    A 2D separable grid (regular lat/lon grid stored in 2D mesh form) should be
    detected and converted to 1D rect-linear form without evaluating full lazy arrays
    or breaking laziness.
    """
    # Create a 2D separable grid (lat varies along dim 0, lon along dim 1)
    lats_1d = np.linspace(-45.0, 45.0, 10)
    lons_1d = np.linspace(0.0, 360.0, 20)
    lon_2d, lat_2d = np.meshgrid(lons_1d, lats_1d)

    # 1. Eager NumPy Dataset
    ds_eager = xr.Dataset(
        coords={
            "lat": (("y", "x"), lat_2d),
            "lon": (("y", "x"), lon_2d),
        }
    )

    lon_eager, lat_eager, shape_eager, dims_eager, is_unstruc_eager = _get_mesh_info(
        ds_eager
    )

    assert not is_unstruc_eager
    assert shape_eager == (10, 20)
    assert dims_eager == ("y", "x")

    # 2. Lazy Dask Dataset
    ds_lazy = ds_eager.chunk({"y": 5, "x": 10})
    assert is_lazy(ds_lazy["lat"])
    assert is_lazy(ds_lazy["lon"])

    lon_lazy, lat_lazy, shape_lazy, dims_lazy, is_unstruc_lazy = _get_mesh_info(ds_lazy)

    assert not is_unstruc_lazy
    assert shape_lazy == (10, 20)
    assert dims_lazy == ("y", "x")

    # Verify that returned mesh arrays retain dask backend for lazy input
    assert is_lazy(lat_lazy)
    assert is_lazy(lon_lazy)

    # Numerical equivalence between eager and lazy output values
    np.testing.assert_allclose(lat_eager.values, lat_lazy.compute().values)
    np.testing.assert_allclose(lon_eager.values, lon_lazy.compute().values)
