from __future__ import annotations

import xarray as xr
from xregrid.utils import create_global_grid, create_grid_from_crs, is_dask


def test_create_global_grid_parity():
    """Verify parity between eager and lazy global grid creation."""
    # 1. Eager (NumPy)
    ds_eager = create_global_grid(res_lat=10, res_lon=10, add_bounds=True)

    # 2. Lazy (Dask)
    ds_lazy = create_global_grid(res_lat=10, res_lon=10, add_bounds=True, chunks=10)

    assert is_dask(ds_lazy)
    assert not is_dask(ds_eager)

    # Check parity of coordinates
    xr.testing.assert_allclose(ds_eager.lat, ds_lazy.lat.compute())
    xr.testing.assert_allclose(ds_eager.lon, ds_lazy.lon.compute())
    xr.testing.assert_allclose(ds_eager.lat_b, ds_lazy.lat_b.compute())
    xr.testing.assert_allclose(ds_eager.lon_b, ds_lazy.lon_b.compute())


def test_create_grid_from_crs_parity():
    """Verify parity between eager and lazy projected grid creation."""
    crs = "EPSG:3857"
    extent = (-1000000, 1000000, -1000000, 1000000)
    res = 500000

    # 1. Eager
    ds_eager = create_grid_from_crs(crs, extent, res, add_bounds=True)

    # 2. Lazy
    ds_lazy = create_grid_from_crs(crs, extent, res, add_bounds=True, chunks=2)

    assert is_dask(ds_lazy)

    # Check parity
    xr.testing.assert_allclose(ds_eager.x, ds_lazy.x.compute())
    xr.testing.assert_allclose(ds_eager.y, ds_lazy.y.compute())
    xr.testing.assert_allclose(ds_eager.lat, ds_lazy.lat.compute())
    xr.testing.assert_allclose(ds_eager.lon, ds_lazy.lon.compute())

    if "lat_b" in ds_eager:
        xr.testing.assert_allclose(ds_eager.lat_b, ds_lazy.lat_b.compute())
        xr.testing.assert_allclose(ds_eager.lon_b, ds_lazy.lon_b.compute())

    if "x_b" in ds_eager:
        xr.testing.assert_allclose(ds_eager.x_b, ds_lazy.x_b.compute())
        xr.testing.assert_allclose(ds_eager.y_b, ds_lazy.y_b.compute())


def test_provenance_tracking():
    """Verify that history attribute is updated."""
    ds = create_global_grid(res_lat=10, res_lon=10)
    assert "history" in ds.attrs
    assert "xregrid" in ds.attrs["history"]


def test_compute_lazy_aware_parity():
    """Verify _compute_lazy_aware on Eager (NumPy) vs Lazy (Dask) data structures."""
    import dask.array as da
    import numpy as np
    from xregrid.utils import _compute_lazy_aware

    # Eager structure
    eager_data = {
        "a": np.array([1.0, 2.0, 3.0]),
        "b": (np.array([4.0, 5.0]), 10.0),
        "c": [np.array([6.0])],
    }

    # Lazy structure
    lazy_data = {
        "a": da.from_array(np.array([1.0, 2.0, 3.0]), chunks=2),
        "b": (da.from_array(np.array([4.0, 5.0]), chunks=1), 10.0),
        "c": [da.from_array(np.array([6.0]), chunks=1)],
    }

    res_eager = _compute_lazy_aware(eager_data)
    res_lazy = _compute_lazy_aware(lazy_data)

    np.testing.assert_allclose(res_eager["a"], res_lazy["a"])
    np.testing.assert_allclose(res_eager["b"][0], res_lazy["b"][0])
    assert res_eager["b"][1] == res_lazy["b"][1]
    np.testing.assert_allclose(res_eager["c"][0], res_lazy["c"][0])


def test_create_grid_like_parity():
    """Verify create_grid_like on Eager (NumPy) vs Lazy (Dask) template objects."""
    import numpy as np
    from xregrid.utils import create_grid_like

    # 1. Eager template
    ds_template = create_global_grid(res_lat=10, res_lon=10)
    da_eager = xr.DataArray(
        np.ones((18, 36)),
        dims=("lat", "lon"),
        coords={"lat": ds_template.lat, "lon": ds_template.lon},
    )

    # 2. Lazy template
    da_lazy = da_eager.chunk({"lat": 6, "lon": 6})

    grid_eager = create_grid_like(da_eager, res=5)
    grid_lazy = create_grid_like(da_lazy, res=5)

    xr.testing.assert_allclose(grid_eager.lat, grid_lazy.lat)
    xr.testing.assert_allclose(grid_eager.lon, grid_lazy.lon)
