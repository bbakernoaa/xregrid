from __future__ import annotations

import numpy as np
import xarray as xr

from xregrid.grid import _get_mesh_info
from xregrid.utils import is_lazy


def test_separable_grid_eager_and_lazy():
    """Verify separable 2D grid detection works for eager data and preserves laziness for dask data without hidden computes."""
    # 1. Create a 2D regular grid stored as 2D curvilinear lat/lon arrays with CF standard_names
    y = np.arange(10)
    x = np.arange(20)
    lat_1d = np.linspace(-90, 90, 10)
    lon_1d = np.linspace(0, 360, 20)
    lat_2d = np.tile(lat_1d[:, None], (1, 20))
    lon_2d = np.tile(lon_1d[None, :], (10, 1))

    ds_eager = xr.Dataset(
        data_vars={"temp": (("y", "x"), np.ones((10, 20)))},
        coords={
            "y": y,
            "x": x,
            "lat": (
                ("y", "x"),
                lat_2d,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "lon": (
                ("y", "x"),
                lon_2d,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
        },
    )

    # Run on eager dataset - should detect separable grid and reduce to 1D broadcast
    lon_e, lat_e, shape_e, dims_e, unstruct_e = _get_mesh_info(ds_eager)
    assert not unstruct_e
    assert shape_e == (10, 20)
    assert dims_e == ("y", "x")

    # 2. Chunk dataset to test lazy path - must preserve laziness and avoid hidden compute
    ds_lazy = ds_eager.chunk({"y": 5, "x": 10})
    lon_l, lat_l, shape_l, dims_l, unstruct_l = _get_mesh_info(ds_lazy)
    assert not unstruct_l
    assert shape_l == (10, 20)
    assert dims_l == ("y", "x")

    # Verify output coordinates remain lazy (Dask-backed)
    assert is_lazy(lat_l)
    assert is_lazy(lon_l)

    # Parity check on computed values
    np.testing.assert_allclose(lat_e.values, lat_l.values)
    np.testing.assert_allclose(lon_e.values, lon_l.values)


def test_non_separable_curvilinear_grid():
    """Verify non-separable curvilinear grid with dimension coordinates is correctly identified for eager and lazy data."""
    y_coords = np.arange(10)
    x_coords = np.arange(20)
    y, x = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 20), indexing="ij")
    # Add curvature so it's not separable
    lat_2d = y + 0.1 * x
    lon_2d = x + 0.1 * y

    ds_eager = xr.Dataset(
        data_vars={"temp": (("y", "x"), np.ones((10, 20)))},
        coords={
            "y": y_coords,
            "x": x_coords,
            "lat": (
                ("y", "x"),
                lat_2d,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "lon": (
                ("y", "x"),
                lon_2d,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
        },
    )

    lon_e, lat_e, shape_e, dims_e, unstruct_e = _get_mesh_info(ds_eager)
    assert not unstruct_e
    assert shape_e == (10, 20)
    # Check that lat_e remains 2D curvilinear (not collapsed to 1D separable)
    assert lat_e.ndim == 2

    ds_lazy = ds_eager.chunk({"y": 5, "x": 10})
    lon_l, lat_l, shape_l, dims_l, unstruct_l = _get_mesh_info(ds_lazy)
    assert not unstruct_l
    assert shape_l == (10, 20)
    assert lat_l.ndim == 2

    # Verify laziness is preserved
    assert is_lazy(lat_l)
    assert is_lazy(lon_l)

    np.testing.assert_allclose(lat_e.values, lat_l.values)
    np.testing.assert_allclose(lon_e.values, lon_l.values)
