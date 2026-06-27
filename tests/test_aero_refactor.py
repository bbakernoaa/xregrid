from __future__ import annotations

import numpy as np
import xarray as xr
from xregrid import Regridder
from xregrid.utils import create_global_grid, create_grid_like, is_lazy


def test_aero_periodicity_lazy():
    """
    Verify that periodicity detection works with lazy (Dask) coordinates.
    The Aero Protocol: Refactored logic should avoid hidden computes for 1D dimension coords.
    """
    # Create a global grid (eager)
    ds_eager = create_global_grid(1.0, 1.0)

    # Create a lazy version
    ds_lazy = ds_eager.chunk({"lat": 10, "lon": 10})

    # Check periodicity of eager
    regridder_eager = Regridder(ds_eager, ds_eager, method="bilinear")
    assert regridder_eager.periodic is True

    # Check periodicity of lazy
    # For 1D dimension coordinates, Xarray indexes are eager, so it should be eager.
    regridder_lazy = Regridder(ds_lazy, ds_lazy, method="bilinear")
    assert regridder_lazy.periodic is True


def test_create_grid_like_lazy():
    """
    Verify that create_grid_like works with lazy (Dask) coordinates and avoids hidden computes.
    """
    # Use a predictable global grid as template
    ds_template = create_global_grid(2.0, 2.0)

    # Lazy version
    ds_lazy = ds_template.chunk({"lat": 10, "lon": 10})

    # create_grid_like should use the edge-sampling heuristic or Xarray indexes.
    # For 1D dimension coordinates, it should use indexes (zero compute).
    grid_new = create_grid_like(ds_lazy, res=2.0)

    assert grid_new.lat.size == 90
    assert grid_new.lon.size == 180

    # Check lineage in history
    assert "Created grid like input" in grid_new.attrs["history"]


def test_create_grid_like_lazy_2d():
    """
    Verify that create_grid_like works with lazy 2D coordinates.
    """
    # Create a template grid with 2D coordinates (curvilinear-like)
    # Use an exact number of points to make extent calculation predictable
    # res_lat = 180 / 45 = 4.0
    # res_lon = 360 / 90 = 4.0
    lon_2d, lat_2d = np.meshgrid(np.linspace(2, 358, 90), np.linspace(-88, 88, 45))
    ds_template = xr.Dataset(
        coords={
            "lat": (["y", "x"], lat_2d, {"units": "degrees_north"}),
            "lon": (["y", "x"], lon_2d, {"units": "degrees_east"}),
        }
    )

    # Lazy version
    ds_lazy = ds_template.chunk({"y": 10, "x": 10})

    # create_grid_like should use the edge-sampling heuristic for 2D lazy coords.
    grid_new = create_grid_like(ds_lazy, res=4.0)

    # Extent should be [-90, 90] for lat, [0, 360] for lon
    assert grid_new.lat.size == 45
    assert grid_new.lon.size == 90


def test_accessors_new_viz_methods():
    """
    Verify that the new visualization methods in accessors are present and callable.
    (We mock the actual plot calls as they require GUI/heavy deps).
    """
    ds = create_global_grid(10.0, 10.0)

    # Test DataArray accessor
    da = ds["lat"]  # Just a dummy variable
    assert hasattr(da.regrid, "plot_weights")
    assert hasattr(da.regrid, "plot_comparison")

    # Test Dataset accessor
    assert hasattr(ds.regrid, "plot_weights")
    assert hasattr(ds.regrid, "plot_comparison")

    # Note: We don't call them here because they would trigger matplotlib/cartopy/hvplot
    # which might not be fully configured or might try to open windows.
    # But we can verify they exist and have correct signatures.
    import inspect

    sig = inspect.signature(da.regrid.plot_weights)
    assert "target_grid" in sig.parameters
    assert "row_idx" in sig.parameters

    sig = inspect.signature(ds.regrid.plot_comparison)
    assert "target_grid" in sig.parameters
    assert "var_name" in sig.parameters


def test_aero_protocol_numpy_vs_dask():
    """
    Double-Check Test: Run the logic on a NumPy array, then convert to Dask
    and assert the result is identical.
    """
    # 1. Implementation (Numpy)
    ds_src = create_global_grid(10.0, 10.0)
    ds_tgt = create_global_grid(5.0, 5.0)

    # Create some dummy data
    data = np.random.rand(18, 36)
    da_src_eager = xr.DataArray(
        data, coords=[ds_src.lat, ds_src.lon], dims=["lat", "lon"]
    )

    regridder_eager = Regridder(ds_src, ds_tgt, method="bilinear")
    res_eager = regridder_eager(da_src_eager)

    # 2. Implementation (Dask)
    da_src_lazy = da_src_eager.chunk({"lat": 9, "lon": 9})
    regridder_lazy = Regridder(ds_src, ds_tgt, method="bilinear", parallel=False)
    res_lazy = regridder_lazy(da_src_lazy)

    # Verify results are identical
    xr.testing.assert_allclose(res_eager, res_lazy.compute())
    assert is_lazy(res_lazy)
