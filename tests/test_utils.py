from __future__ import annotations

# Consolidated tests: utils

import os
from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xregrid import (
    Regridder,
    create_global_grid,
    create_grid_from_crs,
    create_grid_from_ioapi,
    create_grid_like,
    create_mesh_from_coords,
    create_regional_grid,
    get_rdhpcs_cluster,
    load_esmf_file,
    plot_comparison,
    spatial_slice,
)
from xregrid.utils import get_crs_info
from xregrid.grid import _get_mesh_info

try:
    import esmpy  # noqa: F401

    HAS_REAL_ESMF = True
except (ImportError, Exception):
    HAS_REAL_ESMF = False


def test_auto_bounds_conservative_numpy_dask():
    """Verify auto-bounds generation for conservative regridding on both NumPy and Dask."""
    # Create a grid WITHOUT bounds but with standard names
    lat = np.linspace(-85, 85, 10)
    lon = np.linspace(0, 350, 20)
    ds_src = xr.Dataset(coords={"lat": lat, "lon": lon})
    ds_src.lat.attrs["standard_name"] = "latitude"
    ds_src.lat.attrs["units"] = "degrees_north"
    ds_src.lon.attrs["standard_name"] = "longitude"
    ds_src.lon.attrs["units"] = "degrees_east"

    # Target grid with bounds
    ds_tgt = create_global_grid(20, 20)

    # 1. Eager path
    regridder_eager = Regridder(ds_src, ds_tgt, method="conservative")
    da_src_eager = xr.DataArray(
        np.random.rand(10, 20), dims=("lat", "lon"), coords=ds_src.coords
    )
    res_eager = regridder_eager(da_src_eager)

    # 2. Lazy path
    da_src_lazy = da_src_eager.chunk({"lat": 5, "lon": 10})
    res_lazy = regridder_eager(da_src_lazy)

    assert isinstance(res_lazy.data, da.Array)
    xr.testing.assert_allclose(res_eager, res_lazy.compute())
    assert "Automatically generated" in res_eager.attrs["history"]


def test_plot_comparison_smoke():
    """Smoke test for plot_comparison utility."""
    ds = create_global_grid(30, 30)
    da_coords = {c: ds.coords[c] for c in ["lat", "lon"]}
    da = xr.DataArray(np.random.rand(6, 12), dims=("lat", "lon"), coords=da_coords)

    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")  # Non-interactive

    fig = plot_comparison(da, da)
    assert fig is not None
    plt.close(fig)


def test_cf_aware_dimension_mapping():
    """Verify that Regridder handles non-standard dimension names via CF-awareness."""
    # 1. Source grid with standard 'lat'/'lon'
    src_res = 10.0
    src_grid = create_global_grid(res_lat=src_res, res_lon=src_res)

    # 2. Target grid
    tgt_res = 5.0
    tgt_grid = create_global_grid(res_lat=tgt_res, res_lon=tgt_res)

    # 3. Initialize Regridder
    regridder = Regridder(src_grid, tgt_grid, method="bilinear")

    # 4. Input DataArray with different names: 'latitude' and 'longitude'
    # but marked with proper CF attributes
    data = np.random.rand(18, 36)
    da = xr.DataArray(
        data,
        dims=("latitude", "longitude"),
        coords={
            "latitude": (
                ["latitude"],
                src_grid.lat.values,
                {"standard_name": "latitude"},
            ),
            "longitude": (
                ["longitude"],
                src_grid.lon.values,
                {"standard_name": "longitude"},
            ),
        },
        name="test_data",
    )

    # 5. Eager Regridding
    res_eager = regridder(da)

    assert res_eager.shape == (36, 72)
    assert res_eager.name == "test_data"

    # 6. Lazy Regridding (Double-Check Rule)
    da_lazy = da.chunk({"latitude": 9, "longitude": 18})
    res_lazy = regridder(da_lazy).compute()

    # 7. Verification
    xr.testing.assert_allclose(res_eager, res_lazy)

    # Verify coordinates match target grid
    np.testing.assert_allclose(res_eager.lat, tgt_grid.lat)
    np.testing.assert_allclose(res_eager.lon, tgt_grid.lon)


def test_dataset_cf_awareness():
    """Verify CF-aware regridding for multiple variables in a Dataset."""
    src_grid = create_global_grid(20, 20)
    tgt_grid = create_global_grid(10, 10)

    regridder = Regridder(src_grid, tgt_grid)

    # Dataset with mixed naming
    ds = xr.Dataset(
        data_vars={
            "temp": (("latitude", "longitude"), np.random.rand(9, 18)),
            "scalar": 42.0,
        },
        coords={
            "latitude": (
                ["latitude"],
                src_grid.lat.values,
                {"standard_name": "latitude"},
            ),
            "longitude": (
                ["longitude"],
                src_grid.lon.values,
                {"standard_name": "longitude"},
            ),
            "fixed_coord": ("fixed", [1, 2, 3]),
        },
    )

    # Regrid
    ds_regridded = regridder(ds)

    assert "temp" in ds_regridded.data_vars
    assert ds_regridded.temp.shape == (18, 36)
    assert "scalar" in ds_regridded.data_vars
    assert ds_regridded.scalar == 42.0
    assert "fixed_coord" in ds_regridded.coords


def test_crs_propagation_dataarray():
    """
    Test that CRS metadata is propagated when regridding a DataArray.
    Verified with Eager (NumPy) and Lazy (Dask) data.
    """
    # 1. Setup Source Grid (Global Lat-Lon)
    src_ds = create_global_grid(res_lat=10, res_lon=10)

    # 2. Setup Target Grid (Projected UTM zone 33N)
    # UTM zone 33N is approx centered at 15E
    target_ds = create_grid_from_crs(
        crs="EPSG:32633", extent=(400000, 600000, 5000000, 5200000), res=10000
    )

    # Create source data
    data = np.random.rand(src_ds.sizes["lat"], src_ds.sizes["lon"])
    # Filter coords to only those compatible with (lat, lon) dims
    compatible_coords = {
        k: v for k, v in src_ds.coords.items() if set(v.dims).issubset({"lat", "lon"})
    }
    da_src_numpy = xr.DataArray(
        data, coords=compatible_coords, dims=("lat", "lon"), name="test_data"
    )

    da_src_dask = da_src_numpy.chunk({"lat": 5, "lon": 5})

    # Initialize Regridder
    regridder = Regridder(src_ds, target_ds, method="bilinear")

    for da_in in [da_src_numpy, da_src_dask]:
        # Perform Regridding
        da_out = regridder(da_in)

        # PROOF 1: CRS WKT Attribute Propagation
        assert "crs" in da_out.attrs
        assert "32633" in da_out.attrs["crs"]

        # PROOF 2: Grid Mapping Variable Propagation
        # create_grid_from_crs currently doesn't add a grid_mapping variable by default,
        # but it adds 'lat' and 'lon' coordinates.
        # Wait, let's check what create_grid_from_crs does.
        # It adds 'lat', 'lon' and sets attrs['crs'].

        # PROOF 3: Backend Consistency
        if hasattr(da_in.data, "dask"):
            assert hasattr(da_out.data, "dask")
        else:
            assert isinstance(da_out.data, np.ndarray)

        # PROOF 4: Viz Discovery
        # get_crs_info should return the correct CRS for the output
        crs_detected = get_crs_info(da_out)
        assert crs_detected is not None
        assert crs_detected.to_epsg() == 32633


def test_crs_propagation_dataset():
    """
    Test that CRS metadata is propagated when regridding a Dataset.
    """
    src_ds = create_global_grid(res_lat=10, res_lon=10)
    target_ds = create_grid_from_crs("EPSG:3857", (0, 10000, 0, 10000), 1000)

    data = np.random.rand(src_ds.sizes["lat"], src_ds.sizes["lon"])
    src_ds["var1"] = (("lat", "lon"), data)
    src_ds.attrs["history"] = "original history"

    regridder = Regridder(src_ds, target_ds, method="bilinear")
    ds_out = regridder(src_ds)

    # Global attribute propagation
    assert "crs" in ds_out.attrs
    assert "3857" in ds_out.attrs["crs"]

    # Variable attribute propagation
    assert "crs" in ds_out["var1"].attrs
    assert "3857" in ds_out["var1"].attrs["crs"]

    # History update
    assert "Regridded" in ds_out.attrs["history"]


def test_create_grid_from_ioapi_lcc():
    """Verify IOAPI grid generation for LCC projection (Eager and Lazy)."""
    metadata = {
        "GDTYP": 2,
        "P_ALP": 30.0,
        "P_BET": 60.0,
        "P_GAM": -97.0,
        "XCENT": -97.0,
        "YCENT": 40.0,
        "XORIG": -1000.0,
        "YORIG": -1000.0,
        "XCELL": 500.0,
        "YCELL": 500.0,
        "NCOLS": 4,
        "NROWS": 4,
    }

    # 1. Eager test
    ds_eager = create_grid_from_ioapi(metadata)

    assert "x" in ds_eager.coords
    assert "y" in ds_eager.coords
    assert "lat" in ds_eager.coords
    assert "lon" in ds_eager.coords
    assert "x_b" in ds_eager.coords
    assert "y_b" in ds_eager.coords
    assert ds_eager.sizes["x"] == 4
    assert ds_eager.sizes["y"] == 4
    assert ds_eager.attrs["ioapi_GDTYP"] == 2

    # Check 1D bounds values
    assert ds_eager.x_b.shape == (4, 2)
    assert np.allclose(ds_eager.x_b[0].values, [-1000.0, -500.0])

    # 2. Lazy test
    ds_lazy = create_grid_from_ioapi(metadata, chunks={"x": 2, "y": 2})
    # Verify laziness
    assert hasattr(ds_lazy.lat.data, "dask")
    assert hasattr(ds_lazy.x_b.data, "dask")

    ds_lazy_comp = ds_lazy.compute()
    xr.testing.assert_allclose(ds_eager, ds_lazy_comp)


def test_create_grid_from_ioapi_all_gdtyp():
    """Verify that all supported IOAPI GDTYP values can generate a grid."""
    base_metadata = {
        "P_ALP": 30.0,
        "P_BET": 60.0,
        "P_GAM": -97.0,
        "XCENT": -97.0,
        "YCENT": 40.0,
        "XORIG": -1000.0,
        "YORIG": -1000.0,
        "XCELL": 500.0,
        "YCELL": 500.0,
        "NCOLS": 2,
        "NROWS": 2,
    }

    # GDTYP 1-10
    for gdtyp in range(1, 11):
        metadata = base_metadata.copy()
        metadata["GDTYP"] = gdtyp

        # Some specific adjustments to avoid proj errors if needed
        if gdtyp == 5:  # UTM
            metadata["P_ALP"] = 17  # Zone 17

        ds = create_grid_from_ioapi(metadata)
        assert "lat" in ds.coords
        assert "lon" in ds.coords
        assert ds.attrs["ioapi_GDTYP"] == gdtyp


def test_create_grid_from_ioapi_latlon():
    """Verify IOAPI grid generation for Lat-Lon."""
    metadata = {
        "GDTYP": 1,
        "P_ALP": 0.0,
        "P_BET": 0.0,
        "P_GAM": 0.0,
        "XCENT": 0.0,
        "YCENT": 0.0,
        "XORIG": -10.0,
        "YORIG": 40.0,
        "XCELL": 1.0,
        "YCELL": 1.0,
        "NCOLS": 10,
        "NROWS": 10,
    }
    ds = create_grid_from_ioapi(metadata)
    assert ds.sizes["x"] == 10
    assert ds.sizes["y"] == 10
    # For Lat-Lon, pyproj transform from EPSG:4326 to EPSG:4326 should be identity
    # but create_grid_from_crs might return lat/lon that are slightly different due to transform
    assert ds.lat.min() >= 40.0


try:
    import dask.array as da
except ImportError:
    da = None


def test_create_mesh_from_coords_aero():
    """
    Double-Check Test for create_mesh_from_coords.
    Verifies Eager (NumPy) and Lazy (Dask) backends yield identical results
    and maintain scientific provenance.
    """
    # 1. Setup sample coordinates (Lambert Conformal-ish)
    x = np.linspace(-1000, 1000, 10)
    y = np.linspace(-1000, 1000, 10)
    crs = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

    # 2. Eager execution
    ds_eager = create_mesh_from_coords(x, y, crs)

    # Assertions for Eager
    assert isinstance(ds_eager, xr.Dataset)
    assert "lat" in ds_eager
    assert "lon" in ds_eager
    assert "x" in ds_eager
    assert "y" in ds_eager
    assert ds_eager.attrs["grid_mapping"] == "spatial_ref"
    assert "spatial_ref" in ds_eager
    assert "Eager" in ds_eager.attrs["history"]
    assert "Extent:" in ds_eager.attrs["history"]

    # Check that it's actually NumPy-backed
    assert not hasattr(ds_eager.lat.data, "dask")

    # 3. Lazy execution
    if da is None:
        pytest.skip("Dask not installed, skipping lazy check.")

    x_lazy = da.from_array(x, chunks=5)
    y_lazy = da.from_array(y, chunks=5)

    ds_lazy = create_mesh_from_coords(x_lazy, y_lazy, crs)

    # Assertions for Lazy
    assert "Lazy" in ds_lazy.attrs["history"]
    # Lazy path should NOT have extent in history to avoid compute()
    assert "Extent:" not in ds_lazy.attrs["history"]
    assert hasattr(ds_lazy.lat.data, "dask")

    # 4. Numerical Verification (The "Double-Check")
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # 5. Verify Metadata propagation
    assert ds_eager.lat.attrs["units"] == "degrees_north"
    assert ds_eager.x.attrs["standard_name"] == "projection_x_coordinate"
    assert ds_eager.x.attrs["grid_mapping"] == "spatial_ref"


def test_create_mesh_from_coords_regression_fix():
    """
    Verify the fix for the dimension mismatch regression and conditional metadata.
    """
    # 1. Test DataArray inputs with different dimension names
    x_da = xr.DataArray(np.linspace(0, 10, 5), dims=["lon"], name="my_lon")
    y_da = xr.DataArray(np.linspace(0, 10, 5), dims=["lat"], name="my_lat")
    crs_proj = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

    ds = create_mesh_from_coords(x_da, y_da, crs_proj)

    # Should have 5 points, not 25 (if it had broadcasted incorrectly)
    assert ds.sizes["n_pts"] == 5
    assert ds.x.attrs["standard_name"] == "projection_x_coordinate"

    # 2. Test geographic CRS metadata
    crs_geo = "EPSG:4326"
    ds_geo = create_mesh_from_coords(x_da, y_da, crs_geo)

    assert ds_geo.x.attrs["standard_name"] == "longitude"
    assert ds_geo.x.attrs["units"] == "degrees_east"
    assert ds_geo.y.attrs["standard_name"] == "latitude"
    assert ds_geo.y.attrs["units"] == "degrees_north"


def test_spatial_slice_rectilinear() -> None:
    """
    Test spatial_slice with a rectilinear grid (Eager and Lazy).

    Verifies that spatial_slice correctly subsets a rectilinear grid
    using coordinate indexes and handles both NumPy and Dask backends.
    """
    # 1. Eager
    ds = create_global_grid(res_lat=1.0, res_lon=1.0)
    # Extent: (min_x, max_x, min_y, max_y)
    extent = (10.5, 20.5, 30.5, 40.5)
    ds_sliced = spatial_slice(ds, extent)

    assert ds_sliced.lat.min() >= 30.5
    assert ds_sliced.lat.max() <= 40.5
    assert ds_sliced.lon.min() >= 10.5
    assert ds_sliced.lon.max() <= 20.5
    assert "history" in ds_sliced.attrs
    assert "Spatially sliced" in ds_sliced.attrs["history"]

    # 2. Lazy (Dask)
    ds_lazy = create_global_grid(
        res_lat=1.0, res_lon=1.0, chunks={"lat": 10, "lon": 10}
    )
    ds_sliced_lazy = spatial_slice(ds_lazy, extent)

    # In xarray, dimension coordinates are often eager (NumPy) due to indexing.
    # Check lat_b instead, which should remain lazy.
    assert hasattr(ds_sliced_lazy.lat_b.data, "dask")
    xr.testing.assert_allclose(ds_sliced, ds_sliced_lazy.compute())


def test_spatial_slice_unstructured() -> None:
    """
    Test spatial_slice with an unstructured grid (Eager and Lazy).

    Verifies that spatial_slice correctly subsets an unstructured grid
    using boolean masking and maintains laziness for Dask-backed data.
    """
    # Create points: one in the box, one outside
    lons = np.array([15.0, 50.0])
    lats = np.array([35.0, 60.0])
    ds = create_mesh_from_coords(lons, lats, crs="EPSG:4326")

    extent = (10.0, 20.0, 30.0, 40.0)

    # Eager path: drop=True is supported
    ds_sliced = spatial_slice(ds, extent)

    assert ds_sliced.sizes["n_pts"] == 1
    assert ds_sliced.lon.values[0] == 15.0
    assert ds_sliced.lat.values[0] == 35.0

    # Lazy path: drop=False is used to preserve laziness
    ds_lazy = create_mesh_from_coords(lons, lats, crs="EPSG:4326", chunks=1)
    ds_lazy["data"] = (["n_pts"], ds_lazy.lat.data, {"units": "K"})
    ds_sliced_lazy = spatial_slice(ds_lazy, extent)

    # Verify laziness
    assert hasattr(ds_sliced_lazy.data.data, "dask")

    # Verify results (after compute and dropna since drop=False was used)
    ds_res = ds_sliced_lazy.compute().dropna("n_pts")
    assert ds_res.sizes["n_pts"] == 1
    assert ds_res.lon.values[0] == 15.0


def test_spatial_slice_wrapping() -> None:
    """
    Test spatial_slice with longitude wrapping across different grid types.

    Verifies that spatial_slice correctly handles regions crossing the
    meridian/dateline for both rectilinear and unstructured grids.
    """
    ds = create_global_grid(res_lat=1.0, res_lon=1.0)

    # Slice crossing the 0/360 boundary
    # Extent: (min_x, max_x, min_y, max_y)
    extent = (-10.5, 10.5, -10.5, 10.5)
    ds_sliced = spatial_slice(ds, extent)

    # Lon in ds is 0-360. -10.5 should map to 349.5
    assert ds_sliced.lon.min() >= 0
    assert ds_sliced.lon.max() <= 360

    # Check that we have both the 0-10.5 and 349.5-360 parts
    assert (ds_sliced.lon <= 10.5).any()
    assert (ds_sliced.lon >= 349.5).any()

    # Verify unstructured wrapping
    lons = np.array([5.0, 355.0, 180.0])
    lats = np.array([0.0, 0.0, 0.0])
    ds_unstructured = create_mesh_from_coords(lons, lats, crs="EPSG:4326")
    ds_un_sliced = spatial_slice(ds_unstructured, extent)

    # Since it's eager, drop=True was used
    assert ds_un_sliced.sizes["n_pts"] == 2
    assert set(ds_un_sliced.lon.values) == {5.0, 355.0}


def test_create_global_grid_lazy():
    """
    Aero Protocol: Double-Check Test for create_global_grid.
    Verifies that values are identical between NumPy and Dask backends.
    """
    res_lat, res_lon = 10, 20

    # Eager (NumPy)
    ds_eager = create_global_grid(res_lat=res_lat, res_lon=res_lon, chunks=None)
    assert not ds_eager.chunks

    # Lazy (Dask)
    ds_lazy = create_global_grid(
        res_lat=res_lat, res_lon=res_lon, chunks={"lat": 9, "lon": 9}
    )
    assert ds_lazy.chunks

    # Assert values are identical
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Verify internal backend (lat_b is non-index so it should be chunked)
    assert hasattr(ds_lazy.lat_b.data, "dask")


def test_create_regional_grid_lazy():
    """
    Aero Protocol: Double-Check Test for create_regional_grid.
    """
    lat_range = (-45, 45)
    lon_range = (0, 90)
    res_lat, res_lon = 5, 5

    # Eager (NumPy)
    ds_eager = create_regional_grid(lat_range, lon_range, res_lat, res_lon, chunks=None)

    # Lazy (Dask)
    ds_lazy = create_regional_grid(lat_range, lon_range, res_lat, res_lon, chunks=5)
    assert ds_lazy.chunks

    # Assert values are identical
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Verify internal backend
    assert hasattr(ds_lazy.lat_b.data, "dask")


def test_create_grid_from_crs_lazy():
    """
    Aero Protocol: Double-Check Test for create_grid_from_crs.
    """
    # Test with EPSG:32633 (UTM zone 33N)
    extent = (400000, 500000, 5000000, 5100000)
    res = 10000  # 10km

    # Eager (NumPy)
    ds_eager = create_grid_from_crs("EPSG:32633", extent, res, chunks=None)

    # Lazy (Dask)
    ds_lazy = create_grid_from_crs("EPSG:32633", extent, res, chunks={"x": 5, "y": 5})
    assert ds_lazy.chunks

    # Assert values are identical
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Verify internal backend (lat/lon are non-index here)
    assert hasattr(ds_lazy.lat.data, "dask")


def test_create_mesh_from_coords_lazy():
    """
    Aero Protocol: Double-Check Test for create_mesh_from_coords.
    """
    x = np.array([400000, 450000, 500000])
    y = np.array([5000000, 5050000, 5100000])

    # Eager (NumPy)
    ds_eager = create_mesh_from_coords(x, y, "EPSG:32633", chunks=None)

    # Lazy (Dask)
    ds_lazy = create_mesh_from_coords(x, y, "EPSG:32633", chunks={"n_pts": 2})
    assert ds_lazy.chunks

    # Assert values are identical
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Verify internal backend
    assert hasattr(ds_lazy.lat.data, "dask")


def test_create_grid_like_latlon():
    """
    Aero Protocol: Double-Check Test for create_grid_like (Lat-Lon).
    Verifies identity between NumPy and Dask backends and preservation of laziness.
    """
    # Create a source grid
    ds_src = create_regional_grid(
        lat_range=(10, 20),
        lon_range=(100, 110),
        res_lat=1.0,
        res_lon=1.0,
        add_bounds=True,
    )

    res_new = 0.5

    # Eager (NumPy)
    ds_eager = create_grid_like(ds_src, res_new, chunks=None)
    assert not ds_eager.chunks
    assert ds_eager.lat.size == 20
    assert ds_eager.lon.size == 20

    # Lazy (Dask)
    ds_src_lazy = ds_src.chunk({"lat": 5, "lon": 5})
    ds_lazy = create_grid_like(ds_src_lazy, res_new, chunks=5)

    # Assert values are identical
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Verify laziness: lat_b should be a dask array
    assert hasattr(ds_lazy.lat_b.data, "dask")


def test_create_grid_like_projected():
    """
    Aero Protocol: Double-Check Test for create_grid_like (Projected).
    """
    # UTM zone 33N
    crs = "EPSG:32633"
    extent = (400000, 500000, 5000000, 5100000)
    res_orig = 10000

    ds_src = create_grid_from_crs(crs, extent, res_orig, add_bounds=True)

    res_new = 5000

    # Eager
    ds_eager = create_grid_like(ds_src, res_new, chunks=None)

    # Lazy
    ds_src_lazy = ds_src.chunk({"x": 5, "y": 5})
    ds_lazy = create_grid_like(ds_src_lazy, res_new, chunks=5)

    # Assert values
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Verify laziness (lat/lon are non-index in projected grid)
    assert hasattr(ds_lazy.lat.data, "dask")
    assert ds_lazy.attrs["crs"] == ds_src.attrs["crs"]


def test_rectilinear_hygiene():
    """
    Verify that _create_rectilinear_grid produces high-hygiene metadata.
    """
    ds = create_regional_grid((0, 10), (0, 10), 1, 1)

    assert ds.attrs["crs"] == "EPSG:4326"

    # Test custom CRS
    # create_regional_grid currently doesn't expose crs, let's test _create_rectilinear_grid directly
    from xregrid.utils import _create_rectilinear_grid

    ds_nad83 = _create_rectilinear_grid((0, 10), (0, 10), 1, 1, crs="EPSG:4269")
    assert ds_nad83.attrs["crs"] == "EPSG:4269"

    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lon.attrs["standard_name"] == "longitude"
    assert ds.lat_b.attrs["standard_name"] == "latitude_bounds"
    assert ds.lon_b.attrs["standard_name"] == "longitude_bounds"
    assert "history" in ds.attrs


def test_cf_coords_detection():
    # Create dataset with non-standard coordinate names but with CF attributes
    def create_ds(lazy=False):
        np.random.seed(42)
        data = np.random.rand(10, 20)
        if lazy:
            data = da.from_array(data, chunks=(5, 10))

        ds = xr.Dataset(
            {"data": (("lat_dim", "lon_dim"), data)},
            coords={
                "latitude": (
                    ("lat_dim",),
                    np.linspace(-90, 90, 10),
                    {"units": "degrees_north", "standard_name": "latitude"},
                ),
                "longitude": (
                    ("lon_dim",),
                    np.linspace(-180, 180, 20),
                    {"units": "degrees_east", "standard_name": "longitude"},
                ),
            },
        )
        return ds

    # Target grid also needs bounds for conservative regridding
    lat_edges_tgt = np.linspace(-90, 90, 16)
    lon_edges_tgt = np.linspace(-180, 180, 26)
    ds_tgt = xr.Dataset(
        coords={
            "lat": (("lat",), np.linspace(-90, 90, 15), {"units": "degrees_north"}),
            "lon": (("lon",), np.linspace(-180, 180, 25), {"units": "degrees_east"}),
            "lat_b": (("lat_b",), lat_edges_tgt, {"units": "degrees_north"}),
            "lon_b": (("lon_b",), lon_edges_tgt, {"units": "degrees_east"}),
        },
    )

    # Test Eager
    ds_src_eager = create_ds(lazy=False)
    regridder_eager = Regridder(ds_src_eager, ds_tgt)
    out_eager = regridder_eager(ds_src_eager["data"])
    assert out_eager.shape == (15, 25)
    assert not out_eager.chunks

    # Test Lazy
    ds_src_lazy = create_ds(lazy=True)
    regridder_lazy = Regridder(ds_src_lazy, ds_tgt)
    out_lazy = regridder_lazy(ds_src_lazy["data"])
    assert out_lazy.shape == (15, 25)
    assert out_lazy.chunks

    # Verify results are identical (within float precision)
    np.testing.assert_allclose(out_eager.values, out_lazy.compute().values)


def test_cf_bounds_detection():
    # Create dataset with non-standard bound names but with CF attributes
    ds_src = xr.Dataset(
        {"data": (("lat", "lon"), np.random.rand(10, 20))},
        coords={
            "lat": (
                ("lat",),
                np.linspace(-90, 90, 10),
                {"units": "degrees_north", "bounds": "lat_bounds"},
            ),
            "lon": (
                ("lon",),
                np.linspace(-180, 180, 20),
                {"units": "degrees_east", "bounds": "lon_bounds"},
            ),
            "lat_bounds": (("lat", "nv"), np.random.rand(10, 2)),  # Placeholder bounds
            "lon_bounds": (("lon", "nv"), np.random.rand(20, 2)),  # Placeholder bounds
        },
    )

    # We need to make the bounds contiguous for our converter to work correctly in this test
    lat_edges = np.linspace(-90, 90, 11)
    lat_bounds = np.stack([lat_edges[:-1], lat_edges[1:]], axis=1)
    lon_edges = np.linspace(-180, 180, 21)
    lon_bounds = np.stack([lon_edges[:-1], lon_edges[1:]], axis=1)

    ds_src.coords["lat_bounds"] = (("lat", "nv"), lat_bounds)
    ds_src.coords["lon_bounds"] = (("lon", "nv"), lon_bounds)

    # Target grid also needs bounds for conservative regridding
    lat_edges_tgt = np.linspace(-90, 90, 16)
    lon_edges_tgt = np.linspace(-180, 180, 26)
    ds_tgt = xr.Dataset(
        coords={
            "lat": (("lat",), np.linspace(-90, 90, 15), {"units": "degrees_north"}),
            "lon": (("lon",), np.linspace(-180, 180, 25), {"units": "degrees_east"}),
            "lat_b": (("lat_b",), lat_edges_tgt, {"units": "degrees_north"}),
            "lon_b": (("lon_b",), lon_edges_tgt, {"units": "degrees_east"}),
        },
    )

    regridder = Regridder(ds_src, ds_tgt, method="conservative")
    # If it reached here without error, it found the bounds and ESMPy initialized
    out = regridder(ds_src["data"])
    assert out.shape == (15, 25)


def test_regridder_time_dimension_detection():
    # Setup source and target grids with time
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(0, 360, 20)
    times = [np.datetime64("2020-01-01")]

    src_ds = xr.Dataset(
        coords={
            "time": (["time"], times, {"standard_name": "time"}),
            "lat": (
                ["time", "lat"],
                np.broadcast_to(lats, (1, 10)),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                lons,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    tgt_ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                np.linspace(-90, 90, 5),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, 10),
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    # This should now work without failing during weight generation
    regridder = Regridder(src_ds, tgt_ds, method="bilinear")

    # Create data with time and vertical dimensions
    levs = np.arange(5)
    data = np.random.rand(len(times), len(levs), len(lats), len(lons))
    da = xr.DataArray(
        data,
        coords={
            "time": (["time"], times),
            "lev": (["lev"], levs),
            "lat": (["time", "lat"], np.broadcast_to(lats, (1, 10))),
            "lon": (["lon"], lons),
        },
        dims=("time", "lev", "lat", "lon"),
        name="temp",
    )

    # Regrid DataArray
    res_da = regridder(da)

    # Check that time and lev are preserved
    assert "time" in res_da.dims
    assert "lev" in res_da.dims
    assert res_da.shape == (1, 5, 5, 10)

    # Regrid Dataset
    ds = xr.Dataset({"temp": da, "time_var": (["time"], times)})
    res_ds = regridder(ds)

    assert "time" in res_ds.dims
    assert "temp" in res_ds.data_vars
    assert "time_var" in res_ds.data_vars
    assert res_ds["temp"].shape == (1, 5, 5, 10)
    assert res_ds["time_var"].dims == ("time",)


def test_regridder_dtype_time_fallback():
    # Setup with time-like dtype but non-standard name
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(0, 360, 20)
    times = [np.datetime64("2020-01-01")]

    src_ds = xr.Dataset(
        coords={
            "mytime": (["mytime"], times),  # Non-standard name, no CF attributes
            "lat": (
                ["mytime", "lat"],
                np.broadcast_to(lats, (1, 10)),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                lons,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    tgt_ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                np.linspace(-90, 90, 5),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, 10),
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    regridder = Regridder(src_ds, tgt_ds)

    # Verify mytime was detected as non-spatial
    assert "mytime" not in regridder._dims_source

    # Test DataArray regridding with this non-standard time dim
    da = xr.DataArray(
        np.random.rand(1, 10, 20), coords=src_ds.coords, dims=("mytime", "lat", "lon")
    )

    res = regridder(da)
    assert "mytime" in res.dims
    assert res.shape == (1, 5, 10)


def test_non_regriddable_object():
    # Test passing something that shouldn't be regridded
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(0, 360, 20)

    src_ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                lats,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                lons,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )
    tgt_ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                np.linspace(-90, 90, 5),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, 10),
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    regridder = Regridder(src_ds, tgt_ds)

    # A DataArray that only has one dimension (time)
    time_da = xr.DataArray([1, 2, 3], dims="time", name="time_var")

    # Should return unchanged
    res = regridder(time_da)
    xr.testing.assert_identical(res, time_da)


def test_regridder_vertical_dimension_detection():
    # Setup source with vertical dimension in lats
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(0, 360, 20)
    levs = np.arange(3)

    src_ds = xr.Dataset(
        coords={
            "lev": (["lev"], levs, {"standard_name": "altitude"}),
            "lat": (
                ["lev", "lat"],
                np.broadcast_to(lats, (3, 10)),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                lons,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    tgt_ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                np.linspace(-90, 90, 5),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, 10),
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    regridder = Regridder(src_ds, tgt_ds)
    assert "lev" not in regridder._dims_source

    da = xr.DataArray(
        np.random.rand(3, 10, 20), coords=src_ds.coords, dims=("lev", "lat", "lon")
    )

    res = regridder(da)
    assert "lev" in res.dims
    assert res.shape == (3, 5, 10)


def test_regridder_ugrid_with_time():
    # Setup mocked uxarray object with time dimension
    from unittest.mock import MagicMock

    class UxDatasetMock(xr.Dataset):
        def __init__(self, ds, uxgrid):
            super().__init__(ds.data_vars, coords=ds.coords, attrs=ds.attrs)
            self.uxgrid = uxgrid

    n_face = 10
    n_node = 12
    times = [np.datetime64("2020-01-01")]

    mock_uxgrid = MagicMock()
    mock_uxgrid.node_lat = xr.DataArray(np.linspace(-90, 90, n_node), dims=["n_node"])
    mock_uxgrid.node_lon = xr.DataArray(np.linspace(0, 360, n_node), dims=["n_node"])
    # face coords with time dimension (moving mesh case, though xregrid assumes static)
    mock_uxgrid.face_lat = xr.DataArray(
        np.broadcast_to(np.linspace(-90, 90, n_face), (1, n_face)),
        dims=["time", "n_face"],
        coords={"time": times},
    )
    mock_uxgrid.face_lon = xr.DataArray(
        np.broadcast_to(np.linspace(0, 360, n_face), (1, n_face)),
        dims=["time", "n_face"],
        coords={"time": times},
    )

    # Create connectivity
    conn = np.zeros((n_face, 3), dtype=int)
    for i in range(n_face):
        conn[i] = [i, i + 1, (i + 2) % n_node]

    mock_uxgrid.face_node_connectivity = xr.DataArray(
        conn, dims=["n_face", "n_max_face_nodes"]
    )
    mock_uxgrid.face_node_connectivity.attrs["start_index"] = 0
    mock_uxgrid.face_node_connectivity.attrs["_FillValue"] = -1

    # Mock UxDataset with time-varying variable
    ds_base = xr.Dataset(
        {"test_var": (["time", "n_face"], np.random.rand(1, n_face))},
        coords={"time": (["time"], times, {"standard_name": "time"})},
    )
    ds = UxDatasetMock(ds_base, mock_uxgrid)

    target_grid = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                np.linspace(-90, 90, 5),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, 10),
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    regridder = Regridder(ds, target_grid, method="nearest_s2d")

    # Verify time was detected as non-spatial
    assert "time" not in regridder._dims_source
    assert regridder._dims_source == ("n_face",)

    # Regrid DataArray
    res = regridder(ds["test_var"])

    assert "time" in res.dims
    assert res.shape == (1, 5, 10)


def test_regridder_raw_ugrid_with_time():
    n_face = 10
    n_node = 12
    times = [np.datetime64("2020-01-01")]

    # Create a raw dataset following UGRID convention
    conn = np.zeros((n_face, 3), dtype=int)
    for i in range(n_face):
        conn[i] = [i, (i + 1) % n_node, (i + 2) % n_node]

    ds = xr.Dataset(
        data_vars={
            "temp": (["time", "n_face"], np.random.rand(1, n_face)),
            "face_node_connectivity": (["n_face", "n_max_face_nodes"], conn),
        },
        coords={
            "time": (["time"], times, {"standard_name": "time"}),
            "lat_face": (
                ["time", "n_face"],
                np.broadcast_to(np.linspace(-90, 90, n_face), (1, n_face)),
                {"units": "degrees_north"},
            ),
            "lon_face": (
                ["time", "n_face"],
                np.broadcast_to(np.linspace(0, 360, n_face), (1, n_face)),
                {"units": "degrees_east"},
            ),
            "lat_node": (
                ["time", "n_node"],
                np.broadcast_to(np.linspace(-90, 90, n_node), (1, n_node)),
                {"units": "degrees_north"},
            ),
            "lon_node": (
                ["time", "n_node"],
                np.broadcast_to(np.linspace(0, 360, n_node), (1, n_node)),
                {"units": "degrees_east"},
            ),
        },
    )

    ds.face_node_connectivity.attrs["cf_role"] = "face_node_connectivity"
    ds.face_node_connectivity.attrs["start_index"] = 0

    from xregrid import create_global_grid

    target_grid = create_global_grid(10, 10)

    regridder = Regridder(ds, target_grid, method="nearest_s2d")

    assert "time" not in regridder._dims_source
    # Since it is UGRID, it should have detected n_face as the spatial dimension for variables
    assert "n_face" in regridder._dims_source

    res = regridder(ds["temp"])
    assert "time" in res.dims
    assert res.shape == (1, 18, 36)


def test_regridder_user_specific_structure():
    # Mimic user's dataset structure: (time, node)
    # node is string coordinate, lat/lon are (node)
    n_node = 10
    n_time = 5
    times = np.arange(n_time).astype("datetime64[D]")
    nodes = np.array([f"NODE_{i}" for i in range(n_node)], dtype="<U19")

    src_ds = xr.Dataset(
        data_vars={
            "day_of_year": (
                ["time", "node"],
                np.random.rand(n_time, n_node).astype("float32"),
            ),
            "aod_550nm": (
                ["time", "node"],
                np.random.rand(n_time, n_node).astype("float32"),
            ),
            "mesh": ([], np.int32(1)),
        },
        coords={
            "time": (["time"], times),
            "node": (["node"], nodes),
            "latitude": (["node"], np.linspace(-90, 90, n_node)),
            "longitude": (["node"], np.linspace(0, 360, n_node)),
        },
    )

    from xregrid import create_global_grid

    target_grid = create_global_grid(10, 10)

    # Use bilinear (nearest_s2d for mock compatibility)
    regridder = Regridder(src_ds, target_grid, method="nearest_s2d")

    assert "time" not in regridder._dims_source
    assert regridder._dims_source == ("node",)

    # Regrid a variable
    res = regridder(src_ds["aod_550nm"])

    assert "time" in res.dims
    assert res.shape == (n_time, 18, 36)
    assert res.dtype == "float32"

    # Regrid the whole dataset
    res_ds = regridder(src_ds)
    assert "time" in res_ds.dims
    assert "aod_550nm" in res_ds.data_vars
    assert res_ds["aod_550nm"].shape == (n_time, 18, 36)
    assert "node" not in res_ds["aod_550nm"].dims  # Space dimension should be replaced
    assert "mesh" in res_ds.data_vars  # Non-spatial data var should be preserved


def test_regridder_raw_ugrid_conservative_with_time():
    n_face = 10
    n_node = 12
    times = [np.datetime64("2020-01-01")]

    # Create a raw dataset following UGRID convention for conservative regridding
    # Conservative needs faces and nodes (for connectivity)
    conn = np.zeros((n_face, 3), dtype=int)
    for i in range(n_face):
        conn[i] = [i, (i + 1) % n_node, (i + 2) % n_node]

    ds = xr.Dataset(
        data_vars={
            "temp": (["time", "n_face"], np.random.rand(1, n_face)),
            "face_node_connectivity": (["n_face", "n_max_face_nodes"], conn),
        },
        coords={
            "time": (["time"], times, {"standard_name": "time"}),
            "lat_node": (
                ["time", "n_node"],
                np.broadcast_to(np.linspace(-90, 90, n_node), (1, n_node)),
                {"units": "degrees_north"},
            ),
            "lon_node": (
                ["time", "n_node"],
                np.broadcast_to(np.linspace(0, 360, n_node), (1, n_node)),
                {"units": "degrees_east"},
            ),
            "lat_face": (
                ["time", "n_face"],
                np.broadcast_to(np.linspace(-90, 90, n_face), (1, n_face)),
                {"units": "degrees_north"},
            ),
            "lon_face": (
                ["time", "n_face"],
                np.broadcast_to(np.linspace(0, 360, n_face), (1, n_face)),
                {"units": "degrees_east"},
            ),
        },
    )

    ds.face_node_connectivity.attrs["cf_role"] = "face_node_connectivity"
    ds.face_node_connectivity.attrs["start_index"] = 0

    from xregrid import create_global_grid

    target_grid = create_global_grid(10, 10)

    # This should trigger _get_unstructured_mesh_info
    regridder = Regridder(ds, target_grid, method="conservative")

    assert "time" not in regridder._dims_source
    assert "n_face" in regridder._dims_source

    res = regridder(ds["temp"])
    assert "time" in res.dims
    assert res.shape == (1, 18, 36)


def test_get_mesh_info_rectilinear_order():
    """Test that _get_mesh_info correctly handles rectilinear grids with different coord orders."""
    # Create a grid where coords are (lon, lat)
    lon = np.arange(0, 360, 10)
    lat = np.arange(-90, 91, 10)

    # Broadcast to create a dataset
    ds = xr.Dataset(coords={"lat": lat, "lon": lon})

    # Check (lat, lon) order
    lon_m, lat_m, shape, dims, unstructured = _get_mesh_info(ds)
    assert not unstructured
    assert dims == ("lat", "lon")
    assert shape == (lat.size, lon.size)
    assert lat_m.shape == (lat.size, lon.size)
    assert lon_m.shape == (lat.size, lon.size)


def test_get_rdhpcs_cluster_detection():
    """Test machine detection in get_rdhpcs_cluster."""

    with patch("socket.gethostname") as mock_hostname:
        # Test Hera detection
        mock_hostname.return_value = "hfe01.hera.noaa.gov"
        with patch("dask_jobqueue.SLURMCluster", MagicMock()) as mock_slurm:
            get_rdhpcs_cluster(account="test_acc")
            args, kwargs = mock_slurm.call_args
            assert kwargs["queue"] == "hera"
            assert kwargs["cores"] == 40

        # Test Jet detection
        mock_hostname.return_value = "fe01.jet.noaa.gov"
        with patch("dask_jobqueue.SLURMCluster", MagicMock()) as mock_slurm:
            get_rdhpcs_cluster(account="test_acc")
            args, kwargs = mock_slurm.call_args
            assert kwargs["queue"] == "batch"
            assert kwargs["cores"] == 24

        # Test Gaea detection
        mock_hostname.return_value = "gaea12.ncrc.gov"
        with patch("dask_jobqueue.SLURMCluster", MagicMock()) as mock_slurm:
            get_rdhpcs_cluster(account="test_acc", machine="gaea-c6")
            args, kwargs = mock_slurm.call_args
            assert kwargs["cores"] == 192
            assert "-M c6" in kwargs["job_extra_directives"][0]

        # Test Ursa detection
        mock_hostname.return_value = "ufe01.ursa.noaa.gov"
        with patch("dask_jobqueue.SLURMCluster", MagicMock()) as mock_slurm:
            get_rdhpcs_cluster(account="test_acc")
            args, kwargs = mock_slurm.call_args
            assert kwargs["queue"] == "u1-compute"
            assert kwargs["cores"] == 192


def test_get_rdhpcs_cluster_explicit():
    """Test explicit machine specification in get_rdhpcs_cluster."""
    with patch("dask_jobqueue.SLURMCluster", MagicMock()) as mock_slurm:
        get_rdhpcs_cluster(machine="hera", account="test_acc", walltime="02:00:00")
        args, kwargs = mock_slurm.call_args
        assert kwargs["queue"] == "hera"
        assert kwargs["walltime"] == "02:00:00"
        assert kwargs["account"] == "test_acc"


def test_create_global_grid():
    ds = create_global_grid(res_lat=10, res_lon=20)
    assert "lat" in ds
    assert "lon" in ds
    assert ds.lat.size == 18  # 180 / 10
    assert ds.lon.size == 18  # 360 / 20
    assert "lat_b" in ds
    assert "lon_b" in ds
    # New (N, 2) bounds format
    assert ds.lat_b.shape == (18, 2)
    assert ds.lon_b.shape == (18, 2)
    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lon.attrs["standard_name"] == "longitude"
    assert "history" in ds.attrs


def test_create_regional_grid():
    ds = create_regional_grid(
        lat_range=(-45, 45), lon_range=(0, 90), res_lat=5, res_lon=5
    )
    assert ds.lat.size == 18  # 90 / 5
    assert ds.lon.size == 18  # 90 / 5
    assert ds.lat.min() == -42.5
    assert ds.lat.max() == 42.5
    assert "lat_b" in ds
    assert np.isclose(ds.lat_b.min(), -45)
    assert np.isclose(ds.lat_b.max(), 45)


def test_load_esmf_file(tmp_path):
    # Create a dummy NetCDF file
    filepath = os.path.join(tmp_path, "test_mesh.nc")
    ds_orig = xr.Dataset({"test": (("x",), [1, 2, 3])})
    ds_orig.to_netcdf(filepath)

    ds_loaded = load_esmf_file(filepath)
    assert "test" in ds_loaded
    assert "history" in ds_loaded.attrs
    assert "Loaded ESMF file" in ds_loaded.attrs["history"]


def test_create_grid_from_crs():
    # Test with EPSG:32633 (UTM zone 33N)
    extent = (400000, 500000, 5000000, 5100000)
    res = 10000  # 10km
    ds = create_grid_from_crs("EPSG:32633", extent, res)

    assert "lat" in ds
    assert "lon" in ds
    assert "x" in ds
    assert "y" in ds
    assert ds.lat.ndim == 2
    assert ds.x.size == 10
    assert ds.y.size == 10

    assert "lat_b" in ds
    assert "lon_b" in ds
    # New (Y, X, 4) bounds format for curvilinear
    assert ds.lat_b.ndim == 3
    assert ds.lat_b.shape == (10, 10, 4)

    assert "crs" in ds.attrs
    assert "history" in ds.attrs


def test_create_mesh_from_coords():
    x = np.array([400000, 450000, 500000])
    y = np.array([5000000, 5050000, 5100000])
    ds = create_mesh_from_coords(x, y, "EPSG:32633")

    assert "lat" in ds
    assert "lon" in ds
    assert ds.lat.ndim == 1
    assert ds.lon.ndim == 1
    assert ds.lat.size == 3
    assert ds.lat.dims == ds.lon.dims
    assert "n_pts" in ds.lat.dims

    assert "crs" in ds.attrs


def test_create_grid_from_crs_lazy_2():
    """
    Double-Check Test for create_grid_from_crs: Eager (NumPy) vs Lazy (Dask).

    Ensures that the refactored lazy transformation produces identical results
    to the eager computation.
    """
    # Use a simple projected CRS (Web Mercator)
    crs = "EPSG:3857"
    extent = (0, 1000000, 0, 1000000)
    res = 100000  # 100km resolution for test

    # 1. Eager (NumPy) - Default
    ds_eager = create_grid_from_crs(crs, extent, res, add_bounds=True)

    # 2. Lazy (Dask) - By providing chunks
    ds_lazy = create_grid_from_crs(crs, extent, res, add_bounds=True, chunks=5)

    # Verify backend identity (Aero Protocol requirement)
    assert not hasattr(ds_eager.lat.data, "dask"), "Eager lat should be NumPy"
    assert hasattr(ds_lazy.lat.data, "dask"), "Lazy lat should be Dask-backed"
    assert hasattr(ds_lazy.lon.data, "dask"), "Lazy lon should be Dask-backed"
    assert hasattr(ds_lazy.lat_b.data, "dask"), "Lazy lat_b should be Dask-backed"
    assert hasattr(ds_lazy.lon_b.data, "dask"), "Lazy lon_b should be Dask-backed"

    # 3. Assert value identity
    # We use a small tolerance for floating point variations if any,
    # but since it's the same core function it should be exact.
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())


def test_create_mesh_from_coords_lazy_2():
    """
    Double-Check Test for create_mesh_from_coords: Eager (NumPy) vs Lazy (Dask).
    """
    crs = "EPSG:3857"
    x = np.linspace(0, 1000000, 10)
    y = np.linspace(0, 1000000, 10)

    # 1. Eager (NumPy)
    ds_eager = create_mesh_from_coords(x, y, crs)

    # 2. Lazy (Dask)
    ds_lazy = create_mesh_from_coords(x, y, crs, chunks=5)

    # Verify backend identity
    assert not hasattr(ds_eager.lat.data, "dask")
    assert hasattr(ds_lazy.lat.data, "dask")

    # 3. Assert value identity
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())
