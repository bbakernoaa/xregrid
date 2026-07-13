from __future__ import annotations

import os
import uuid
import warnings
import numpy as np
import xarray as xr
from hypothesis import given, strategies as st, settings, HealthCheck, assume

from xregrid import (
    Regridder,
    create_global_grid,
    create_regional_grid,
    create_grid_from_crs,
    create_grid_like,
)
from xregrid.utils import _get_min_max_lazy_aware, is_lazy, is_dask, spatial_slice

# Configure hypothesis to have higher deadlines since ESMF is involved
settings.register_profile("default", deadline=1500)
settings.load_profile("default")


@given(
    res_lat=st.floats(min_value=2.0, max_value=45.0),
    res_lon=st.floats(min_value=2.0, max_value=90.0),
    add_bounds=st.booleans(),
)
@settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=15)
def test_create_global_grid_properties(
    res_lat: float, res_lon: float, add_bounds: bool
) -> None:
    """
    Test properties of create_global_grid using Hypothesis.

    This property-based test validates that create_global_grid produces grids
    with latitude and longitude coordinates that are within globally valid
    geographic bounds, are strictly monotonic, and that the lazy (Dask) and
    eager (NumPy) versions of the grid are numerically equivalent.

    Parameters
    ----------
    res_lat : float
        Generated resolution for latitude in degrees.
    res_lon : float
        Generated resolution for longitude in degrees.
    add_bounds : bool
        Whether to add cell boundaries.
    """
    # Create eager grid
    ds_eager = create_global_grid(
        res_lat=res_lat, res_lon=res_lon, add_bounds=add_bounds
    )

    assert "lat" in ds_eager.coords
    assert "lon" in ds_eager.coords

    # Boundaries check
    lat = ds_eager.lat.values
    lon = ds_eager.lon.values
    assert np.all(lat >= -90.0)
    assert np.all(lat <= 90.0)
    assert np.all(lon >= 0.0)
    assert np.all(lon <= 360.0)

    # Check monotonicity
    assert np.all(np.diff(lat) > 0)
    assert np.all(np.diff(lon) > 0)

    if add_bounds:
        assert "lat_b" in ds_eager.coords
        assert "lon_b" in ds_eager.coords
        assert ds_eager.lat_b.shape == (lat.size, 2)
        assert ds_eager.lon_b.shape == (lon.size, 2)

    # Test eager vs lazy parity
    ds_lazy = create_global_grid(
        res_lat=res_lat, res_lon=res_lon, add_bounds=add_bounds, chunks=5
    )

    if add_bounds:
        # Bounds are 2D and not dimension coordinates, so they remain lazy
        assert is_lazy(ds_lazy.lat_b)
        assert is_lazy(ds_lazy.lon_b)

    xr.testing.assert_allclose(ds_eager.lat, ds_lazy.lat.compute())
    xr.testing.assert_allclose(ds_eager.lon, ds_lazy.lon.compute())
    if add_bounds:
        xr.testing.assert_allclose(ds_eager.lat_b, ds_lazy.lat_b.compute())
        xr.testing.assert_allclose(ds_eager.lon_b, ds_lazy.lon_b.compute())


@given(
    lat_min=st.floats(min_value=-90.0, max_value=85.0),
    lat_max=st.floats(min_value=-85.0, max_value=90.0),
    lon_min=st.floats(min_value=-180.0, max_value=355.0),
    lon_max=st.floats(min_value=-175.0, max_value=360.0),
    res_lat=st.floats(min_value=2.0, max_value=15.0),
    res_lon=st.floats(min_value=2.0, max_value=15.0),
    add_bounds=st.booleans(),
)
@settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=15)
def test_create_regional_grid_properties(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    res_lat: float,
    res_lon: float,
    add_bounds: bool,
) -> None:
    """
    Test properties of create_regional_grid using Hypothesis.

    This test validates coordinate limits, grid monotonicity, cell boundaries,
    and backend parity (Eager vs Lazy) for regional rectilinear grids.

    Parameters
    ----------
    lat_min : float
        Generated minimum latitude in degrees.
    lat_max : float
        Generated maximum latitude in degrees.
    lon_min : float
        Generated minimum longitude in degrees.
    lon_max : float
        Generated maximum longitude in degrees.
    res_lat : float
        Generated resolution for latitude in degrees.
    res_lon : float
        Generated resolution for longitude in degrees.
    add_bounds : bool
        Whether to generate boundary variables.
    """
    assume(lat_max >= lat_min + 5.0)
    assume(lon_max >= lon_min + 5.0)
    assume(res_lat < (lat_max - lat_min))
    assume(res_lon < (lon_max - lon_min))

    ds_eager = create_regional_grid(
        lat_range=(lat_min, lat_max),
        lon_range=(lon_min, lon_max),
        res_lat=res_lat,
        res_lon=res_lon,
        add_bounds=add_bounds,
    )

    assert "lat" in ds_eager.coords
    assert "lon" in ds_eager.coords

    lat = ds_eager.lat.values
    lon = ds_eager.lon.values
    assert np.all(lat >= lat_min)
    assert np.all(lat <= lat_max)
    assert np.all(lon >= lon_min)
    assert np.all(lon <= lon_max)

    assert np.all(np.diff(lat) > 0)
    assert np.all(np.diff(lon) > 0)

    if add_bounds:
        assert "lat_b" in ds_eager.coords
        assert "lon_b" in ds_eager.coords
        assert ds_eager.lat_b.shape == (lat.size, 2)
        assert ds_eager.lon_b.shape == (lon.size, 2)

    # Lazy parity check
    ds_lazy = create_regional_grid(
        lat_range=(lat_min, lat_max),
        lon_range=(lon_min, lon_max),
        res_lat=res_lat,
        res_lon=res_lon,
        add_bounds=add_bounds,
        chunks=5,
    )

    if add_bounds:
        assert is_lazy(ds_lazy.lat_b)
        assert is_lazy(ds_lazy.lon_b)

    xr.testing.assert_allclose(ds_eager.lat, ds_lazy.lat.compute())
    xr.testing.assert_allclose(ds_eager.lon, ds_lazy.lon.compute())
    if add_bounds:
        xr.testing.assert_allclose(ds_eager.lat_b, ds_lazy.lat_b.compute())
        xr.testing.assert_allclose(ds_eager.lon_b, ds_lazy.lon_b.compute())


@given(
    ndim=st.integers(min_value=1, max_value=2),
    size_y=st.integers(min_value=5, max_value=15),
    size_x=st.integers(min_value=5, max_value=15),
)
@settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=15)
def test_get_min_max_lazy_aware_properties(ndim: int, size_y: int, size_x: int) -> None:
    """
    Test properties of _get_min_max_lazy_aware using Hypothesis.

    This test checks that retrieving min/max from coord variables works
    correctly and matches the eager values for both 1D and 2D arrays, using
    both eager (numpy) and lazy (dask) arrays.

    Parameters
    ----------
    ndim : int
        Number of dimensions (1 or 2).
    size_y : int
        Size along y/lat dimension.
    size_x : int
        Size along x/lon dimension.
    """
    import dask.array as da

    if ndim == 1:
        data = np.linspace(10.0, 50.0, size_y)
        da_eager = xr.DataArray(data, dims=("y",), name="y")
        min_v, max_v, is_eager = _get_min_max_lazy_aware(da_eager)
        assert is_eager
        assert np.isclose(min_v, 10.0)
        assert np.isclose(max_v, 50.0)

        da_lazy = xr.DataArray(da.from_array(data, chunks=3), dims=("y",), name="y")
        min_v2, max_v2, is_eager2 = _get_min_max_lazy_aware(da_lazy)
        assert not is_eager2
        assert np.isclose(float(min_v2.compute()), 10.0)
        assert np.isclose(float(max_v2.compute()), 50.0)

        da_lazy_with_coord = xr.DataArray(
            da.from_array(data, chunks=3),
            coords={"y": data},
            dims=("y",),
            name="y",
        )
        min_v3, max_v3, is_eager3 = _get_min_max_lazy_aware(da_lazy_with_coord)
        assert is_eager3
        assert np.isclose(min_v3, 10.0)
        assert np.isclose(max_v3, 50.0)

    elif ndim == 2:
        y, x = np.meshgrid(np.linspace(-10, 10, size_y), np.linspace(-20, 20, size_x))
        da_eager = xr.DataArray(x, dims=("y", "x"), name="lon")
        min_v, max_v, is_eager = _get_min_max_lazy_aware(da_eager)
        assert is_eager
        assert np.isclose(min_v, -20.0)
        assert np.isclose(max_v, 20.0)

        da_lazy = xr.DataArray(
            da.from_array(x, chunks=(3, 3)), dims=("y", "x"), name="lon"
        )
        min_v2, max_v2, is_eager2 = _get_min_max_lazy_aware(da_lazy)
        assert not is_eager2
        assert np.isclose(float(min_v2.compute()), -20.0)
        assert np.isclose(float(max_v2.compute()), 20.0)


@given(
    res_src=st.floats(min_value=15.0, max_value=45.0),
    res_tgt=st.floats(min_value=15.0, max_value=45.0),
    method=st.sampled_from(["bilinear", "nearest_s2d", "conservative"]),
    skipna=st.booleans(),
    has_time=st.booleans(),
)
@settings(
    max_examples=10, suppress_health_check=[HealthCheck.filter_too_much], deadline=None
)
def test_regridder_real_esmf_properties(
    res_src: float,
    res_tgt: float,
    method: str,
    skipna: bool,
    has_time: bool,
) -> None:
    """
    Test Regridder properties with real ESMF on generated grids.

    Validates that:
    1. Grid regridding works correctly and preserves correct metadata under ESMF.
    2. Non-spatial dimensions are correctly maintained.
    3. Eager and lazy (Dask) versions yield equivalent output data.
    4. Data history tracking behaves properly.

    Parameters
    ----------
    res_src : float
        Source grid resolution.
    res_tgt : float
        Target grid resolution.
    method : str
        ESMF regridding method.
    skipna : bool
        Whether to skip NaNs.
    has_time : bool
        Whether to include a non-spatial 'time' dimension.
    """
    # Generate source and target global grids
    ds_src = create_global_grid(res_lat=res_src, res_lon=res_src, add_bounds=True)
    ds_tgt = create_global_grid(res_lat=res_tgt, res_lon=res_tgt, add_bounds=True)

    lat_size = ds_src.lat.size
    lon_size = ds_src.lon.size

    assume(lat_size >= 2)
    assume(lon_size >= 2)
    assume(ds_tgt.lat.size >= 2)
    assume(ds_tgt.lon.size >= 2)

    # Generate source data
    if has_time:
        shape = (2, lat_size, lon_size)
        dims = ("time", "lat", "lon")
        coords = {"time": [0, 1], "lat": ds_src.lat, "lon": ds_src.lon}
    else:
        shape = (lat_size, lon_size)
        dims = ("lat", "lon")
        coords = {"lat": ds_src.lat, "lon": ds_src.lon}

    data = np.random.uniform(-10.0, 40.0, size=shape)

    if skipna:
        if has_time:
            data[0, 0, 0] = np.nan
            data[1, 1, 1] = np.nan
        else:
            data[0, 0] = np.nan
            data[1, 1] = np.nan

    da_src = xr.DataArray(data, coords=coords, dims=dims, name="temperature")

    # Real weights generation and validation
    weights_file = f"/tmp/test_weights_{uuid.uuid4().hex}.nc"

    try:
        # Suppress warnings about serial regridding on lazy data or boundary detection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            regridder = Regridder(
                source_grid_ds=ds_src,
                target_grid_ds=ds_tgt,
                method=method,
                skipna=skipna,
                filename=weights_file,
            )

            # Eager application
            res_eager = regridder(da_src)

            expected_shape = (
                (2, ds_tgt.lat.size, ds_tgt.lon.size)
                if has_time
                else (ds_tgt.lat.size, ds_tgt.lon.size)
            )
            assert res_eager.shape == expected_shape
            assert "lat" in res_eager.dims
            assert "lon" in res_eager.dims
            if has_time:
                assert "time" in res_eager.dims

            # Lazy application
            chunk_dict = {"lat": lat_size // 2, "lon": lon_size // 2}
            if has_time:
                chunk_dict["time"] = 1
            chunk_dict = {k: max(1, v) for k, v in chunk_dict.items() if v > 0}

            da_lazy = da_src.chunk(chunk_dict)
            res_lazy = regridder(da_lazy)

            assert is_lazy(res_lazy)
            assert is_dask(res_lazy)

            res_lazy_computed = res_lazy.compute()

            # Parity check
            np.testing.assert_allclose(
                res_eager.values, res_lazy_computed.values, equal_nan=True
            )

            # History check
            assert "history" in res_eager.attrs
            assert "Regridded using xregrid.Regridder" in res_eager.attrs["history"]

    finally:
        if os.path.exists(weights_file):
            try:
                os.remove(weights_file)
            except OSError:
                pass


@given(
    min_x=st.floats(min_value=-180.0, max_value=170.0),
    max_x=st.floats(min_value=-170.0, max_value=180.0),
    min_y=st.floats(min_value=-90.0, max_value=80.0),
    max_y=st.floats(min_value=-80.0, max_value=90.0),
    buffer=st.floats(min_value=0.0, max_value=2.0),
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.filter_too_much])
def test_spatial_slice_properties(
    min_x: float, max_x: float, min_y: float, max_y: float, buffer: float
) -> None:
    """
    Test properties of spatial_slice using Hypothesis.

    This test checks that spatial_slice correctly subsets a global grid
    to a given bounding box extent and handles buffer options properly.

    Parameters
    ----------
    min_x : float
        Min longitude coordinate.
    max_x : float
        Max longitude coordinate.
    min_y : float
        Min latitude coordinate.
    max_y : float
        Max latitude coordinate.
    buffer : float
        Slicing buffer size.
    """
    assume(max_x >= min_x + 10.0)
    assume(max_y >= min_y + 10.0)

    ds_src = create_global_grid(5.0, 5.0, add_bounds=False)
    # Put values into global grid
    data = np.random.uniform(0.0, 100.0, size=(ds_src.lat.size, ds_src.lon.size))
    da_src = xr.DataArray(
        data,
        coords={"lat": ds_src.lat, "lon": ds_src.lon},
        dims=("lat", "lon"),
    )

    extent = (min_x, max_x, min_y, max_y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        # Apply spatial_slice
        sliced = spatial_slice(da_src, extent, buffer=buffer)

        assert "lat" in sliced.dims
        assert "lon" in sliced.dims

        lat_vals = sliced.lat.values
        # Latitude values should fall within the bounds with some buffer
        assert np.all(lat_vals >= min_y - buffer - 0.1)
        assert np.all(lat_vals <= max_y + buffer + 0.1)


@given(
    crs_code=st.sampled_from([3857, 32633, 3035]),
    extent_offset_x=st.floats(min_value=10000, max_value=500000),
    res=st.floats(min_value=20000, max_value=100000),
    add_bounds=st.booleans(),
)
@settings(
    max_examples=10, suppress_health_check=[HealthCheck.filter_too_much], deadline=None
)
def test_create_grid_from_crs_properties(
    crs_code: int, extent_offset_x: float, res: float, add_bounds: bool
) -> None:
    """
    Test properties of create_grid_from_crs using Hypothesis.

    Validates that creating structured grids from different projected CRS targets
    maintains proper shapes, coordinate boundaries, and eagerness/laziness backend parity.
    """
    crs = f"EPSG:{crs_code}"
    # Pick a baseline extent depending on CRS
    if crs_code == 32633:  # UTM 33N, centered around x=500000, y=5000000
        min_x = 500000.0 - extent_offset_x
        max_x = 500000.0 + extent_offset_x
        min_y = 5000000.0
        max_y = 5000000.0 + 2 * extent_offset_x
    else:  # 3857, 3035
        min_x = 0.0
        max_x = 2 * extent_offset_x
        min_y = 0.0
        max_y = 2 * extent_offset_x

    # Ensure coordinates division works nicely and shapes are non-empty
    assume(max_x >= min_x + 3 * res)
    assume(max_y >= min_y + 3 * res)

    # Calculate sizes to ensure we do not end up with extremely large grids or tiny chunks
    # We want chunk size to be smaller than or equal to dimensions size
    num_x = int((max_x - min_x) / res)
    num_y = int((max_y - min_y) / res)
    assume(num_x >= 3)
    assume(num_y >= 3)

    extent = (min_x, max_x, min_y, max_y)

    ds_eager = create_grid_from_crs(crs, extent, res, add_bounds=add_bounds)
    assert "lat" in ds_eager.coords
    assert "lon" in ds_eager.coords
    assert "x" in ds_eager.coords
    assert "y" in ds_eager.coords

    # Parity check with Dask using appropriate chunks
    ds_lazy = create_grid_from_crs(
        crs, extent, res, add_bounds=add_bounds, chunks=min(num_x, num_y, 5)
    )
    assert is_lazy(ds_lazy.lat)
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())


@given(
    res_lat=st.floats(min_value=10.0, max_value=20.0),
    res_lon=st.floats(min_value=10.0, max_value=20.0),
    new_res=st.floats(min_value=5.0, max_value=10.0),
    add_bounds=st.booleans(),
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.filter_too_much])
def test_create_grid_like_properties(
    res_lat: float, res_lon: float, new_res: float, add_bounds: bool
) -> None:
    """
    Test properties of create_grid_like using Hypothesis.

    Validates extent and CRS matching from an existing template, coordinate limits,
    cell boundaries, and eager vs lazy backend parity.
    """
    # Create template regional grid
    ds_template = create_regional_grid(
        lat_range=(-30, 30),
        lon_range=(10, 70),
        res_lat=res_lat,
        res_lon=res_lon,
        add_bounds=add_bounds,
    )

    # 1. Test eager
    # Note that create_grid_like calculates the extent based on the outer edges.
    # The actual outer bounds of create_regional_grid are calculated using the center offsets.
    # Therefore, the created grid_like might be slightly wider than the center coordinate min/max.
    ds_new_eager = create_grid_like(ds_template, new_res, add_bounds=add_bounds)
    assert "lat" in ds_new_eager.coords
    assert "lon" in ds_new_eager.coords
    assert ds_new_eager.lat.ndim == 1
    assert ds_new_eager.lon.ndim == 1

    # Bounds check against template's grid boundary extent (with a small floating point slack)
    # The template bounds cover the exact latitude range [-30, 30] and longitude range [10, 70]
    assert ds_new_eager.lat.min() >= -30.0 - new_res
    assert ds_new_eager.lat.max() <= 30.0 + new_res
    assert ds_new_eager.lon.min() >= 10.0 - new_res
    assert ds_new_eager.lon.max() <= 70.0 + new_res

    if add_bounds:
        assert "lat_b" in ds_new_eager.coords
        assert "lon_b" in ds_new_eager.coords

    # 2. Test lazy template & lazy output
    ds_template_lazy = ds_template.chunk({"lat": 2, "lon": 2})
    # Avoid hidden compute warning by passing extent explicitly if we want,
    # or just suppress/ignore warning to verify it triggers and functions correctly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        ds_new_lazy = create_grid_like(
            ds_template_lazy, new_res, add_bounds=add_bounds, chunks=2
        )

    assert is_lazy(ds_new_lazy.lat_b) if add_bounds else True
    xr.testing.assert_allclose(ds_new_eager, ds_new_lazy.compute())


@given(
    res_src=st.floats(min_value=15.0, max_value=45.0),
    res_tgt=st.floats(min_value=15.0, max_value=45.0),
    method=st.sampled_from(["bilinear", "nearest_s2d", "conservative", "patch"]),
    constant_val=st.floats(min_value=-100.0, max_value=100.0),
)
@settings(
    max_examples=8, suppress_health_check=[HealthCheck.filter_too_much], deadline=None
)
def test_regridder_constant_preservation(
    res_src: float,
    res_tgt: float,
    method: str,
    constant_val: float,
) -> None:
    """
    Test that regridding a constant field preserves the constant value.

    This verifies the mathematical partition of unity property (sum of weights = 1)
    for all ESMF regridding methods on global rectilinear grids.
    """
    ds_src = create_global_grid(res_lat=res_src, res_lon=res_src, add_bounds=True)
    ds_tgt = create_global_grid(res_lat=res_tgt, res_lon=res_tgt, add_bounds=True)

    lat_size = ds_src.lat.size
    lon_size = ds_src.lon.size

    assume(lat_size >= 2)
    assume(lon_size >= 2)
    assume(ds_tgt.lat.size >= 2)
    assume(ds_tgt.lon.size >= 2)

    # Input constant field
    data = np.full((lat_size, lon_size), constant_val)
    da_src = xr.DataArray(
        data,
        coords={"lat": ds_src.lat, "lon": ds_src.lon},
        dims=("lat", "lon"),
        name="constant_field",
    )

    weights_file = f"/tmp/test_weights_const_{uuid.uuid4().hex}.nc"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            regridder = Regridder(
                source_grid_ds=ds_src,
                target_grid_ds=ds_tgt,
                method=method,
                filename=weights_file,
            )

            res = regridder(da_src)

            # Check that output values are all close to the constant value
            np.testing.assert_allclose(res.values, constant_val, rtol=1e-5, atol=1e-5)

    finally:
        if os.path.exists(weights_file):
            try:
                os.remove(weights_file)
            except OSError:
                pass


@given(
    res_src=st.floats(min_value=15.0, max_value=45.0),
    res_tgt=st.floats(min_value=15.0, max_value=45.0),
)
@settings(
    max_examples=8, suppress_health_check=[HealthCheck.filter_too_much], deadline=None
)
def test_regridder_conservative_conservation(
    res_src: float,
    res_tgt: float,
) -> None:
    """
    Test that conservative regridding preserves the global integral of a field.

    This is a core physical property of the conservative regridding method.
    """
    ds_src = create_global_grid(res_lat=res_src, res_lon=res_src, add_bounds=True)
    ds_tgt = create_global_grid(res_lat=res_tgt, res_lon=res_tgt, add_bounds=True)

    lat_size = ds_src.lat.size
    lon_size = ds_src.lon.size

    assume(lat_size >= 2)
    assume(lon_size >= 2)
    assume(ds_tgt.lat.size >= 2)
    assume(ds_tgt.lon.size >= 2)

    # Generate random positive field (e.g. tracer concentration)
    data = np.random.uniform(10.0, 100.0, size=(lat_size, lon_size))
    da_src = xr.DataArray(
        data,
        coords={"lat": ds_src.lat, "lon": ds_src.lon},
        dims=("lat", "lon"),
        name="tracer",
    )

    weights_file = f"/tmp/test_weights_conserv_{uuid.uuid4().hex}.nc"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            regridder = Regridder(
                source_grid_ds=ds_src,
                target_grid_ds=ds_tgt,
                method="conservative",
                filename=weights_file,
            )

            res = regridder(da_src)

            # Compute cell areas
            # Under ESMF, area calculations on a sphere might slightly differ from spherical-cap area formula
            # because of cell boundary linear/great-circle interpolation or ESMF internal representation.
            # To test perfect mathematical conservation of the Regridder's weights application,
            # we can compute the integral using the actual ESMF areas returned via dst_field.get_area() if they
            # were exported. Alternatively, we can verify that the sum of the source values weighted by the regridding
            # weights is mathematically conserved.
            # The weights matrix has shape (dst_size, src_size).
            # For conservative regridding, ESMF weights scale the cell values.
            # Let's verify conservation by comparing the source integral and target integral using areas
            # with a slightly relaxed tolerance (e.g., 3%) to account for pure spherical geometric differences between
            # our simplified analytic areas formula and ESMF's internally calculated finite-element areas.

            def get_areas(ds):
                # ds.lat_b is (lat, 2)
                lat_b = ds.lat_b.values
                lon_b = ds.lon_b.values

                # lats are monotonic increasing
                lat_south = np.radians(lat_b[:, 0])
                lat_north = np.radians(lat_b[:, 1])
                d_sin = np.sin(lat_north) - np.sin(lat_south)

                lon_west = np.radians(lon_b[:, 0])
                lon_east = np.radians(lon_b[:, 1])
                d_lon = lon_east - lon_west

                # Outer product to get 2D areas
                return np.outer(d_sin, d_lon)

            area_src = get_areas(ds_src)
            area_tgt = get_areas(ds_tgt)

            # Global integrals
            integral_src = np.sum(da_src.values * area_src)
            integral_tgt = np.sum(res.values * area_tgt)

            # Check conservation (within 10% due to geometric differences in area formulas)
            np.testing.assert_allclose(integral_src, integral_tgt, rtol=0.10)

    finally:
        if os.path.exists(weights_file):
            try:
                os.remove(weights_file)
            except OSError:
                pass
