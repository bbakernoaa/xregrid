from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from xregrid.regridder import Regridder


def create_polygon_grid(n_corners: int) -> xr.Dataset:
    """
    Generate an unstructured grid of regular polygons with `n_corners` sides.
    """
    centers = [(0.5, 0.5), (1.5, 0.5)]
    r = 0.4

    lat_b = []
    lon_b = []
    for c_lon, c_lat in centers:
        angles = np.linspace(0, 2 * np.pi, n_corners, endpoint=False)
        lon_corners = c_lon + r * np.cos(angles)
        lat_corners = c_lat + r * np.sin(angles)
        lat_b.append(lat_corners)
        lon_b.append(lon_corners)

    lat_b = np.array(lat_b)
    lon_b = np.array(lon_b)

    lat_c = np.array([0.5, 0.5])
    lon_c = np.array([0.5, 1.5])

    return xr.Dataset(
        coords={
            "lat": (["grid_size"], lat_c),
            "lon": (["grid_size"], lon_c),
            "lat_b": (["grid_size", "n_corners"], lat_b),
            "lon_b": (["grid_size", "n_corners"], lon_b),
        }
    )


@pytest.mark.parametrize("n_corners", [3, 4, 5, 6, 7, 8])
def test_unstructured_conservative_weight_scaling_properties(n_corners: int):
    """
    Verify conservative weight scaling on unstructured meshes with regular
    polygons of varying sides (3 to 8) under both mock and real ESMF.
    """
    import esmpy

    is_mock = hasattr(esmpy, "_is_mock")

    # Regular polygon target mesh
    ds_tgt = create_polygon_grid(n_corners)

    # Rectilinear source grid covering [0, 2] degrees
    lat_src = np.array([0.5, 1.5])
    lon_src = np.array([0.5, 1.5])
    lat_src_b = np.array([0.0, 1.0, 2.0])
    lon_src_b = np.array([0.0, 1.0, 2.0])

    ds_src = xr.Dataset(
        coords={
            "lat": (["lat"], lat_src),
            "lon": (["lon"], lon_src),
            "lat_b": (["lat_b"], lat_src_b),
            "lon_b": (["lon_b"], lon_src_b),
        }
    )

    # Input constant field of ones
    da_src = xr.DataArray(
        np.ones((2, 2)),
        dims=["lat", "lon"],
        coords={
            "lat": ds_src.lat,
            "lon": ds_src.lon,
        },
    )

    # Eager NumPy Path
    regridder = Regridder(ds_src, ds_tgt, method="conservative")
    res_eager = regridder(da_src)

    if is_mock:
        # Under mock ESMF, each polygon of N sides is split into N-2 triangles.
        # So the expected weight row sum is exactly 1 / (N - 2).
        expected_val = 1.0 / (n_corners - 2)
        np.testing.assert_allclose(res_eager.values, [expected_val, 0.0], rtol=1e-5)
        weights_sum = np.array(regridder.weights.sum(axis=1)).flatten()
        np.testing.assert_allclose(weights_sum, [expected_val, 0.0], rtol=1e-5)
    else:
        # Under real ESMF, output values must be exactly 1.0 (fully covered and normalized)
        np.testing.assert_allclose(res_eager.values, np.ones(2), rtol=1e-5)
        weights_sum = np.array(regridder.weights.sum(axis=1)).flatten()
        np.testing.assert_allclose(weights_sum, np.ones(2), rtol=1e-5)


def test_unstructured_conservative_weight_scaling():
    """
    Verify that conservative regridding on an unstructured destination mesh
    does not silently double or scale values, and that weight rows sum to 1.
    """
    import esmpy

    is_mock = hasattr(esmpy, "_is_mock")

    # SCRIP style target grid (unstructured mesh of quadrilaterals)
    # 4 cells, each having 4 corners
    lat_b = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],  # cell 0
            [0.0, 1.0, 1.0, 0.0],  # cell 1
            [1.0, 2.0, 2.0, 1.0],  # cell 2
            [1.0, 2.0, 2.0, 1.0],  # cell 3
        ]
    )
    lon_b = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],  # cell 0
            [1.0, 1.0, 2.0, 2.0],  # cell 1
            [0.0, 0.0, 1.0, 1.0],  # cell 2
            [1.0, 1.0, 2.0, 2.0],  # cell 3
        ]
    )
    lat_c = np.array([0.5, 0.5, 1.5, 1.5])
    lon_c = np.array([0.5, 1.5, 0.5, 1.5])

    ds_tgt = xr.Dataset(
        coords={
            "lat": (["grid_size"], lat_c),
            "lon": (["grid_size"], lon_c),
            "lat_b": (["grid_size", "n_corners"], lat_b),
            "lon_b": (["grid_size", "n_corners"], lon_b),
        }
    )

    # Rectilinear source grid covering [0, 2] degrees
    lat_src = np.array([0.5, 1.5])
    lon_src = np.array([0.5, 1.5])
    lat_src_b = np.array([0.0, 1.0, 2.0])
    lon_src_b = np.array([0.0, 1.0, 2.0])

    ds_src = xr.Dataset(
        coords={
            "lat": (["lat"], lat_src),
            "lon": (["lon"], lon_src),
            "lat_b": (["lat_b"], lat_src_b),
            "lon_b": (["lon_b"], lon_src_b),
        }
    )

    # Input constant field of ones
    da_src = xr.DataArray(
        np.ones((2, 2)),
        dims=["lat", "lon"],
        coords={
            "lat": ds_src.lat,
            "lon": ds_src.lon,
        },
    )

    # 1. Eager NumPy Path (Verifies weight scaling math on both real and mock ESMF)
    regridder = Regridder(ds_src, ds_tgt, method="conservative")
    res_eager = regridder(da_src)

    if is_mock:
        # Under mock ESMF, the single mock weight (1.0) is mapped to cell 0
        # and scaled by the element area fraction (1/2 = 0.5)
        np.testing.assert_allclose(res_eager.values, [0.5, 0.0, 0.0, 0.0], rtol=1e-5)
        weights_sum = np.array(regridder.weights.sum(axis=1)).flatten()
        np.testing.assert_allclose(weights_sum, [0.5, 0.0, 0.0, 0.0], rtol=1e-5)
    else:
        # Under real ESMF, output values must be exactly 1.0 (fully covered and normalized)
        np.testing.assert_allclose(res_eager.values, np.ones(4), rtol=1e-5)
        weights_sum = np.array(regridder.weights.sum(axis=1)).flatten()
        np.testing.assert_allclose(weights_sum, np.ones(4), rtol=1e-5)

    # 2. Lazy Dask Path (Verifies serial/Dask weights application on both real and mock ESMF)
    da_src_lazy = da_src.chunk({"lat": 1, "lon": 1})
    res_lazy = regridder(da_src_lazy)
    assert isinstance(res_lazy.data, da.Array)
    if is_mock:
        np.testing.assert_allclose(
            res_lazy.compute().values, [0.5, 0.0, 0.0, 0.0], rtol=1e-5
        )
    else:
        np.testing.assert_allclose(res_lazy.compute().values, np.ones(4), rtol=1e-5)

    # 3. Parallel Dask Path
    # Since parallel weight generation using LocalCluster with real ESMF on standard GitHub Actions
    # runners triggers C-level ESMF/MPI threading/forking limitations, we run this verification
    # in the mock environment (where Dask parallel workers run safely without C-level restrictions).
    if is_mock:
        import dask.distributed

        cluster = dask.distributed.LocalCluster(
            n_workers=1, threads_per_worker=1, processes=False
        )
        client = dask.distributed.Client(cluster)
        try:
            regridder_parallel = Regridder(
                ds_src, ds_tgt, method="conservative", parallel=True
            )
            res_parallel = regridder_parallel(da_src_lazy)
            assert isinstance(res_parallel.data, da.Array)
            # Under mock ESMF with n_workers=1, 2 chunks are created (size 2 each).
            # Each chunk gets its own mock weight of 1.0, scaled to 0.5,
            # so cell 0 and cell 2 get weight 0.5.
            np.testing.assert_allclose(
                res_parallel.compute().values, [0.5, 0.0, 0.5, 0.0], rtol=1e-5
            )
            weights_sum_parallel = np.array(
                regridder_parallel.weights.sum(axis=1)
            ).flatten()
            np.testing.assert_allclose(
                weights_sum_parallel, [0.5, 0.0, 0.5, 0.0], rtol=1e-5
            )
        finally:
            client.close()
            cluster.close()
