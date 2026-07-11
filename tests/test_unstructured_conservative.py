from __future__ import annotations

import dask.array as da
import numpy as np
import xarray as xr
from xregrid.regridder import Regridder


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

    # 1. Eager NumPy Path
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

    # 2. Lazy Dask Path (Serial weights application)
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
    # Start a local in-process cluster so mock esmpy module is inherited by the workers
    import dask.distributed

    cluster = dask.distributed.LocalCluster(n_workers=2, processes=False)
    client = dask.distributed.Client(cluster)
    try:
        regridder_parallel = Regridder(
            ds_src, ds_tgt, method="conservative", parallel=True
        )
        res_parallel = regridder_parallel(da_src_lazy)
        assert isinstance(res_parallel.data, da.Array)
        if is_mock:
            # Under mock ESMF, each of the 4 chunks gets its own mock weight of 1.0,
            # which is scaled to 0.5, so each of the 4 target cells gets weight 0.5.
            np.testing.assert_allclose(
                res_parallel.compute().values, [0.5, 0.5, 0.5, 0.5], rtol=1e-5
            )
            weights_sum_parallel = np.array(
                regridder_parallel.weights.sum(axis=1)
            ).flatten()
            np.testing.assert_allclose(
                weights_sum_parallel, [0.5, 0.5, 0.5, 0.5], rtol=1e-5
            )
        else:
            np.testing.assert_allclose(
                res_parallel.compute().values, np.ones(4), rtol=1e-5
            )
            weights_sum_parallel = np.array(
                regridder_parallel.weights.sum(axis=1)
            ).flatten()
            np.testing.assert_allclose(weights_sum_parallel, np.ones(4), rtol=1e-5)
    finally:
        client.close()
        cluster.close()
