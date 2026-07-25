from __future__ import annotations

import numpy as np
import xarray as xr

from xregrid.utils import _compute_lazy_aware, is_cubed, is_dask, is_lazy


def test_is_dask_and_is_cubed_nested() -> None:
    """
    Test recursive detection of dask and cubed arrays in nested structures.
    """
    # 1. Eager arrays
    da_eager = xr.DataArray(np.arange(10))
    nested_eager = {"a": [da_eager], "b": (da_eager,)}

    assert not is_dask(da_eager)
    assert not is_dask(nested_eager)
    assert not is_cubed(da_eager)
    assert not is_cubed(nested_eager)
    assert not is_lazy(nested_eager)

    # 2. Lazy (Dask) arrays
    da_lazy = da_eager.chunk(5)
    nested_lazy = {"a": [da_lazy], "b": (da_lazy,)}

    assert is_dask(da_lazy)
    assert is_dask(nested_lazy)
    assert is_lazy(nested_lazy)

    # Cubed is not installed or active, but shouldn't raise errors
    assert not is_cubed(da_lazy)
    assert not is_cubed(nested_lazy)


def test_compute_lazy_aware_eager_and_lazy() -> None:
    """
    Test the _compute_lazy_aware utility for eager and lazy data.

    Verifies the logic twice:
    - Once with eager data (returns immediately without changing structure)
    - Once with lazy (Dask) data (fully computes and returns eager data)
    """
    # 1. Eager NumPy Path
    eager_val = xr.DataArray(np.arange(5))
    eager_nested = {"a": eager_val, "b": [eager_val, 42], "c": (eager_val,)}

    res_eager = _compute_lazy_aware(eager_nested)
    assert not is_lazy(res_eager)
    assert isinstance(res_eager["a"], xr.DataArray)
    np.testing.assert_array_equal(res_eager["a"].values, np.arange(5))
    np.testing.assert_array_equal(res_eager["b"][0].values, np.arange(5))
    assert res_eager["b"][1] == 42
    np.testing.assert_array_equal(res_eager["c"][0].values, np.arange(5))

    # 2. Lazy Dask Path
    lazy_val = eager_val.chunk(2)
    lazy_nested = {"a": lazy_val, "b": [lazy_val, 42], "c": (lazy_val,)}

    assert is_lazy(lazy_nested)

    res_lazy = _compute_lazy_aware(lazy_nested)
    assert not is_lazy(res_lazy)
    assert isinstance(res_lazy["a"], xr.DataArray)
    np.testing.assert_array_equal(res_lazy["a"].values, np.arange(5))
    np.testing.assert_array_equal(res_lazy["b"][0].values, np.arange(5))
    assert res_lazy["b"][1] == 42
    np.testing.assert_array_equal(res_lazy["c"][0].values, np.arange(5))
