from __future__ import annotations

import numpy as np
import pytest
from xregrid.utils import _compute_lazy_aware, is_lazy, is_dask

try:
    import dask.array as da

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

import importlib.util

HAS_DASK = importlib.util.find_spec("dask") is not None
HAS_CUBED = importlib.util.find_spec("cubed") is not None


def test_compute_lazy_aware_numpy():
    """Verify compute_lazy_aware handles eager NumPy data."""
    data = np.array([1, 2, 3])
    res = _compute_lazy_aware(data)
    assert np.array_equal(res, data)
    assert not is_lazy(res)


def test_compute_lazy_aware_dict_numpy():
    """Verify compute_lazy_aware handles dicts of eager data."""
    data = {"a": np.array([1]), "b": 2}
    res = _compute_lazy_aware(data)
    assert res == data
    assert res["a"] is data["a"]


@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
def test_compute_lazy_aware_dask():
    """Verify compute_lazy_aware handles Dask objects."""
    dask_arr = da.from_array(np.array([1, 2, 3]), chunks=2)
    assert is_dask(dask_arr)

    res = _compute_lazy_aware(dask_arr)
    assert np.array_equal(res, np.array([1, 2, 3]))
    assert not is_lazy(res)


@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
def test_compute_lazy_aware_dask_dict():
    """Verify compute_lazy_aware handles dicts of Dask objects."""
    dask_arr = da.from_array(np.array([1]), chunks=1)
    data = {"a": dask_arr, "b": 2}

    res = _compute_lazy_aware(data)
    assert res["a"] == 1
    assert res["b"] == 2
    assert not is_lazy(res["a"])


@pytest.mark.skipif(not HAS_CUBED, reason="Cubed not installed")
def test_compute_lazy_aware_cubed_mock(monkeypatch):
    """Verify compute_lazy_aware handles (mocked) Cubed objects."""
    # We can't easily create a real Cubed array without a spec/plan,
    # so we mock the cubed.compute call and is_cubed check.

    data_val = np.array([10])

    def mock_cubed_compute(*args, **kwargs):
        return [data_val for _ in args]

    monkeypatch.setattr("cubed.compute", mock_cubed_compute)

    # Create an object that passes is_cubed
    # In our implementation, is_cubed checks isinstance(obj, cubed.Array)
    # or DataArray.data is cubed.Array

    class FakeCubed:
        pass

    monkeypatch.setattr("cubed.Array", FakeCubed)

    obj = FakeCubed()
    # Ensure is_cubed returns True
    monkeypatch.setattr("xregrid.utils.is_cubed", lambda x: True)

    res = _compute_lazy_aware(obj)
    assert np.array_equal(res, data_val)


def test_compute_lazy_aware_generic_compute():
    """Verify compute_lazy_aware handles objects with a .compute() method."""

    class GenericLazy:
        def compute(self):
            return "computed"

    obj = GenericLazy()
    res = _compute_lazy_aware(obj)
    assert res == "computed"


if __name__ == "__main__":
    pytest.main([__file__])
