from __future__ import annotations

import numpy as np
import xarray as xr
import pytest
import dask.array as da
from dask.callbacks import Callback
from xregrid import Regridder


class ComputeCounter(Callback):
    def __init__(self):
        self.count = 0

    def _start(self, dsk):
        self.count += 1


def test_regridder_periodicity_eager():
    """Verify periodicity detection on eager NumPy grids (Track A)."""
    # Create an eager 2D curvilinear grid
    lon_raw = np.linspace(0, 359, 100)
    lat_raw = np.linspace(-90, 90, 50)
    lon_2d, lat_2d = np.meshgrid(lon_raw, lat_raw)

    ds_src = xr.Dataset({"lon": (("y", "x"), lon_2d), "lat": (("y", "x"), lat_2d)})
    ds_src.lon.attrs["units"] = "degrees_east"
    ds_src.lat.attrs["units"] = "degrees_north"

    # Target grid
    ds_tgt = xr.Dataset(
        {
            "lon": (["lon"], np.linspace(0, 359, 10), {"units": "degrees_east"}),
            "lat": (["lat"], np.linspace(-90, 90, 10), {"units": "degrees_north"}),
        }
    )

    # Eager grids should NOT warn and should detect periodicity
    regridder = Regridder(ds_src, ds_tgt, method="bilinear")
    assert regridder.periodic is True


def test_regridder_periodicity_lazy():
    """Verify periodicity detection on lazy Dask grids (Track B)."""
    # Create a lazy 2D curvilinear grid
    lon_raw = np.linspace(0, 359, 100)
    lat_raw = np.linspace(-90, 90, 50)
    lon_2d, lat_2d = np.meshgrid(lon_raw, lat_raw)

    # Chunk it to make it Dask-backed
    lon_da = xr.DataArray(
        da.from_array(lon_2d, chunks=(50, 50)), dims=("y", "x"), name="lon"
    )
    lat_da = xr.DataArray(
        da.from_array(lat_2d, chunks=(50, 50)), dims=("y", "x"), name="lat"
    )

    ds_src = xr.Dataset({"lon": lon_da, "lat": lat_da})
    ds_src.lon.attrs["units"] = "degrees_east"
    ds_src.lat.attrs["units"] = "degrees_north"

    # Target grid
    ds_tgt = xr.Dataset(
        {
            "lon": (["lon"], np.linspace(0, 359, 10), {"units": "degrees_east"}),
            "lat": (["lat"], np.linspace(-90, 90, 10), {"units": "degrees_north"}),
        }
    )

    # Case 1: Lazy curvilinear triggers a warning
    with pytest.warns(
        UserWarning, match="Triggering hidden compute in _detect_periodicity"
    ):
        regridder = Regridder(ds_src, ds_tgt, method="bilinear")
    assert regridder.periodic is True

    # Case 2: Metadata-based avoids compute and warning
    ds_src_meta = ds_src.copy()
    ds_src_meta.lon.attrs["boundary"] = "periodic"

    # We mock _generate_weights to isolate periodicity detection computes
    from unittest.mock import patch

    with patch("xregrid.regridder.Regridder._generate_weights"):
        counter = ComputeCounter()
        with counter:
            regridder = Regridder(ds_src_meta, ds_tgt, method="bilinear")
        assert regridder.periodic is True
        assert counter.count == 0, (
            f"Metadata-based detection triggered {counter.count} computes"
        )

    # Case 3: Explicit periodicity avoids compute and warning
    with patch("xregrid.regridder.Regridder._generate_weights"):
        counter = ComputeCounter()
        with counter:
            regridder = Regridder(ds_src, ds_tgt, method="bilinear", periodic=True)
        assert regridder.periodic is True
        assert counter.count == 0, (
            f"Explicit periodicity triggered {counter.count} computes"
        )


if __name__ == "__main__":
    pytest.main([__file__])
