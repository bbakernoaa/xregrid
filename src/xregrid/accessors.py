from __future__ import annotations

from typing import Any, Union

import xarray as xr

from xregrid.regridder import Regridder


@xr.register_dataarray_accessor("regrid")
class RegridDataArrayAccessor:
    """
    Xarray Accessor for regridding DataArrays.
    """

    def __init__(self, xarray_obj: xr.DataArray):
        """
        Initialize the DataArray regrid accessor.

        Parameters
        ----------
        xarray_obj : xr.DataArray
            The DataArray to regrid.
        """
        self._obj = xarray_obj

    def to(
        self, target_grid: Union[xr.Dataset, Regridder], **kwargs: Any
    ) -> xr.DataArray:
        """
        Regrid the DataArray to a target grid or using a pre-computed Regridder.

        Parameters
        ----------
        target_grid : xr.Dataset or Regridder
            The target grid dataset or an existing Regridder instance.
        **kwargs : Any
            Arguments passed to the Regridder constructor if target_grid is a Dataset.

        Returns
        -------
        xr.DataArray
            The regridded DataArray.
        """
        if isinstance(target_grid, Regridder):
            return target_grid(self._obj)

        # Convert DataArray to Dataset to ensure compatibility with Regridder
        # if the Regridder needs to inspect the source grid.
        source_ds = self._obj.to_dataset(name="_tmp_data")
        regridder = Regridder(source_ds, target_grid, **kwargs)
        return regridder(self._obj)

    def get_regridder(self, target_grid: xr.Dataset, **kwargs: Any) -> Regridder:
        """
        Create a Regridder instance for this DataArray.

        Parameters
        ----------
        target_grid : xr.Dataset
            The target grid dataset.
        **kwargs : Any
            Arguments passed to the Regridder constructor.

        Returns
        -------
        Regridder
            The initialized Regridder instance.
        """
        source_ds = self._obj.to_dataset(name="_tmp_data")
        return Regridder(source_ds, target_grid, **kwargs)

    def plot_diagnostics(
        self, target_grid: xr.Dataset, mode: str = "static", **kwargs: Any
    ) -> Any:
        """
        Visualize regridding diagnostics between this DataArray and a target grid.

        Parameters
        ----------
        target_grid : xr.Dataset
            The target grid dataset.
        mode : str, default 'static'
            The plotting mode: 'static' or 'interactive'.
        **kwargs : Any
            Arguments passed to Regridder.plot_diagnostics.

        Returns
        -------
        Any
            The plot object.
        """
        regridder = self.get_regridder(target_grid, **kwargs)
        return regridder.plot_diagnostics(mode=mode, **kwargs)

    def plot_weights(
        self, target_grid: xr.Dataset, row_idx: int, mode: str = "static", **kwargs: Any
    ) -> Any:
        """
        Visualize source points contributing to a specific destination point.

        Parameters
        ----------
        target_grid : xr.Dataset
            The target grid dataset.
        row_idx : int
            The index of the destination point.
        mode : str, default 'static'
            The plotting mode: 'static' or 'interactive'.
        **kwargs : Any
            Arguments passed to Regridder.plot_weights.

        Returns
        -------
        Any
            The plot object.
        """
        regridder = self.get_regridder(target_grid, **kwargs)
        return regridder.plot_weights(row_idx, mode=mode, **kwargs)

    def plot_comparison(
        self, target_grid: xr.Dataset, mode: str = "static", **kwargs: Any
    ) -> Any:
        """
        Unified comparison plot (Source, Target, Difference).

        Parameters
        ----------
        target_grid : xr.Dataset
            The target grid dataset.
        mode : str, default 'static'
            The plotting mode: 'static' or 'interactive'.
        **kwargs : Any
            Arguments passed to Regridder.plot_comparison.

        Returns
        -------
        Any
            The plot object.
        """
        regridder = self.get_regridder(target_grid, **kwargs)
        da_tgt = regridder(self._obj)
        return regridder.plot_comparison(self._obj, da_tgt, mode=mode, **kwargs)


@xr.register_dataset_accessor("regrid")
class RegridDatasetAccessor:
    """
    Xarray Accessor for regridding Datasets.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        """
        Initialize the Dataset regrid accessor.

        Parameters
        ----------
        xarray_obj : xr.Dataset
            The Dataset to regrid.
        """
        self._obj = xarray_obj

    def to(
        self, target_grid: Union[xr.Dataset, Regridder], **kwargs: Any
    ) -> xr.Dataset:
        """
        Regrid the Dataset to a target grid or using a pre-computed Regridder.

        Parameters
        ----------
        target_grid : xr.Dataset or Regridder
            The target grid dataset or an existing Regridder instance.
        **kwargs : Any
            Arguments passed to the Regridder constructor if target_grid is a Dataset.

        Returns
        -------
        xr.Dataset
            The regridded Dataset.
        """
        if isinstance(target_grid, Regridder):
            return target_grid(self._obj)

        regridder = Regridder(self._obj, target_grid, **kwargs)
        return regridder(self._obj)

    def get_regridder(self, target_grid: xr.Dataset, **kwargs: Any) -> Regridder:
        """
        Create a Regridder instance for this Dataset.

        Parameters
        ----------
        target_grid : xr.Dataset
            The target grid dataset.
        **kwargs : Any
            Arguments passed to the Regridder constructor.

        Returns
        -------
        Regridder
            The initialized Regridder instance.
        """
        return Regridder(self._obj, target_grid, **kwargs)

    def plot_diagnostics(
        self, target_grid: xr.Dataset, mode: str = "static", **kwargs: Any
    ) -> Any:
        """
        Visualize regridding diagnostics between this Dataset and a target grid.

        Parameters
        ----------
        target_grid : xr.Dataset
            The target grid dataset.
        mode : str, default 'static'
            The plotting mode: 'static' or 'interactive'.
        **kwargs : Any
            Arguments passed to Regridder.plot_diagnostics.

        Returns
        -------
        Any
            The plot object.
        """
        regridder = self.get_regridder(target_grid, **kwargs)
        return regridder.plot_diagnostics(mode=mode, **kwargs)

    def plot_weights(
        self, target_grid: xr.Dataset, row_idx: int, mode: str = "static", **kwargs: Any
    ) -> Any:
        """
        Visualize source points contributing to a specific destination point.

        Parameters
        ----------
        target_grid : xr.Dataset
            The target grid dataset.
        row_idx : int
            The index of the destination point.
        mode : str, default 'static'
            The plotting mode: 'static' or 'interactive'.
        **kwargs : Any
            Arguments passed to Regridder.plot_weights.

        Returns
        -------
        Any
            The plot object.
        """
        regridder = self.get_regridder(target_grid, **kwargs)
        return regridder.plot_weights(row_idx, mode=mode, **kwargs)

    def plot_comparison(
        self,
        target_grid: xr.Dataset,
        var_name: str,
        mode: str = "static",
        **kwargs: Any,
    ) -> Any:
        """
        Unified comparison plot (Source, Target, Difference) for a specific variable.

        Parameters
        ----------
        target_grid : xr.Dataset
            The target grid dataset.
        var_name : str
            The name of the variable to compare.
        mode : str, default 'static'
            The plotting mode: 'static' or 'interactive'.
        **kwargs : Any
            Arguments passed to Regridder.plot_comparison.

        Returns
        -------
        Any
            The plot object.
        """
        regridder = self.get_regridder(target_grid, **kwargs)
        da_src = self._obj[var_name]
        da_tgt = regridder(da_src)
        return regridder.plot_comparison(da_src, da_tgt, mode=mode, **kwargs)
