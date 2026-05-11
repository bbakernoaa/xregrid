# Scientific Hygiene

XRegrid is built on the **Aero Protocol**, a set of principles designed to ensure that Earth Science data processing remains flexible, maintainable, and scientifically robust. This guide details how XRegrid helps you maintain high standards of scientific hygiene.

## 1. Provenance Tracking

Automatically tracking the lineage of your data is critical for reproducibility. XRegrid automatically updates the `history` attribute of your xarray objects whenever a transformation occurs.

- **Weight Generation**: When a `Regridder` is initialized, it records the ESMF version, the method used, and any specific parameters (like periodicity or extrapolation).
- **Data Application**: Every time you call a regridder on a `DataArray` or `Dataset`, a timestamped message is prepended to the `history` attribute, detailing the backend used (Eager, Dask, or Cubed) and the specific regridding parameters.

```python
# View the history of a regridded object
print(regridded_da.attrs['history'])
```

## 2. Handling Missing Data (NaNs)

In many Earth Science datasets, "missing data" (represented by NaNs) must be handled carefully to avoid biasing results, especially during conservative regridding.

### Weight Re-normalization (`skipna=True`)

When `skipna=True` is set in the `Regridder`, XRegrid handles NaNs by re-normalizing the interpolation weights based only on the "valid" (non-NaN) source points.

- **Mechanism**: For every destination cell, XRegrid sums the weights of all contributing non-NaN source points. The interpolated value is then divided by this sum.
- **Stationary Mask Caching**: XRegrid is optimized for datasets where the mask is stationary over time (e.g., a fixed land-sea mask). It detects if the NaN locations are identical across time steps and caches the normalization factors to provide a ~2x speedup.

### Validation Threshold (`na_thres`)

Even with re-normalization, you may want to mask destination cells that don't have enough valid source data. The `na_thres` parameter (default 1.0) controls this:

- `na_thres=1.0`: Only mask destination cells that have **zero** valid source points.
- `na_thres=0.5`: Mask destination cells where less than 50% of the original weight sum is represented by valid source points.

```python
regridder = Regridder(ds_src, ds_tgt, skipna=True, na_thres=0.7)
```

## 3. Weight Diagnostics

You should always verify the quality of your regridding weights before trusting the results. XRegrid provides built-in tools for this.

### Spatial Diagnostics

The `.diagnostics()` method returns an xarray Dataset on the target grid containing:

- **`weight_sum`**: The sum of weights for each destination cell. For methods like bilinear or conservative, this should ideally be 1.0.
- **`unmapped_mask`**: A binary mask where 1 indicates a destination cell that does not overlap with any source cells.

```python
diag = regridder.diagnostics()
diag.weight_sum.plot()
```

### Quality Reports

The `.quality_report()` method provides a summary of the regridding quality:

```python
report = regridder.quality_report()
print(f"Unmapped fraction: {report['unmapped_fraction']:.2%}")
```

## 4. Coordinate Reference Systems (CRS)

XRegrid ensures that your data remains "geospatially aware" by propagating CRS metadata.

- **Automated Propagation**: If the target grid has a `crs` (WKT) or `grid_mapping` attribute, XRegrid automatically attaches it to the regridded output.
- **CF-Compliance**: XRegrid uses `cf-xarray` and `pyproj` to robustly identify and manage coordinate systems, ensuring compatibility with other geospatial tools.

## 5. Backend Agnosticism

Following the **"Optional Dask" Rule**, XRegrid functions are designed to work regardless of whether your data is backed by NumPy (Eager) or Dask/Cubed (Lazy).

- **No Hidden Computes**: XRegrid never calls `.compute()` or `.values` inside a processing function, ensuring that laziness is preserved for large-scale workflows.
- **Vectorized Logic**: Computations are written using `xarray.apply_ufunc` with `dask='parallelized'`, allowing the same code to run efficiently on single machines or distributed clusters.
