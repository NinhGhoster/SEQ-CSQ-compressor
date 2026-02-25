# AGENTS.md — AI Coding Guidelines

## Project Overview
`thermal-compress` is a CLI tool for lossless compression of FLIR SEQ/CSQ radiometric thermal video into NetCDF4 format.

## Tech Stack
- **Language:** Python 3.12+
- **CLI:** Click
- **Data:** numpy, netCDF4, xarray, h5py
- **FLIR SDK:** `fnv` (proprietary, lazy-imported)
- **Concurrency:** `concurrent.futures.ProcessPoolExecutor`
- **Testing:** pytest

## Architecture Rules & Constraints
1. **Lazy Imports:** All FLIR SDK (`fnv`) imports must be **lazy** — wrapped in functions or guarded by try/except so the tool works without the SDK for non-encode tasks (`info`, `decode`, `verify`).
2. **Compression Isolation:** Compression settings live in `compression.py` as a frozen dataclass. Do not scatter magic numbers.
3. **Multiprocessing Limits:** The FLIR `fnv` library is a compiled C++ extension that is **strictly not thread-safe**. Frame reading must occur sequentially in the main process before being dispatched to the worker pool.
4. **HDF5 Direct Writes:** `netCDF4` and `h5py` compression pipelines are strictly single-threaded. To achieve multiprocessing, the `encoder.py` manually shuffles and zlib-compresses chunks in Python worker processes, then directly injects the compressed bytes into the file using `h5py.Dataset.id.write_direct_chunk()`.
5. **Fixed Dimensions:** To use HDF5 direct chunk writes, the `frame` dimension *must* have a fixed size upon creation. Do not use the `UNLIMITED` dimension size or `write_direct_chunk` will fail with an undefined address error.
6. **Metadata:** NetCDF4 files must always extract and include the global SDK attributes (camera model, lens, exact recording date from frame timestamps, and object parameters like emissivity and distance).
7. **Thresholding:** The `--threshold` behavior must mask cold pixels by rounding them to the nearest integer (`np.round()`), rather than using `NaN` or a fill value. This permanently preserves the file's visual colormap rendering in downstream tools like Firebrand Thermal Analysis, while drastically dropping the entropy for `zlib` compression. Hot pixels (`>= threshold`) must retain their full float precision.

## Data Schema (NetCDF4)
```
Dimensions:
  frame  = <fixed_num_frames>
  y      = <height>       (e.g. 512)
  x      = <width>        (e.g. 640)

Variables:
  temperature(frame, y, x) — float32 or scaled int16
    units:       "degC" (or "K")
    long_name:   "Radiometric temperature"
    chunksizes:  (10, height, width)
    scale_factor: 0.01 (if int16)
```

## Testing
- Tests use **synthetic numpy arrays** to avoid the proprietary FLIR SDK dependency in CI.
- Mock `fnv` in encoder/verify tests (see `tests/test_encoder.py`).
- Since direct chunk writing requires exact chunk boundaries, tests using tiny files must carefully simulate padding or clamp their chunks.
- Run: `pytest tests/ -v`

## CLI Layer
- The CLI (`cli.py`) is a thin wrapper. It only handles argument parsing and `click.echo()` output, delegating all logic to `encoder`, `decoder`, and `verify`.

## File Conventions
- Formatting: `black`, `isort`
- Type hints on all public functions
- Docstrings: Google style
