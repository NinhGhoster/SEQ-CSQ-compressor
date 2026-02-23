# AGENTS.md — AI Coding Guidelines

## Project Overview
`thermal-compress` is a CLI tool for lossless compression of FLIR SEQ/CSQ radiometric thermal video into NetCDF4 format.

## Tech Stack
- **Language:** Python 3.12+
- **CLI:** Click
- **Data:** numpy, netCDF4, xarray, h5py
- **FLIR SDK:** `fnv` (proprietary, lazy-imported)
- **Testing:** pytest

## Architecture Rules
1. All FLIR SDK (`fnv`) imports must be **lazy** — wrapped in functions or guarded by try/except so the tool works without the SDK for non-encode tasks.
2. Compression settings live in `compression.py` as a dataclass. Do not scatter magic numbers.
3. The CLI (`cli.py`) is a thin layer — it parses arguments and delegates to `encoder`, `decoder`, `verify`.
4. NetCDF4 files must always include the global attributes defined in `THERMAL_COMPRESSION_SPEC.md`.

## Testing
- Tests use **synthetic numpy arrays** to avoid FLIR SDK dependency.
- Mock `fnv` in encoder/verify tests.
- Run: `pytest tests/ -v`

## File Conventions
- Formatting: `black`, `isort`
- Type hints on all public functions
- Docstrings: Google style
