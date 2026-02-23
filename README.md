# SEQ-CSQ-compressor / thermal-compress

Lossless compression of FLIR SEQ/CSQ radiometric thermal video into **NetCDF4** format.

Converts ~30 GB raw files to ~5–12 GB while preserving all radiometric temperature data, embedded metadata, and random-access playback.

## Features

- **Lossless** — every temperature value survives the round-trip
- **50–83% compression** via zlib + byte-shuffle (optionally scaled int16)
- **Self-describing** — camera model, emissivity, frame rate, experiment info baked into the file
- **Random access** — read any frame without decompressing the whole file
- **Universal** — output readable by Python/xarray, MATLAB, R, Julia

## Installation

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate thermal-compress

# Or pip
pip install -e .
```

> **Note:** The FLIR Science File SDK (`fnv`) is required for reading SEQ/CSQ files.
> Install the platform-specific wheel separately.

## Usage

```bash
# Convert a single file
thermal-compress encode input.seq -o output.nc \
  --emissivity 0.95 \
  --experiment "Stringybark, 50kW/m, Rep 3"

# Batch convert a folder
thermal-compress encode /path/to/seq_folder/ -o /path/to/output/ --batch

# Verify lossless round-trip
thermal-compress verify input.seq output.nc

# Inspect metadata
thermal-compress info output.nc

# Extract frames to numpy
thermal-compress decode output.nc -o frames/ --format npy
```

## Expected Compression

| Storage method | Size (15 min) | Reduction |
|---|---|---|
| Raw SEQ/CSQ | ~30 GB | — |
| NetCDF4 float32 + zlib + shuffle | ~10–12 GB | ~60–65% |
| NetCDF4 scaled int16 + zlib + shuffle | ~5–8 GB | ~73–83% |

## Project Structure

```
thermal_compress/
├── cli.py            # CLI entry point (Click)
├── encoder.py        # SEQ/CSQ → NetCDF4
├── decoder.py        # NetCDF4 → raw frames
├── verify.py         # Bitwise verification
├── metadata.py       # FLIR metadata extraction
└── compression.py    # Compression config
```

## Related

- [Firebrand Thermal Analysis](https://github.com/NinhGhoster/Firebrand-Thermal-Analysis) — dashboard for analysing the compressed files
