# SEQ-CSQ-compressor / thermal-compress

Lossless compression of FLIR SEQ/CSQ radiometric thermal video into **NetCDF4** format.

Converts ~30 GB raw files to ~5–12 GB while preserving all radiometric temperature data, embedded metadata, and random-access playback.

## Features

- **Extreme Speed** — encodes 37 GB videos in ~3.5 minutes using a custom `multiprocessing` worker pool that bypasses standard single-threaded HDF5 compression bottlenecks.
- **Two Precision Modes**:
  - **Lossless (Default)**: Every single temperature value survives the exact mathematical `float32` round-trip.
  - **Archival (`--int16`)**: Highly recommended scaled packing that cuts file sizes by roughly 50% while preserving a strict `0.01 °C` precision (well within the physical noise floor of FLIR cameras).
- **50–83% compression** — via zlib + HDF5 byte-shuffle.
- **Self-describing Metadata** — automatically embeds the exact `camera_model`, `lens`, `emissivity`, optical `distance`, `relative_humidity`, and frame rate directly into the NetCDF headers using the proprietary FLIR SDK.
- **Random access** — read any individual frame instantly without decompressing the whole file.
- **Universal** — `.nc` output is natively readable by Python (`xarray`, `netCDF4`), MATLAB, R, and Julia.

## Installation

```bash
# Conda (recommended)
git clone https://github.com/NinhGhoster/SEQ-CSQ-compressor.git
cd "SEQ-CSQ-compressor"

conda env create -f environment.yml
conda activate thermal-compress
pip install -e .
```

> **Note:** The FLIR Science File SDK (`fnv`) is required for reading and encoding SEQ/CSQ files. Install the platform-specific wheel separately.
> ```bash
> pip install fnv-<version>-<platform>.whl
> ```

## Usage

```bash
# Convert a single file (Best compression & speed)
thermal-compress encode input.seq -o output.nc \
  --emissivity 0.9 \
  --experiment "Stringybark, 50kW/m, Rep 3" \
  --workers 12 \
  --int16

# Run a quick 2000-frame benchmark on your machine
thermal-compress encode input.seq -o output.nc --workers 12 --limit 2000

# Batch convert an entire folder
thermal-compress encode /path/to/seq_folder/ -o /path/to/output/ --batch --workers 12

# Verify lossless round-trip
thermal-compress verify input.seq output.nc

# Inspect metadata (hardware stats, humidity, distance, etc)
thermal-compress info output.nc

# Extract frames to standard numpy arrays
thermal-compress decode output.nc -o frames/ --format npy
```

## Expected Compression & Benchmarks

*Tested on an Apple M2 Max reading a 37 GB (25,000 frames) `B9.seq` file.*

| Storage method | Precision | Encoding Time | File Size | Reduction |
|---|---|---|---|---|
| **Raw SEQ/CSQ** | Lossless | — | 37 GB | 0% |
| **Float32 default** (`--workers 1`) | Lossless | ~22 minutes | 20 GB | 46% |
| **Float32 multi** (`--workers 10`) | Lossless | **3m 42s** | 20 GB | 46% |
| **Scaled int16** (`--workers 10 --int16`) | 0.01 °C | *~3m 30s* | **11 GB** | **70%** |

> **Why `--int16`?** Using the `--int16` flag is highly recommended for bulk archival. It drops the precision of the output past the second decimal place (e.g., `21.4391` becomes `21.44`). Since the thermal noise floor (NETD) of high-end FLIR cameras like the T1040 is `~0.02 °C`, discarding these extra noisy decimal digits retains 100% of the true physical sensor data, while drastically boosting `zlib` compression ratios by roughly 50%.

> **The Compression Pipeline:** When you use `--int16`, the tool runs a three-stage compression pass:
> 1. **Quantization (`int16`)**: `float32` temperatures are multiplied by 100 and stored as 16-bit integers, halving the raw footprint.
> 2. **HDF5 Shuffle Filter**: The bytes of these integers are grouped algorithmically to create long repeating sequences.
> 3. **Deflate (`zlib` level 6)**: The sequences are compressed using universally compatible `zlib`. Level 6 is the engineered "sweet spot." Benchmarks explicitly show that increasing this to Level 9 maxes out CPU time but yields **0% additional file size improvement** due to the chaotic entropy of physical thermal noise.
## Project Architecture

```
thermal_compress/
├── cli.py            # CLI entry point (Click)
├── encoder.py        # SEQ/CSQ → NetCDF4 (Multiprocessing & h5py direct chunk writes)
├── decoder.py        # NetCDF4 → raw frames back-conversion
├── verify.py         # Bitwise roundtrip verification
├── metadata.py       # Rich FLIR metadata extraction
└── compression.py    # Compression logic configuration
```

## Context
- This tool is designed as a companion to the [Firebrand Thermal Analysis](https://github.com/NinhGhoster/Firebrand-Thermal-Analysis) repository to drastically reduce the size of the required dataset inputs.
