# SEQ-CSQ-compressor / thermal-compress

High-performance, parallelized compression tool for converting massive proprietary FLIR SEQ/CSQ radiometric thermal video files into the universally supported **NetCDF4** standard.

Reduces raw 37+ GB video files by up to **96%** (yielding ~1.3 GB files) while fully preserving all scientifically relevant thermal data, automatically embedding rich camera hardware metadata, and enabling instant random-access playback for analysis.

## Features

- **Extreme Speed** — encodes 37 GB videos in ~3.5 minutes using a custom `multiprocessing` worker pool that bypasses standard single-threaded HDF5 compression bottlenecks.
- **Two Precision Modes**:
  - **Lossless (Default)**: Every single temperature value survives the exact mathematical `float32` round-trip.
  - **Archival (`--int16`)**: Highly recommended scaled packing that cuts file sizes by roughly 50% while preserving a strict `0.01 °C` precision (well within the physical noise floor of FLIR cameras).
- **50–96% compression** — via zlib + HDF5 byte-shuffle, with optional `--int16` scaling and `--threshold` masking for extreme reduction.
- **Self-describing Metadata** — automatically embeds the exact `camera_model`, `lens`, `emissivity`, optical `distance`, `relative_humidity`, and frame rate directly into the NetCDF headers using the proprietary FLIR SDK.
- **Random access** — read any individual frame instantly without decompressing the whole file.
- **Universal** — `.nc` output is natively readable by Python (`xarray`, `netCDF4`), MATLAB, R, and Julia.

### What Metadata is Preserved?
The tool physically extracts deep hardware and environmental parameters using the proprietary FLIR SDK and bakes them permanently into the NetCDF file's global attributes. Running `thermal-compress info` will output something like:

```json
{
  "global_attributes": {
    "source_file": "B9.seq",
    "width": 1024,
    "height": 768,
    "num_frames": 25028,
    "software": "thermal-compress v1.0",
    "compression": "netcdf4-zlib-shuffle",
    "camera_model": "FLIR T1040 45__",
    "camera_serial": "72503206",
    "camera_part_number": "72501-0303",
    "lens": "FOL21",
    "lens_part_number": "T198940",
    "lens_serial": "74000964",
    "date": "2025-08-02T07:19:23.845000",
    "emissivity_original": 0.94,
    "distance": 2.0,
    "relative_humidity": 0.5,
    "reflected_temp": 293.15,
    "atmosphere_temp": 293.15,
    "atmospheric_transmission": 0.991,
    "frame_rate": 14.93
  }
}
```

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
# Convert a single file (maximum compression & speed)
thermal-compress encode input.seq -o output.nc \
  --emissivity 0.95 \
  --experiment "Stringybark, 50kW/m, Rep 3" \
  --threshold 299 \
  --int16

# Run a quick 2000-frame benchmark
thermal-compress encode input.seq -o output.nc --limit 2000 --threshold 299 --int16

# Batch convert an entire folder
thermal-compress encode /path/to/seq_folder/ -o /path/to/output/ --batch

# Verify lossless round-trip
thermal-compress verify input.seq output.nc

# Inspect metadata (hardware stats, humidity, distance, etc)
thermal-compress info output.nc

# Extract frames to standard numpy arrays
thermal-compress decode output.nc -o frames/ --format npy
```

> **Note:** The encoder automatically uses all available CPU cores. Use `--workers N` to override if needed.

## Expected Compression & Benchmarks

*Tested on an Apple M2 Max reading a 37 GB (25,000 frames) `B9.seq` file.*

| Storage method | Precision | File Size | Reduction |
|---|---|---|---|
| **Raw SEQ/CSQ** | — | 37 GB | 0% |
| **Float32** (default) | Lossless | 20 GB | 46% |
| **Scaled int16** (`--int16`) | 0.01 °C | 11 GB | 70% |
| **Threshold + float32** (`--threshold 299`) | 1 °C background | **~7.5 GB** | **80%** |

> **How `--threshold` works:** Pixels **below** the threshold are rounded to the nearest whole degree (e.g., `25.77°C` → `26°C`). This preserves the full visual colormap in tools like FTA — you can still see the entire scene. Pixels **at or above** the threshold keep their full precision for scientific analysis. The rounding creates many fewer unique values, which `zlib` compresses dramatically better.

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
