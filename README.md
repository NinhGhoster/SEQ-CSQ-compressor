# SEQ-CSQ-compressor / CSQ Compression

High-performance, parallelized compression tool for converting massive proprietary FLIR SEQ/CSQ radiometric thermal video files into the universally supported **NetCDF4** standard.

Reduces raw 37+ GB video files by up to **80%** (yielding ~7.5 GB files) while fully preserving all scientifically relevant thermal data, automatically embedding rich camera hardware metadata, and enabling instant random-access playback for analysis.

## Features

- **Extreme Speed** — encodes 37 GB videos in ~3.5 minutes using a custom `multiprocessing` worker pool that bypasses standard single-threaded HDF5 compression bottlenecks.
- **Three Precision Modes**:
  - **Lossless (Default)**: Every single temperature value survives the exact mathematical `float32` round-trip.
  - **Archival (`--int16`)**: Highly recommended scaled packing that cuts file sizes by roughly 50% while preserving a strict `0.01 °C` precision.
  - **Threshold Masking (`--threshold`)**: Extreme compression mode that rounds useless cold background pixels to whole integers (`1 °C` precision) while keeping full float precision for hot anomalies. Drops file sizes by up to 80% while perfectly preserving visual rendering!
- **50–80% compression** — via zlib + HDF5 byte-shuffle, using the optimized precision modes above.
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

### HPC Setup

The workflow below was used successfully on an HPC system for large-scale batch compression.

Example repository location:

```bash
/data/nguyen_h/proj-6600_3d_firebrand-1128.4.1067/code/SEQ-CSQ-compressor
```

Example SDK location in home:

```bash
/home/nguyen_h/SDK/FileSDK-2024.7.1-cp312-cp312-linux_x86_64.whl
```

Micromamba environment used:

```bash
export MAMBA_ROOT_PREFIX=/home/nguyen_h/micromamba
/home/nguyen_h/micromamba-bin/micromamba create -y -n thermal-compress -f environment.yml
/home/nguyen_h/micromamba-bin/micromamba run -n thermal-compress pip install /home/nguyen_h/SDK/FileSDK-2024.7.1-cp312-cp312-linux_x86_64.whl
/home/nguyen_h/micromamba-bin/micromamba run -n thermal-compress pip install -e .
```

Because the SDK wheel depends on newer runtime libraries than the system defaults, batch jobs may need to export:

```bash
ENV_PREFIX=/home/nguyen_h/micromamba/envs/thermal-compress
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/lib/python3.12/site-packages/fnv/_lib"
```

Smoke test:

```bash
$ENV_PREFIX/bin/python -c "import fnv; print('fnv_ok')"
$ENV_PREFIX/bin/thermal-compress --help
```

## Usage

```bash
# Convert a single file (maximum compression & speed)
thermal-compress encode input.seq -o output.nc \
  --emissivity 0.9 \
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

## HPC Batch Submission

For very large datasets, the fastest practical HPC strategy is a Slurm array with one thermal file per task, rather than one large monolithic job.

Recommended storage method for bulk compression:

```bash
thermal-compress encode input.seq -o output.nc --threshold 299
```

This is the `Threshold + float32` mode from the benchmark table. Do not add `--int16` if you want that exact mode.

### Example HPC Array Workflow

Source tree used:

```bash
/data/nguyen_h/proj-6600_3d_firebrand-1128.4.1067/level1/thermal
```

Output tree used:

```bash
/scratch/nguyen_h/thermal_compressed_threshold299/level1/thermal
```

Shared file list:

```bash
find /data/nguyen_h/proj-6600_3d_firebrand-1128.4.1067/level1/thermal \
  -type f \( -iname '*.seq' -o -iname '*.csq' \) \
  | LC_ALL=C sort > /scratch/nguyen_h/thermal_compressed_threshold299_file_list.txt
```

Example array script:

```bash
#!/bin/bash
#SBATCH --job-name=thermal299
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=intel
#SBATCH --time=24:00:00
#SBATCH --output=/home/nguyen_h/thermal_compress_threshold299-%A_%a.out

set -euo pipefail

ENV_PREFIX="/home/nguyen_h/micromamba/envs/thermal-compress"
THERMAL_COMPRESS="$ENV_PREFIX/bin/thermal-compress"
SOURCE_ROOT="/data/nguyen_h/proj-6600_3d_firebrand-1128.4.1067/level1/thermal"
OUTPUT_ROOT="/scratch/nguyen_h/thermal_compressed_threshold299/level1/thermal"
FILE_LIST="/scratch/nguyen_h/thermal_compressed_threshold299_file_list.txt"

export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/lib/python3.12/site-packages/fnv/_lib"

src=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$FILE_LIST")
rel="${src#${SOURCE_ROOT}/}"
out="$OUTPUT_ROOT/${rel%.*}.nc"

mkdir -p "$(dirname "$out")"
"$THERMAL_COMPRESS" encode "$src" -o "$out" --threshold 299 --workers "${SLURM_CPUS_PER_TASK:-4}"
```

Example submission with 64 concurrent tasks:

```bash
COUNT=$(wc -l < /scratch/nguyen_h/thermal_compressed_threshold299_file_list.txt)
sbatch --array=0-$((COUNT-1))%64 thermal_compress_threshold299_array.sbatch
```

Why this is better than one large job:

- spreads work across many nodes
- improves cluster-wide throughput
- makes retries simpler if a few files fail
- preserves the input folder structure in the output tree

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
> 3. **Deflate (`zlib` level 9)**: The sequences are compressed using universally compatible `zlib` at maximum compression. Although benchmarks show level 6 and level 9 produce nearly identical file sizes for thermal data (due to the chaotic entropy of physical thermal noise), level 9 is used by default to squeeze every last byte.
## Project Architecture

```
SEQ-CSQ-compressor/
├── cli.py            # CLI entry point (Click)
├── encoder.py        # SEQ/CSQ → NetCDF4 (Multiprocessing & h5py direct chunk writes)
├── decoder.py        # NetCDF4 → raw frames back-conversion
├── verify.py         # Bitwise roundtrip verification
├── metadata.py       # Rich FLIR metadata extraction
└── compression.py    # Compression logic configuration
```

## Context
- This tool is designed as a companion to the [Firebrand Thermal Analysis](https://github.com/NinhGhoster/Firebrand-Thermal-Analysis) repository to drastically reduce the size of the required dataset inputs.
