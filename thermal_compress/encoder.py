"""Encode SEQ/CSQ radiometric video → NetCDF4."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import netCDF4 as nc
import numpy as np
from tqdm import tqdm

from thermal_compress.compression import CompressionConfig, DEFAULT_CONFIG
from thermal_compress.metadata import extract_metadata, open_imager


def encode(
    input_path: str | Path,
    output_path: str | Path,
    *,
    emissivity: Optional[float] = None,
    experiment: Optional[str] = None,
    config: CompressionConfig = DEFAULT_CONFIG,
) -> Path:
    """Convert a single SEQ/CSQ file to compressed NetCDF4.

    Args:
        input_path: Path to the ``.seq`` or ``.csq`` source file.
        output_path: Destination ``.nc`` file path.
        emissivity: Surface emissivity to record as a global attribute.
        experiment: Free-text experiment description.
        config: Compression settings (complevel, int16, chunking, etc.).

    Returns:
        Path to the written NetCDF4 file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- open source file ------------------------------------------------
    im = open_imager(input_path)
    height, width = im.height, im.width
    num_frames = im.num_frames

    # --- metadata --------------------------------------------------------
    meta = extract_metadata(input_path)
    if emissivity is not None:
        meta["emissivity"] = float(emissivity)
    if experiment is not None:
        meta["experiment"] = experiment

    # --- create NetCDF4 --------------------------------------------------
    ds = nc.Dataset(str(output_path), "w", format="NETCDF4")

    try:
        # global attributes
        for key, value in meta.items():
            ds.setncattr(key, value)

        # dimensions
        ds.createDimension("frame", None)  # UNLIMITED
        ds.createDimension("y", height)
        ds.createDimension("x", width)

        # temperature variable
        var_kw = config.variable_kwargs(height, width)
        temp_var = ds.createVariable("temperature", **var_kw)
        for attr_name, attr_val in config.variable_attributes().items():
            temp_var.setncattr(attr_name, attr_val)

        # --- write frames ------------------------------------------------
        for idx in tqdm(range(num_frames), desc="Encoding", unit="frame"):
            im.get_frame(idx)
            frame = np.array(im.final, copy=True).reshape((height, width))

            # When use_int16 is True, the variable has scale_factor/add_offset
            # attributes.  netCDF4 auto-packs (divides by scale_factor) on
            # write, so we always write the raw temperature floats.
            temp_var[idx, :, :] = frame

    finally:
        ds.close()

    return output_path


def encode_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    emissivity: Optional[float] = None,
    experiment: Optional[str] = None,
    config: CompressionConfig = DEFAULT_CONFIG,
) -> list[Path]:
    """Batch-convert every SEQ/CSQ file in a directory.

    Args:
        input_dir: Directory containing ``.seq`` / ``.csq`` files.
        output_dir: Directory where ``.nc`` files will be written.
        emissivity: Surface emissivity (applied to all files).
        experiment: Experiment description (applied to all files).
        config: Compression settings.

    Returns:
        List of output file paths.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".seq", ".csq"}
    )

    if not seq_files:
        raise FileNotFoundError(f"No SEQ/CSQ files found in {input_dir}")

    outputs: list[Path] = []
    for seq_file in seq_files:
        out = output_dir / seq_file.with_suffix(".nc").name
        encode(
            seq_file, out,
            emissivity=emissivity,
            experiment=experiment,
            config=config,
        )
        outputs.append(out)

    return outputs
