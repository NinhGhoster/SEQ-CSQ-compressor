"""Decode NetCDF4 → raw frames and inspect file metadata."""

from __future__ import annotations

import json
from pathlib import Path

import netCDF4 as nc
import numpy as np


def decode(
    nc_path: str | Path,
    output_dir: str | Path,
    *,
    fmt: str = "npy",
) -> list[Path]:
    """Extract temperature frames from a NetCDF4 file.

    Args:
        nc_path: Path to the ``.nc`` file.
        output_dir: Directory to write extracted frames.
        fmt: Output format — ``"npy"`` (default) or ``"csv"``.

    Returns:
        List of paths to extracted frame files.
    """
    nc_path = Path(nc_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = nc.Dataset(str(nc_path), "r")
    try:
        temp = ds.variables["temperature"]
        num_frames = temp.shape[0]

        outputs: list[Path] = []
        for idx in range(num_frames):
            frame = np.array(temp[idx, :, :])
            if fmt == "npy":
                out = output_dir / f"frame_{idx:06d}.npy"
                np.save(str(out), frame)
            elif fmt == "csv":
                out = output_dir / f"frame_{idx:06d}.csv"
                np.savetxt(str(out), frame, delimiter=",", fmt="%.4f")
            else:
                raise ValueError(f"Unsupported format: {fmt!r}. Use 'npy' or 'csv'.")
            outputs.append(out)
    finally:
        ds.close()

    return outputs


def info(nc_path: str | Path) -> dict:
    """Print and return metadata summary from a NetCDF4 file.

    Args:
        nc_path: Path to the ``.nc`` file.

    Returns:
        Dictionary of file metadata and shape info.
    """
    nc_path = Path(nc_path)
    ds = nc.Dataset(str(nc_path), "r")

    try:
        result: dict = {}

        # global attributes
        global_attrs = {}
        for attr in ds.ncattrs():
            val = ds.getncattr(attr)
            # Convert numpy types for clean display
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            global_attrs[attr] = val
        result["global_attributes"] = global_attrs

        # dimensions
        dims = {}
        for name, dim in ds.dimensions.items():
            dims[name] = len(dim)
        result["dimensions"] = dims

        # variable info
        if "temperature" in ds.variables:
            temp = ds.variables["temperature"]
            result["temperature"] = {
                "dtype": str(temp.dtype),
                "shape": list(temp.shape),
                "chunking": list(temp.chunking()) if temp.chunking() != "contiguous" else "contiguous",
            }
            # variable-level attributes
            var_attrs = {}
            for attr in temp.ncattrs():
                val = temp.getncattr(attr)
                if isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = float(val)
                var_attrs[attr] = val
            result["temperature"]["attributes"] = var_attrs
    finally:
        ds.close()

    # pretty-print
    print(json.dumps(result, indent=2))
    return result
