"""FLIR metadata extraction.

The ``fnv`` package (FLIR Science File SDK) is lazily imported so that
modules which do not touch SEQ/CSQ files can work without the SDK.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _import_fnv():
    """Lazy-import the FLIR SDK, raising a clear error if missing."""
    try:
        import fnv
        import fnv.file  # submodule must be imported explicitly
        return fnv
    except ImportError as exc:
        raise ImportError(
            "The FLIR Science File SDK (fnv) is required to read SEQ/CSQ files. "
            "Install the platform-specific wheel from FLIR."
        ) from exc


def extract_metadata(path: str | Path) -> dict[str, Any]:
    """Read camera / recording metadata from a SEQ/CSQ file.

    Extracts information from the SDK's ``source_info`` (camera, lens,
    recording date) and ``object_parameters`` (emissivity, distance, etc.).
    Frame rate is derived from the timestamps of the first two frames.

    Args:
        path: Path to the ``.seq`` or ``.csq`` file.

    Returns:
        Dictionary of global attributes suitable for embedding in NetCDF4.
    """
    fnv = _import_fnv()
    im = fnv.file.ImagerFile(str(path))
    im.unit = fnv.Unit.TEMPERATURE_FACTORY
    im.temp_type = fnv.TempType.CELSIUS

    meta: dict[str, Any] = {
        "source_file": Path(path).name,
        "width": im.width,
        "height": im.height,
        "num_frames": im.num_frames,
        "software": "thermal-compress v1.0",
        "compression": "netcdf4-zlib-shuffle",
    }

    # --- source_info (camera / lens / recording date) --------------------
    try:
        si = im.source_info
        meta["camera_model"] = getattr(si, "camera_model", "unknown")
        meta["camera_serial"] = getattr(si, "camera_serial", "")
        meta["camera_part_number"] = getattr(si, "camera_part_number", "")
        meta["lens"] = getattr(si, "lens", "")
        meta["lens_part_number"] = getattr(si, "lens_part_number", "")
        meta["lens_serial"] = getattr(si, "lens_serial", "")
        if getattr(si, "time_valid", False) and si.time:
            meta["date"] = si.time.isoformat()
        else:
            meta["date"] = datetime.now(tz=timezone.utc).isoformat()
    except Exception:
        meta["camera_model"] = "unknown"
        meta["date"] = datetime.now(tz=timezone.utc).isoformat()

    # --- object_parameters (emissivity, distance, atmosphere) ------------
    try:
        op = im.object_parameters
        meta["emissivity_original"] = float(getattr(op, "emissivity", 0.0))
        meta["distance"] = float(getattr(op, "distance", 0.0))
        meta["relative_humidity"] = float(getattr(op, "relative_humidity", 0.0))
        meta["reflected_temp"] = float(getattr(op, "reflected_temp", 0.0))
        meta["atmosphere_temp"] = float(getattr(op, "atmosphere_temp", 0.0))
        meta["atmospheric_transmission"] = float(
            getattr(op, "atmospheric_transmission", 0.0)
        )
    except Exception:
        pass

    # --- frame_rate (derived from first two frame timestamps) ------------
    try:
        if im.num_frames >= 2:
            im.get_frame(0)
            t0 = im.frame_info.time
            im.get_frame(1)
            t1 = im.frame_info.time
            dt = (t1 - t0).total_seconds()
            if dt > 0:
                meta["frame_rate"] = round(1.0 / dt, 2)
    except Exception:
        pass

    return meta


def open_imager(path: str | Path):
    """Open a SEQ/CSQ file and return a configured ``ImagerFile``.

    The imager is set to return temperature values in Celsius.

    Args:
        path: Path to the ``.seq`` or ``.csq`` file.

    Returns:
        Configured ``fnv.file.ImagerFile`` instance.
    """
    fnv = _import_fnv()
    im = fnv.file.ImagerFile(str(path))
    im.unit = fnv.Unit.TEMPERATURE_FACTORY
    im.temp_type = fnv.TempType.CELSIUS
    return im
