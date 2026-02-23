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
        import fnv  # type: ignore[import-untyped]
        return fnv
    except ImportError as exc:
        raise ImportError(
            "The FLIR Science File SDK (fnv) is required to read SEQ/CSQ files. "
            "Install the platform-specific wheel from FLIR."
        ) from exc


def extract_metadata(path: str | Path) -> dict[str, Any]:
    """Read camera / recording metadata from a SEQ/CSQ file.

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
        "camera_model": getattr(im, "camera_model", "unknown"),
        "frame_rate": float(getattr(im, "frame_rate", 0.0)),
        "width": im.width,
        "height": im.height,
        "num_frames": im.num_frames,
        "date": datetime.now(tz=timezone.utc).isoformat(),
        "software": "thermal-compress v1.0",
        "compression": "netcdf4-zlib-shuffle",
    }
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
