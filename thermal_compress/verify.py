"""Frame-by-frame verification of SEQ/CSQ ↔ NetCDF4 round-trip."""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np
from tqdm import tqdm

from thermal_compress.metadata import open_imager


def verify(
    seq_path: str | Path,
    nc_path: str | Path,
    *,
    atol: float = 1e-6,
    use_int16: bool = False,
    scale_factor: float = 0.01,
) -> bool:
    """Compare original SEQ/CSQ frames against a NetCDF4 file.

    Args:
        seq_path: Path to the original ``.seq`` / ``.csq`` file.
        nc_path: Path to the ``.nc`` file to verify.
        atol: Absolute tolerance for ``np.allclose``.  Use a larger value
            (e.g. ``scale_factor / 2``) when int16 scaling was used.
        use_int16: If True, the NetCDF values are scaled int16 and will be
            converted back via ``value * scale_factor`` before comparison.
        scale_factor: Scale factor used during int16 encoding.

    Returns:
        True if every frame matches within tolerance.

    Raises:
        AssertionError: On the first mismatched frame, with diagnostic info.
    """
    seq_path = Path(seq_path)
    nc_path = Path(nc_path)

    im = open_imager(seq_path)
    height, width = im.height, im.width
    num_frames = im.num_frames

    ds = nc.Dataset(str(nc_path), "r")
    try:
        temp = ds.variables["temperature"]

        nc_frames = temp.shape[0]
        assert nc_frames == num_frames, (
            f"Frame count mismatch: SEQ has {num_frames}, NC has {nc_frames}"
        )

        max_err: float = 0.0

        for idx in tqdm(range(num_frames), desc="Verifying", unit="frame"):
            im.get_frame(idx)
            original = np.array(im.final, copy=True).reshape((height, width))
            restored = np.array(temp[idx, :, :])

            # netCDF4 auto-applies scale_factor on read, so restored values
            # are already in the original temperature units.

            frame_err = float(np.max(np.abs(original - restored)))
            max_err = max(max_err, frame_err)

            if not np.allclose(original, restored, atol=atol):
                raise AssertionError(
                    f"Frame {idx} mismatch! "
                    f"Max absolute error: {frame_err:.8f} "
                    f"(tolerance: {atol})"
                )

        print(f"✓ All {num_frames} frames match (max error: {max_err:.8e})")
    finally:
        ds.close()

    return True
