"""Tests for the decoder module."""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

from thermal_compress.decoder import decode, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_nc(path: Path, frames: np.ndarray, attrs: dict | None = None):
    """Write a minimal NetCDF4 file from a numpy array."""
    num_frames, height, width = frames.shape
    ds = nc.Dataset(str(path), "w", format="NETCDF4")
    ds.createDimension("frame", None)
    ds.createDimension("y", height)
    ds.createDimension("x", width)
    temp = ds.createVariable(
        "temperature", "f4", ("frame", "y", "x"),
        chunksizes=(min(10, num_frames), height, width),
        zlib=True, complevel=4, shuffle=True,
    )
    temp.units = "degC"
    temp.long_name = "Radiometric temperature"
    for i in range(num_frames):
        temp[i, :, :] = frames[i]
    if attrs:
        for k, v in attrs.items():
            ds.setncattr(k, v)
    ds.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDecode:
    """Decode NetCDF4 → frame files."""

    def test_decode_npy(self, tmp_path: Path):
        """Decoding to npy should produce one file per frame."""
        frames = np.random.rand(3, 4, 6).astype(np.float32)
        nc_file = tmp_path / "test.nc"
        _create_test_nc(nc_file, frames)

        out_dir = tmp_path / "frames"
        outputs = decode(nc_file, out_dir, fmt="npy")

        assert len(outputs) == 3
        for i, p in enumerate(outputs):
            assert p.exists()
            loaded = np.load(str(p))
            np.testing.assert_allclose(loaded, frames[i], atol=1e-6)

    def test_decode_csv(self, tmp_path: Path):
        """Decoding to csv should produce readable text files."""
        frames = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        nc_file = tmp_path / "test.nc"
        _create_test_nc(nc_file, frames)

        out_dir = tmp_path / "frames"
        outputs = decode(nc_file, out_dir, fmt="csv")

        assert len(outputs) == 1
        loaded = np.loadtxt(str(outputs[0]), delimiter=",")
        np.testing.assert_allclose(loaded, frames[0], atol=1e-4)

    def test_decode_bad_format(self, tmp_path: Path):
        """Unsupported format should raise ValueError."""
        frames = np.random.rand(1, 2, 2).astype(np.float32)
        nc_file = tmp_path / "test.nc"
        _create_test_nc(nc_file, frames)

        with pytest.raises(ValueError, match="Unsupported format"):
            decode(nc_file, tmp_path / "out", fmt="parquet")


class TestInfo:
    """Inspect NetCDF4 metadata."""

    def test_info_returns_metadata(self, tmp_path: Path, capsys):
        """info() should return a dict with dimensions and attributes."""
        frames = np.random.rand(5, 8, 10).astype(np.float32)
        nc_file = tmp_path / "test.nc"
        _create_test_nc(nc_file, frames, attrs={
            "source_file": "sample.seq",
            "camera_model": "TestCam",
        })

        result = info(nc_file)

        assert result["dimensions"]["frame"] == 5
        assert result["dimensions"]["y"] == 8
        assert result["dimensions"]["x"] == 10
        assert result["global_attributes"]["camera_model"] == "TestCam"
        assert result["temperature"]["dtype"] == "float32"
        assert result["temperature"]["attributes"]["units"] == "degC"

        # also check that it printed JSON
        captured = capsys.readouterr()
        assert '"camera_model": "TestCam"' in captured.out
