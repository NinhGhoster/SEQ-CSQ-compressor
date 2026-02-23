"""Tests for the encoder module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import netCDF4 as nc
import numpy as np
import pytest

from thermal_compress.compression import CompressionConfig
from thermal_compress.encoder import encode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_imager(frames: np.ndarray):
    """Build a mock ``fnv.file.ImagerFile`` that yields *frames*.

    Args:
        frames: 3-D array shaped (num_frames, height, width).
    """
    num_frames, height, width = frames.shape
    im = MagicMock()
    im.height = height
    im.width = width
    im.num_frames = num_frames
    im.camera_model = "MockCam"
    im.frame_rate = 30.0

    flat_frames = [f.ravel().tolist() for f in frames]

    def _get_frame(idx):
        im.final = flat_frames[idx]

    im.get_frame = _get_frame
    return im


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEncode:
    """Basic encode tests using synthetic data and mocked FLIR SDK."""

    def test_encode_creates_nc_file(self, tmp_path: Path):
        """Encoding should produce a valid NetCDF4 file with correct shape."""
        frames = np.random.rand(5, 8, 10).astype(np.float32) * 100
        mock_im = _make_mock_imager(frames)

        fake_seq = tmp_path / "test.seq"
        fake_seq.touch()
        out_nc = tmp_path / "output.nc"

        with (
            patch("thermal_compress.encoder.open_imager", return_value=mock_im),
            patch("thermal_compress.encoder.extract_metadata", return_value={
                "source_file": "test.seq",
                "camera_model": "MockCam",
                "software": "thermal-compress v1.0",
            }),
        ):
            result = encode(fake_seq, out_nc)

        assert result == out_nc
        assert out_nc.exists()

        ds = nc.Dataset(str(out_nc), "r")
        try:
            assert "temperature" in ds.variables
            temp = ds.variables["temperature"]
            assert temp.shape == (5, 8, 10)
            assert temp.units == "degC"
        finally:
            ds.close()

    def test_encode_float32_data_matches(self, tmp_path: Path):
        """Float32 encoding should produce near-exact output."""
        frames = np.random.rand(3, 4, 6).astype(np.float32) * 200 - 50
        mock_im = _make_mock_imager(frames)

        fake_seq = tmp_path / "input.seq"
        fake_seq.touch()
        out_nc = tmp_path / "output.nc"

        with (
            patch("thermal_compress.encoder.open_imager", return_value=mock_im),
            patch("thermal_compress.encoder.extract_metadata", return_value={}),
        ):
            encode(fake_seq, out_nc)

        ds = nc.Dataset(str(out_nc), "r")
        try:
            for i in range(3):
                restored = np.array(ds.variables["temperature"][i, :, :])
                np.testing.assert_allclose(restored, frames[i], atol=1e-5)
        finally:
            ds.close()

    def test_encode_int16_reduces_precision(self, tmp_path: Path):
        """int16 encoding should round to scale_factor precision."""
        frames = np.array([[[25.123, 30.456], [15.789, 40.012]]], dtype=np.float32)
        mock_im = _make_mock_imager(frames)

        fake_seq = tmp_path / "input.seq"
        fake_seq.touch()
        out_nc = tmp_path / "output.nc"

        config = CompressionConfig(use_int16=True, scale_factor=0.01)

        with (
            patch("thermal_compress.encoder.open_imager", return_value=mock_im),
            patch("thermal_compress.encoder.extract_metadata", return_value={}),
        ):
            encode(fake_seq, out_nc, config=config)

        ds = nc.Dataset(str(out_nc), "r")
        try:
            temp = ds.variables["temperature"]
            # On disk the dtype is int16; netCDF4 auto-unpacks to float on read
            temp.set_auto_scale(False)
            assert temp.dtype == np.int16
        finally:
            ds.close()

    def test_encode_global_attributes(self, tmp_path: Path):
        """Custom emissivity and experiment should appear as global attrs."""
        frames = np.random.rand(1, 2, 2).astype(np.float32)
        mock_im = _make_mock_imager(frames)

        fake_seq = tmp_path / "input.seq"
        fake_seq.touch()
        out_nc = tmp_path / "output.nc"

        with (
            patch("thermal_compress.encoder.open_imager", return_value=mock_im),
            patch("thermal_compress.encoder.extract_metadata", return_value={
                "source_file": "input.seq",
            }),
        ):
            encode(fake_seq, out_nc, emissivity=0.95, experiment="Test burn")

        ds = nc.Dataset(str(out_nc), "r")
        try:
            assert ds.getncattr("emissivity") == 0.95
            assert ds.getncattr("experiment") == "Test burn"
        finally:
            ds.close()
