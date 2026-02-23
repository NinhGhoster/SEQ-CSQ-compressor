"""Round-trip tests: synthetic encode → decode, verify data integrity."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import netCDF4 as nc
import numpy as np
import pytest

from thermal_compress.compression import CompressionConfig, DEFAULT_CONFIG
from thermal_compress.encoder import encode
from thermal_compress.decoder import decode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_imager(frames: np.ndarray):
    """Build a mock ``fnv.file.ImagerFile``."""
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

class TestRoundTrip:
    """Encode then decode and compare."""

    def test_float32_roundtrip(self, tmp_path: Path):
        """Float32 encode → decode should preserve data within float32 precision."""
        original = np.random.rand(10, 16, 20).astype(np.float32) * 300 - 50
        mock_im = _make_mock_imager(original)

        fake_seq = tmp_path / "input.seq"
        fake_seq.touch()
        nc_file = tmp_path / "output.nc"
        frames_dir = tmp_path / "frames"

        with (
            patch("thermal_compress.encoder.open_imager", return_value=mock_im),
            patch("thermal_compress.encoder.extract_metadata", return_value={}),
        ):
            encode(fake_seq, nc_file, config=DEFAULT_CONFIG)

        decoded_files = decode(nc_file, frames_dir, fmt="npy")

        assert len(decoded_files) == 10
        for i, p in enumerate(decoded_files):
            restored = np.load(str(p))
            np.testing.assert_allclose(restored, original[i], atol=1e-5)

    def test_int16_roundtrip(self, tmp_path: Path):
        """int16 encode → decode should be within scale_factor/2 precision."""
        original = np.array([
            [[20.12, 30.45], [15.78, 40.01]],
            [[25.99, 10.01], [35.50, 22.22]],
        ], dtype=np.float32)
        mock_im = _make_mock_imager(original)

        fake_seq = tmp_path / "input.seq"
        fake_seq.touch()
        nc_file = tmp_path / "output.nc"
        frames_dir = tmp_path / "frames"

        config = CompressionConfig(use_int16=True, scale_factor=0.01)

        with (
            patch("thermal_compress.encoder.open_imager", return_value=mock_im),
            patch("thermal_compress.encoder.extract_metadata", return_value={}),
        ):
            encode(fake_seq, nc_file, config=config)

        decoded_files = decode(nc_file, frames_dir, fmt="npy")

        for i, p in enumerate(decoded_files):
            restored = np.load(str(p))
            # netCDF4 auto-applies scale_factor on read, so restored values
            # are already in °C.  Compare against the quantised original.
            expected = np.round(original[i].astype(np.float64) / config.scale_factor) * config.scale_factor
            np.testing.assert_allclose(
                restored, expected,
                atol=config.scale_factor / 2 + 1e-6,
            )

    def test_large_frame_roundtrip(self, tmp_path: Path):
        """Roundtrip with a realistic 512×640 frame."""
        original = np.random.rand(2, 512, 640).astype(np.float32) * 500
        mock_im = _make_mock_imager(original)

        fake_seq = tmp_path / "input.seq"
        fake_seq.touch()
        nc_file = tmp_path / "output.nc"
        frames_dir = tmp_path / "frames"

        with (
            patch("thermal_compress.encoder.open_imager", return_value=mock_im),
            patch("thermal_compress.encoder.extract_metadata", return_value={}),
        ):
            encode(fake_seq, nc_file)

        decoded_files = decode(nc_file, frames_dir)
        for i, p in enumerate(decoded_files):
            restored = np.load(str(p))
            np.testing.assert_allclose(restored, original[i], atol=1e-5)
