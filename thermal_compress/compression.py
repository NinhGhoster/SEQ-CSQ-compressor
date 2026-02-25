"""Compression configuration for NetCDF4 output."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CompressionConfig:
    """Settings that control how temperature data is stored in NetCDF4.

    Attributes:
        complevel: zlib compression level (1–9). Higher = smaller but slower.
        shuffle: Enable byte-shuffle filter before compression (big win for
            16-bit integer data).
        chunk_frames: Number of frames per chunk along the time axis.
            Affects the trade-off between sequential playback speed and
            single-frame random-access latency.
        use_int16: Store temperatures as scaled int16 instead of float32.
            Roughly halves storage and compresses better, at the cost of
            quantisation to ``scale_factor`` precision.
        scale_factor: When ``use_int16`` is True, the value written is
            ``temperature / scale_factor``, stored as int16.  Default 0.01
            gives 0.01 °C precision.
    """

    complevel: int = 9
    shuffle: bool = True
    chunk_frames: int = 10
    use_int16: bool = False
    scale_factor: float = 0.01

    # -- helpers ----------------------------------------------------------

    @property
    def dtype(self) -> str:
        """NetCDF4 data-type string."""
        return "i2" if self.use_int16 else "f4"

    def variable_kwargs(self, height: int, width: int) -> dict:
        """Return keyword arguments for ``Dataset.createVariable()``.

        Args:
            height: Frame height in pixels.
            width: Frame width in pixels.

        Returns:
            Dict ready to be unpacked into ``createVariable()``.
        """
        kwargs: dict = {
            "datatype": self.dtype,
            "dimensions": ("frame", "y", "x"),
            "chunksizes": (self.chunk_frames, height, width),
            "zlib": True,
            "complevel": self.complevel,
            "shuffle": self.shuffle,
        }
        return kwargs

    def variable_attributes(self) -> dict:
        """Return CF-convention attributes for the temperature variable."""
        attrs: dict = {
            "units": "degC",
            "long_name": "Radiometric temperature",
        }
        if self.use_int16:
            attrs["scale_factor"] = self.scale_factor
            attrs["add_offset"] = 0.0
        return attrs


# ---- presets ------------------------------------------------------------

DEFAULT_CONFIG = CompressionConfig()
"""Default: float32, complevel 9, shuffle on."""

ARCHIVAL_CONFIG = CompressionConfig(complevel=9, use_int16=True)
"""Archival: scaled int16, complevel 9 — maximum compression."""
