"""thermal-compress — Lossless compression of FLIR SEQ/CSQ to NetCDF4."""

__version__ = "1.0.0"

from thermal_compress.compression import CompressionConfig, DEFAULT_CONFIG, ARCHIVAL_CONFIG
from thermal_compress.encoder import encode
from thermal_compress.decoder import decode, info
from thermal_compress.verify import verify
