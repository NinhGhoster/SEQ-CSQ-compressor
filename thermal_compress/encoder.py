"""Encode SEQ/CSQ radiometric video → NetCDF4.

Supports parallel compression via multiprocessing for significant speedup.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import netCDF4 as nc
import numpy as np
import zlib
import h5py
import concurrent.futures
from tqdm import tqdm

from thermal_compress.compression import CompressionConfig, DEFAULT_CONFIG
from thermal_compress.metadata import extract_metadata, open_imager

# ---------------------------------------------------------------------------
# Parallel helpers
# ---------------------------------------------------------------------------

def _shuffle_bytes(data: bytes, itemsize: int) -> bytes:
    """Apply HDF5-compatible byte shuffling to a buffer.
    
    HDF5 shuffle filter rearranges bytes to group all first bytes together,
    all second bytes together, etc., which greatly improves zlib compression.
    """
    arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, itemsize)
    return arr.T.tobytes()


def _compress_chunk_worker(
    frames: np.ndarray,
    start_idx: int,
    complevel: int,
    use_int16: bool = False,
    scale_factor: float = 0.01,
    threshold: float | None = None,
) -> tuple[int, bytes]:
    """Worker function: optionally threshold & scale, then shuffle and compress a chunk."""
    # Mask pixels below the threshold with NaN (float32) or fill value (int16)
    if threshold is not None:
        frames = frames.copy()  # don't mutate the original batch
        frames[frames < threshold] = np.nan

    if use_int16:
        # Scale and convert to int16 before compressing
        # NaN pixels become the int16 fill value (-32767)
        int_frames = np.empty(frames.shape, dtype=np.int16)
        mask = np.isnan(frames)
        int_frames[~mask] = np.round(frames[~mask] / scale_factor).astype(np.int16)
        int_frames[mask] = -32767  # CF-convention _FillValue
        frames = int_frames
    
    # itemsize is 2 for int16, 4 for float32
    itemsize = frames.dtype.itemsize 
    
    # 1. Shuffle bytes (HDF5 filter id 2)
    shuffled = _shuffle_bytes(frames.tobytes(), itemsize)
    
    # 2. Deflate/zlib compress (HDF5 filter id 1)
    compressed = zlib.compress(shuffled, complevel)
    
    return start_idx, compressed


def _read_batch(im, start: int, count: int, height: int, width: int) -> np.ndarray:
    """Read *count* frames from the imager starting at *start*.

    Returns an array shaped ``(count, height, width)`` of float32.
    """
    batch = np.empty((count, height, width), dtype=np.float32)
    for i in range(count):
        im.get_frame(start + i)
        batch[i] = np.array(im.final, copy=True).reshape((height, width))
    return batch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode(
    input_path: str | Path,
    output_path: str | Path,
    *,
    emissivity: Optional[float] = None,
    experiment: Optional[str] = None,
    config: CompressionConfig = DEFAULT_CONFIG,
    workers: int = 0,
    batch_size: int = 100,
    limit: Optional[int] = None,
    threshold: Optional[float] = None,
) -> Path:
    """Convert a single SEQ/CSQ file to compressed NetCDF4.

    Args:
        input_path: Path to the ``.seq`` or ``.csq`` source file.
        output_path: Destination ``.nc`` file path.
        emissivity: Surface emissivity to record as a global attribute.
        experiment: Free-text experiment description.
        config: Compression settings (complevel, int16, chunking, etc.).
        workers: Number of parallel workers.  0 = auto (half the CPU cores).
            1 = single-threaded without multiprocessing overhead.
        batch_size: Number of frames to read into memory at once before
            dispatching to a worker. This must be a multiple of the HDF5 
            chunk size (config.chunk_frames) for direct writes to work.
            Default: 100.
        limit: Only process the first N frames.
        threshold: If set, mask pixels below this temperature (°C) with
            NaN (float32) or _FillValue (int16). Dramatically improves
            compression when most of the frame is below the threshold.

    Returns:
        Path to the written NetCDF4 file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if workers == 0:
        workers = max(1, os.cpu_count() // 2)

    # Enforce batch_size to be a multiple of chunk_frames for direct writes
    chunk_frames = config.chunk_frames
    if batch_size % chunk_frames != 0:
        batch_size = max(chunk_frames, (batch_size // chunk_frames) * chunk_frames)

    # --- open source file ------------------------------------------------
    im = open_imager(input_path)
    height, width = im.height, im.width
    num_frames = im.num_frames
    
    if limit is not None:
        num_frames = min(num_frames, limit)

    # --- metadata --------------------------------------------------------
    meta = extract_metadata(input_path)
    if emissivity is not None:
        meta["emissivity"] = float(emissivity)
    if experiment is not None:
        meta["experiment"] = experiment
    if threshold is not None:
        meta["threshold_degC"] = float(threshold)

    # --- create NetCDF4 via h5py for direct chunk access -----------------
    # NetCDF4 is just HDF5 with specific conventions.
    
    with h5py.File(str(output_path), "w") as f:
        # NetCDF4 global attributes
        for key, value in meta.items():
            f.attrs[key] = value

        # NetCDF4 dimensions (as HDF5 datasets/attributes)
        # Note: h5py doesn't strictly create "dimensions" the same way the C netCDF
        # library does, but xarray/netCDF4-python reads h5py files fine if structured 
        # correctly. Wait, actually it's much safer to initialize the file with netCDF4 
        # to get the exact CF conventions, close it, and then reopen with h5py for 
        # direct writing! Let's do that.

    # 1. Initialize empty file structure using netCDF4 library
    with nc.Dataset(str(output_path), "w", format="NETCDF4") as ds:
        for key, value in meta.items():
            ds.setncattr(key, value)
        
        # We MUST use a fixed dimension size for direct chunk writes to work,
        # otherwise HDF5 throws "Can't write unprocessed chunk data (addr undefined)"
        # because the chunk hasn't been allocated in the file yet.
        ds.createDimension("frame", num_frames)
        ds.createDimension("y", height)
        ds.createDimension("x", width)

        var_kw = config.variable_kwargs(height, width)
        # Clamp chunking to actual size for very short test files
        chunk_frames = var_kw.get("chunksizes", (1, 1, 1))[0]
        if num_frames > 0 and chunk_frames > num_frames:
            var_kw["chunksizes"] = (num_frames, var_kw["chunksizes"][1], var_kw["chunksizes"][2])
            chunk_frames = num_frames

        temp_var = ds.createVariable("temperature", **var_kw)
        for attr_name, attr_val in config.variable_attributes().items():
            temp_var.setncattr(attr_name, attr_val)

    # 2. Re-open with h5py to write chunks directly, bypassing single-threaded compression
    try:
        f = h5py.File(str(output_path), "r+")
        temp_ds = f["temperature"]
        
        # --- write frames in parallel batches ----------------------------
        pbar = tqdm(total=num_frames, desc="Encoding", unit="frame")

        if workers > 1:
            executor = ProcessPoolExecutor(max_workers=workers)
            futures = set()
            
            idx = 0
            while idx < num_frames:
                # Read frames synchronously (SDK is not thread-safe)
                count = min(batch_size, num_frames - idx)
                batch = _read_batch(im, idx, count, height, width)
                
                # We must split the batch into exact HDF5 chunks
                for c_start in range(0, count, chunk_frames):
                    c_end = min(c_start + chunk_frames, count)
                    c_frames = batch[c_start:c_end]
                    
                    # Pad chunk if it's the last one and doesn't fit exactly
                    # (HDF5 chunks must be written at their full size)
                    if len(c_frames) < chunk_frames:
                        pad_shape = (chunk_frames - len(c_frames), height, width)
                        c_frames = np.concatenate([c_frames, np.zeros(pad_shape, dtype=c_frames.dtype)])

                    fut = executor.submit(
                        _compress_chunk_worker,
                        c_frames,
                        idx + c_start,
                        config.complevel,
                        config.use_int16,
                        config.scale_factor if config.use_int16 else 1.0,
                        threshold,
                    )
                    futures.add(fut)
                
                # Drain futures to prevent unbounded memory growth
                # (keep pool full, but don't hold thousands of arrays in queue)
                while len(futures) >= workers * 2 or (idx + count >= num_frames and futures):
                    done, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for pf in done:
                        chunk_idx, compressed_bytes = pf.result()
                        # Write directly to HDF5 bypassing the filter pipeline
                        temp_ds.id.write_direct_chunk((chunk_idx, 0, 0), compressed_bytes, filter_mask=0)
                        
                        # Update progress bar
                        frames_written = min(chunk_frames, num_frames - chunk_idx)
                        pbar.update(frames_written)
                
                idx += count
                
            executor.shutdown()
            
        else:
            # Single-threaded path
            idx = 0
            while idx < num_frames:
                count = min(batch_size, num_frames - idx)
                batch = _read_batch(im, idx, count, height, width)
                
                for c_start in range(0, count, chunk_frames):
                    c_end = min(c_start + chunk_frames, count)
                    c_frames = batch[c_start:c_end]
                    
                    if len(c_frames) < chunk_frames:
                        pad_shape = (chunk_frames - len(c_frames), height, width)
                        c_frames = np.concatenate([c_frames, np.zeros(pad_shape, dtype=c_frames.dtype)])

                    chunk_idx, compressed_bytes = _compress_chunk_worker(
                        c_frames,
                        idx + c_start,
                        config.complevel,
                        config.use_int16,
                        config.scale_factor if config.use_int16 else 1.0,
                        threshold,
                    )
                    temp_ds.id.write_direct_chunk((chunk_idx, 0, 0), compressed_bytes, filter_mask=0)
                    
                    frames_written = min(chunk_frames, num_frames - chunk_idx)
                    pbar.update(frames_written)

                idx += count

        pbar.close()
    
    finally:
        f.close()

    return output_path


def encode_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    emissivity: Optional[float] = None,
    experiment: Optional[str] = None,
    config: CompressionConfig = DEFAULT_CONFIG,
    workers: int = 0,
    limit: Optional[int] = None,
    threshold: Optional[float] = None,
) -> list[Path]:
    """Batch-convert every SEQ/CSQ file in a directory.

    Args:
        input_dir: Directory containing ``.seq`` / ``.csq`` files.
        output_dir: Directory where ``.nc`` files will be written.
        emissivity: Surface emissivity (applied to all files).
        experiment: Experiment description (applied to all files).
        config: Compression settings.
        workers: Number of parallel workers per file.
        limit: Only process the first N frames per file.
        threshold: Mask pixels below this temperature (°C).

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
            workers=workers,
            limit=limit,
            threshold=threshold,
        )
        outputs.append(out)

    return outputs
