"""CLI entry point for thermal-compress."""

from __future__ import annotations

from pathlib import Path

import click

from thermal_compress.compression import CompressionConfig


@click.group()
@click.version_option(package_name="thermal-compress")
def cli():
    """thermal-compress — Lossless compression of FLIR SEQ/CSQ to NetCDF4."""


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", required=True, type=click.Path(),
              help="Output .nc file (or directory in batch mode).")
@click.option("--emissivity", type=float, default=None,
              help="Surface emissivity (e.g. 0.95).")
@click.option("--experiment", type=str, default=None,
              help="Free-text experiment description.")
@click.option("--complevel", type=click.IntRange(1, 9), default=9,
              help="zlib compression level (1–9). Default: 9.")
@click.option("--int16", "use_int16", is_flag=True, default=False,
              help="Store as scaled int16 for ~2× extra compression.")
@click.option("--batch", is_flag=True, default=False,
              help="Treat INPUT_PATH as a directory and convert all SEQ/CSQ files.")
@click.option("--workers", type=int, default=0,
              help="Parallel workers (0 = auto, all CPU cores). Default: 0.")
@click.option("--batch-size", type=int, default=100,
              help="Frames per read batch. Larger = faster but more RAM. Default: 100.")
@click.option("--limit", type=int, default=None,
              help="Only encode the first N frames (useful for benchmarking).")
@click.option("--threshold", type=float, default=None,
              help="Mask pixels below this temperature (°C) to dramatically reduce file size.")
def encode(input_path, output_path, emissivity, experiment, complevel, use_int16,
           batch, workers, batch_size, limit, threshold):
    """Convert SEQ/CSQ file(s) to compressed NetCDF4."""
    from thermal_compress.encoder import encode as _encode, encode_batch

    config = CompressionConfig(complevel=complevel, use_int16=use_int16)

    if batch:
        outputs = encode_batch(
            input_path, output_path,
            emissivity=emissivity,
            experiment=experiment,
            config=config,
            workers=workers,
            limit=limit,
            threshold=threshold,
        )
        click.echo(f"✓ Converted {len(outputs)} file(s):")
        for p in outputs:
            click.echo(f"  {p}")
    else:
        out = _encode(
            input_path, output_path,
            emissivity=emissivity,
            experiment=experiment,
            config=config,
            workers=workers,
            batch_size=batch_size,
            limit=limit,
            threshold=threshold,
        )
        click.echo(f"✓ Written: {out}")


@cli.command()
@click.argument("nc_path", type=click.Path(exists=True))
@click.option("-o", "--output", "output_dir", required=True, type=click.Path(),
              help="Output directory for extracted frames.")
@click.option("--format", "fmt", type=click.Choice(["npy", "csv"]), default="npy",
              help="Output format (default: npy).")
def decode(nc_path, output_dir, fmt):
    """Extract frames from a NetCDF4 file."""
    from thermal_compress.decoder import decode as _decode

    outputs = _decode(nc_path, output_dir, fmt=fmt)
    click.echo(f"✓ Extracted {len(outputs)} frames to {output_dir}/")


@cli.command()
@click.argument("seq_path", type=click.Path(exists=True))
@click.argument("nc_path", type=click.Path(exists=True))
@click.option("--int16", "use_int16", is_flag=True, default=False,
              help="File was encoded with --int16; use wider tolerance.")
def verify(seq_path, nc_path, use_int16):
    """Verify lossless round-trip between SEQ/CSQ and NetCDF4."""
    from thermal_compress.verify import verify as _verify

    atol = 0.005 if use_int16 else 1e-6
    _verify(seq_path, nc_path, atol=atol, use_int16=use_int16)
    click.echo("✓ Verification passed.")


@cli.command()
@click.argument("nc_path", type=click.Path(exists=True))
def info(nc_path):
    """Display metadata and shape info from a NetCDF4 file."""
    from thermal_compress.decoder import info as _info

    _info(nc_path)


if __name__ == "__main__":
    cli()
