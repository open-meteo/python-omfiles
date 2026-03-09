#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "omfiles>=1.1.1",  # x-release-please-version
#     "dask[array]>=2023.1.0",
# ]
# ///
#
# This example demonstrates writing a dask array that is larger than the
# available process memory to an OM file using streaming writes.
#
# A process memory limit is set via resource.setrlimit to simulate a
# constrained environment. The dask array is never fully materialized —
# only one chunk is held in memory at a time thanks to write_dask_array().
#
# NOTE: resource.setrlimit(RLIMIT_AS) is only enforced on Linux.
# On macOS the kernel ignores RSS/AS limits, so the script uses
# tracemalloc and ru_maxrss to prove that peak memory stays low.

import os
import platform
import resource
import tempfile
import tracemalloc

import dask.array as da
import numpy as np
from omfiles import OmFileReader, OmFileWriter
from omfiles.dask import write_dask_array

# Configuration
MEMORY_LIMIT_MB = 128  # process memory cap (enforced on Linux)
DATASET_SIZE_MB = 512  # total size of the dask array
CHUNK_SIZE = 1024  # chunk edge length (CHUNK_SIZE x CHUNK_SIZE)
DTYPE = np.float32  # 4 bytes per element

# Derived constants
bytes_per_element = np.dtype(DTYPE).itemsize
total_elements = (DATASET_SIZE_MB * 1024 * 1024) // bytes_per_element
side_length = int(np.sqrt(total_elements))  # square array for simplicity
actual_size_mb = (side_length * side_length * bytes_per_element) / (1024 * 1024)


def set_memory_limit(limit_mb: int) -> bool:
    """Try to cap the process address space. Returns True if enforced."""
    limit_bytes = limit_mb * 1024 * 1024
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))
        if platform.system() == "Linux":
            print(f"  Memory limit set to {limit_mb} MB (enforced on Linux)")
            return True
        else:
            print(
                f"  Memory limit requested ({limit_mb} MB) but {platform.system()} "
                "does not enforce RLIMIT_AS — relying on memory tracking instead"
            )
            return False
    except (ValueError, OSError, AttributeError) as e:
        print(f"  Could not set memory limit: {e}")
        return False


def get_peak_rss_mb() -> float:
    """Return peak RSS in MB (works on Linux and macOS)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == "Darwin":
        return usage.ru_maxrss / (1024 * 1024)  # macOS reports bytes
    return usage.ru_maxrss / 1024  # Linux reports kilobytes


def main():
    print("=" * 60)
    print("Dask larger-than-RAM write example")
    print("=" * 60)

    # Set memory limit
    print(f"\nSetting process memory limit to {MEMORY_LIMIT_MB} MB...")
    enforced = set_memory_limit(MEMORY_LIMIT_MB)

    # Start memory tracking
    tracemalloc.start()

    # Create a dask array larger than the memory limit
    print(
        f"\nCreating dask array: {side_length} x {side_length} {DTYPE.__name__} "
        f"({actual_size_mb:.0f} MB, chunked {CHUNK_SIZE} x {CHUNK_SIZE})"
    )

    data = da.random.random(
        (side_length, side_length),
        chunks=(CHUNK_SIZE, CHUNK_SIZE),
    ).astype(DTYPE)

    print(f"   Shape: {data.shape}")
    print(f"   Chunks: {data.chunksize}")
    print(f"   Num blocks: {data.numblocks} ({np.prod(data.numblocks)} total)")

    # Write to .om file via streaming
    fd, filepath = tempfile.mkstemp(suffix=".om")
    os.close(fd)

    print(f"\nWriting to {filepath} ...")
    writer = OmFileWriter(filepath)
    root = write_dask_array(writer, data, name="temperature")
    writer.close(root)

    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"   File size on disk: {file_size_mb:.1f} MB (compression ratio: {actual_size_mb / file_size_mb:.1f}x)")

    # Read back a slice and verify
    print("\nReading back a slice to verify...")
    with OmFileReader(filepath) as reader:
        print(f"   Reader shape: {reader.shape}, dtype: {reader.dtype}")
        sample = reader[0:10, 0:10]
        print(f"   Sample slice [0:10, 0:10] shape: {sample.shape}")
        print(f"   Sample values (first row): {sample[0, :5]}")
        assert sample.shape == (10, 10), "Unexpected slice shape"
        assert not np.any(np.isnan(sample)), "Found NaN values in readback"

    print("   Verification passed!")

    # Memory summary
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_traced_mb = peak / (1024 * 1024)
    peak_rss_mb = get_peak_rss_mb()

    print("\n" + "=" * 60)
    print("Memory summary")
    print("=" * 60)
    print(f"  Dataset size:          {actual_size_mb:.0f} MB")
    if enforced:
        print(f"  Process memory limit:  {MEMORY_LIMIT_MB} MB (enforced)")
    else:
        print(f"  Process memory limit:  {MEMORY_LIMIT_MB} MB (not enforced on {platform.system()})")
    print(f"  Peak traced (Python):  {peak_traced_mb:.1f} MB")
    print(f"  Peak RSS (process):    {peak_rss_mb:.1f} MB")
    print(f"  Ratio (dataset/peak):  {actual_size_mb / peak_rss_mb:.1f}x")
    print()

    if peak_rss_mb < actual_size_mb:
        print("The entire dataset was written WITHOUT loading it all into memory.")
    else:
        print("WARNING: Peak RSS exceeded dataset size — streaming may not have worked as expected.")

    # Cleanup
    os.unlink(filepath)


if __name__ == "__main__":
    main()
