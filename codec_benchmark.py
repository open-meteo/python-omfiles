import cProfile
import os
import pstats
import shutil
import time

import numpy as np
from numcodecs.zarr3 import Blosc, Delta, FixedScaleOffset, PCodec, Quantize
from omfiles.omfiles_numcodecs import PyPforDelta2dCodec, PyPforDelta2dSerializer
from tabulate import tabulate
from zarr import create_array
from zarr.storage import LocalStore

delta_config = {
    'float32': '<f4',
    'float64': '<f8',
    'int8': '<i1',
    'uint8': '<u1',
    'int16': '<i2',
    'uint16': '<u2',
    'int32': '<i4',
    'uint32': '<u4',
    'int64': '<i8',
    'uint64': '<u8'
}

# Test data generators
def generate_int_data(shape, dtype, pattern='sequential'):
    if pattern == 'sequential':
        base = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    elif pattern == 'random':
        base = np.random.randint(0, 1000, size=shape, dtype=dtype)
    elif pattern == 'incremental':
        # Data with small differences between adjacent values
        base = np.zeros(shape, dtype=dtype)
        for i in range(shape[0]):
            base[i] = np.arange(shape[1], dtype=dtype) + i
    else:
        raise ValueError(f"pattern {pattern} not defined")
    return base

def generate_float_data(shape, dtype, pattern='sequential'):
    if pattern == 'sequential':
        base = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        base = base / 10.0  # Convert to float with decimals
    elif pattern == 'random':
        base = np.random.random(shape).astype(dtype) * 1000
    elif pattern == 'incremental':
        # Data with small differences between adjacent values
        base = np.zeros(shape, dtype=dtype)
        for i in range(shape[0]):
            base[i] = np.arange(shape[1], dtype=dtype) / 10.0 + i/10.0
    else:
        raise ValueError(f"pattern {pattern} not defined")
    return base

# Codec configurations
def get_codecs(dtype_name):

    codecs = {
        'none': None,
        'pfordelta': PyPforDelta2dCodec(dtype=dtype_name),
        'pfordelta_serializer': PyPforDelta2dSerializer(dtype=dtype_name),
        'pcodec': PCodec(level = 8, mode_spec="auto"),
        'blosc': Blosc(cname='zstd', clevel=5),
        'blosc_lz4': Blosc(cname='lz4', clevel=5),
    }

    return codecs

def benchmark_codec(dtype, data_pattern, data_size, tmp_dir):
    """Benchmark a codec with specific data parameters."""
    dtype_name = dtype.__name__
    is_float = 'float' in dtype_name

    # Generate appropriate test data
    if is_float:
        data = generate_float_data(data_size, dtype, data_pattern)
    else:
        data = generate_int_data(data_size, dtype, data_pattern)

    # Get codec configurations for this dtype
    codecs = get_codecs(dtype_name)

    # Set up filters based on data type
    if is_float:
        filters = [Delta(dtype=delta_config[dtype_name]), Quantize(digits=2, dtype=delta_config[dtype_name])]
    else:
        filters = [Delta(dtype=delta_config[dtype_name])]

    filters = []
    results = []

    for codec_name, codec in codecs.items():
        is_serializer = codec_name in ["pcodec", "pfordelta_serializer"]

        chunk_shape = (min(data_size[0], 1000), min(data_size[1], 100))
        # chunk_shape = (data_size[0], data_size[1])
        print(f"Testing codec: {codec_name} with data shape {data.shape} and chunk shape {chunk_shape}")
        # Path for this specific codec test
        path = os.path.join(tmp_dir, f"{dtype_name}_{codec_name}_{data_pattern}")
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        store = LocalStore(path)

        # Time for array creation + data write
        start_time = time.time()

        if not is_serializer:
            z = create_array(
                store,
                shape=data.shape,
                chunks=chunk_shape,
                dtype=data.dtype,
                fill_value=0,
                filters=[],
                compressors=codec,
                serializer="auto"
            )
        else :
            filters = [
                FixedScaleOffset(offset=0, scale=1, dtype=data.dtype, astype=data.dtype),
                Delta(dtype=data.dtype, astype=data.dtype)
            ]
            z = create_array(
                store,
                shape=data.shape,
                chunks=chunk_shape,
                dtype=data.dtype,
                fill_value=0,
                filters=filters,
                serializer=codec
            )

        z[:] = data
        write_time = time.time() - start_time

        # Time for data read
        start_time = time.time()
        read_data = z[:]
        read_time = time.time() - start_time

        # Verify data
        if is_float:
            np.testing.assert_array_almost_equal(read_data, data, decimal=1)
        else:
            np.testing.assert_array_equal(read_data, data)

        # Get compression stats
        original_size = data.nbytes
        compressed_size = z.nbytes_stored()
        compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')

        result = {
            'codec': codec_name,
            'dtype': dtype_name,
            'pattern': data_pattern,
            'data_shape': data.shape,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'write_time': write_time,
            'read_time': read_time
        }

        results.append(result)

    return results

def main():
    tmp_dir = "zarr_codec_benchmark"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    dtypes = [np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]
    # dtypes = [np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64]
    patterns = ['sequential', 'incremental', 'random']
    sizes = [
        # (3, 10),
        (100, 1000),
        (1000, 100),
        (10000, 50)
    ]

    all_results = []

    # Run all benchmarks
    for dtype in dtypes:
        for pattern in patterns:
            for size in sizes:
                print(f"Benchmarking {dtype.__name__} - {pattern} - {size}...")
                results = benchmark_codec(dtype, pattern, size, tmp_dir)
                all_results.extend(results)

                # Print results for this configuration
                size_str = f"{size[0]}x{size[1]}"
                print(f"\n{dtype.__name__} - {pattern} - {size_str}:")
                rows = []
                for r in sorted(results, key=lambda x: x['codec']):
                    rows.append([
                        r['codec'],
                        f"{r['compression_ratio']:.2f}x",
                        f"{r['write_time']:.4f}s",
                        f"{r['read_time']:.4f}s"
                    ])
                print(tabulate(rows, headers=["Codec", "Comp Ratio", "Write Time", "Read Time"], tablefmt="grid"))
                print()

    # Generate summary report
    print("\n===== BENCHMARK SUMMARY =====")

    # Group by dtype and pattern
    for dtype in dtypes:
        dtype_name = dtype.__name__
        print(f"\n== {dtype_name} Summary ==")

        for pattern in patterns:
            filtered_results = [r for r in all_results if r['dtype'] == dtype_name and r['pattern'] == pattern]
            if filtered_results:
                print(f"\n{pattern.capitalize()} data:")
                rows = []
                for r in sorted(filtered_results, key=lambda x: (x['data_shape'], x['codec'])):
                    size_str = f"{r['data_shape'][0]}x{r['data_shape'][1]}"
                    rows.append([
                        r['codec'],
                        size_str,
                        f"{r['compression_ratio']:.2f}x",
                        f"{r['compressed_size']:.2f} bytes",
                        f"{r['write_time']:.4f}s",
                        f"{r['read_time']:.4f}s"
                    ])
                print(tabulate(rows,
                      headers=["Codec", "Data Size", "Comp Ratio", "Comp Size", "Write Time", "Read Time"],
                      tablefmt="grid"))

    # Cleanup
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

def main_with_profiling():
    profiler = cProfile.Profile()
    profiler.enable()

    # Your existing code here
    main()

    profiler.disable()

    # Print sorted stats
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(50)  # Print top 30 time-consuming functions

    # Optionally save to file for more detailed analysis
    stats.dump_stats('benchmark_profile.prof')

if __name__ == "__main__":
    # main()
    main_with_profiling()
