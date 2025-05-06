from argparse import Namespace

from helpers.args import parse_args
from helpers.formats import FormatFactory
from helpers.generate_data import generate_test_data
from helpers.prints import print_data_info
from helpers.stats import (
    measure_execution,
    print_read_benchmark_results,
    print_write_benchmark_results,
    run_multiple_benchmarks,
)
from zarr.core.buffer import NDArrayLike

# Define separate dictionaries for read and write formats and filenames
write_formats_and_filenames = {
    "h5": "benchmark_files/data.h5",
    "zarr": "benchmark_files/data.zarr",
    "nc": "benchmark_files/data.nc",
    "om": "benchmark_files/data.om",
}

read_formats_and_filenames = {
    "h5": "benchmark_files/data.h5",
    "h5hidefix": "benchmark_files/data.h5",
    "zarr": "benchmark_files/data.zarr",
    "zarrTensorStore": "benchmark_files/data.zarr",
    "nc": "benchmark_files/data.nc",
    "om": "benchmark_files/data.om",
}


def bm_write_all_formats(args: Namespace, data: NDArrayLike):
    write_results = {}
    for format_name, file in write_formats_and_filenames.items():
        writer = FormatFactory.create_writer(format_name, file)

        @measure_execution
        def write():
            writer.write(data, args.chunk_size)

        try:
            write_stats = run_multiple_benchmarks(write, args.iterations)
            write_stats.file_size = writer.get_file_size()
            write_results[format_name] = write_stats
        except Exception as e:
            print(f"Error with {format_name}: {e}")

    print_write_benchmark_results(write_results)


def bm_read_all_formats(args: Namespace):
    read_results = {}
    for format_name, file in read_formats_and_filenames.items():
        reader = FormatFactory.create_reader(format_name, file)

        @measure_execution
        def read():
            return reader.read(args.read_index)

        try:
            # sample_data = reader.read(args.read_index)  # Get sample data for verification
            read_stats = run_multiple_benchmarks(read, args.iterations)
            read_results[format_name] = read_stats

        except Exception as e:
            print(f"Error with {format_name}: {e}")
        finally:
            reader.close()

    print_read_benchmark_results(read_results)


def main():
    # Defines chunk and array sizes
    args = parse_args()

    data = generate_test_data(args.array_size, noise_level=5, amplitude=20, offset=20)
    print_data_info(data, args.chunk_size)
    bm_write_all_formats(args, data)

    bm_read_all_formats(args)


if __name__ == "__main__":
    main()
