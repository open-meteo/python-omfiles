import argparse

from omfiles.types import BasicSelection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark different file formats")
    parser.add_argument("--array-size", type=tuple, default=(10, 10, 1, 10, 10, 10), help="Size of the test array")
    parser.add_argument("--chunk-size", type=tuple, default=(10, 5, 1, 1, 1, 1), help="Chunk size for writing data")
    parser.add_argument("--read-index", type=parse_tuple, default=(0, 0, 0, 0, ...), help="Index for reading")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations for each test")
    return parser.parse_args()


def parse_tuple(string: str) -> BasicSelection:
    # Remove parentheses and split by comma
    items = string.strip("()").split(",")
    # Convert each item to the appropriate type
    result = []
    for item in items:
        item = item.strip()
        if item == "...":
            result.append(Ellipsis)
        elif item.lower() == "none":
            result.append(None)
        elif item:  # Skip empty strings from trailing commas
            result.append(int(item))
    return tuple(result)
