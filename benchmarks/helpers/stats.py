import gc
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, NamedTuple, TypeVar

T = TypeVar("T")


@dataclass
class BenchmarkStats:
    mean: float
    std: float
    min: float
    max: float
    cpu_mean: float
    cpu_std: float
    memory_usage: float
    file_size: float = 0


class MeasurementResult(NamedTuple):
    result: Any
    elapsed: float
    cpu_elapsed: float
    memory_delta: float


def measure_execution(func: Callable[..., T]) -> Callable[..., MeasurementResult]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> MeasurementResult:
        # measure time
        start_time = time.time()
        cpu_start_time = time.process_time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        cpu_elapsed_time = time.process_time() - cpu_start_time

        # measure memory
        del result
        gc.collect()
        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()
        result = func(*args, **kwargs)
        end_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        memory_delta = sum(stat.size_diff for stat in end_snapshot.compare_to(start_snapshot, "lineno"))

        # fmt: off
        return MeasurementResult(
            result=result,
            elapsed=elapsed_time,
            cpu_elapsed=cpu_elapsed_time,
            memory_delta=memory_delta
        )

    return wrapper


def run_multiple_benchmarks(func: Callable[..., MeasurementResult], iterations: int = 5) -> BenchmarkStats:
    times: List[float] = []
    cpu_times: List[float] = []
    memory_usages: List[float] = []

    for _ in range(iterations):
        result = func()
        times.append(result.elapsed)
        cpu_times.append(result.cpu_elapsed)
        memory_usages.append(result.memory_delta)

    return BenchmarkStats(
        mean=statistics.mean(times),
        std=statistics.stdev(times) if len(times) > 1 else 0,
        min=min(times),
        max=max(times),
        cpu_mean=statistics.mean(cpu_times),
        cpu_std=statistics.stdev(cpu_times) if len(cpu_times) > 1 else 0,
        memory_usage=statistics.mean(memory_usages),
    )


def print_write_benchmark_results(results: Dict[str, BenchmarkStats]) -> None:
    for fmt, result in results.items():
        print(f"\n{fmt} Write Results:")
        _print_benchmark_stats(result)
        print(f"  File Size: {result.file_size} B")


def print_read_benchmark_results(results: Dict[str, BenchmarkStats]) -> None:
    for fmt, result in results.items():
        print(f"\n{fmt} Read Results:")
        _print_benchmark_stats(result)


def _print_benchmark_stats(stats: BenchmarkStats) -> None:
    print(f"  Time: {stats.mean:.6f}s ± {stats.std:.6f}s (min: {stats.min:.6f}s, max: {stats.max:.6f}s)")
    print(f"  CPU Time: {stats.cpu_mean:.6f}s ± {stats.cpu_std:.6f}s")
    print(f"  Memory Delta: {stats.memory_usage:.2f} B")
