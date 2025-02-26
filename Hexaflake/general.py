import numpy as np
import os
from time import time


def time_wrapper():
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time()
            result = func(*args, **kwargs)
            end = time()
            print(f"Function {func.__name__} took {end-start} seconds to run.")
            return result
        return wrapper
    return decorator


def iterative_filename(filename):
    base, extension = os.path.splitext(filename)
    counter = 0
    new_filename = f"{base}_{counter}{extension}"
    while os.path.exists(new_filename):
        counter += 1
        new_filename = f"{base}_{counter}{extension}"
    return new_filename

def test_performance(func, n_times, *args, **kwargs):
    run_times = []
    for _ in range(n_times):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        run_times.append(end-start)
    mean_time = np.mean(run_times)
    std_time = np.std(run_times)
    print(f"Mean runtime: {mean_time:.2e} seconds ({n_times} trials)")
    print(f"Standard deviation of runtime: {std_time:.2e} seconds")
    return result


def example_function(n):
    total = 0
    for i in range(n):
        total += i
    return total

# Example usage
if __name__ == "__main__":
    test_performance(example_function, 100, 1000000)