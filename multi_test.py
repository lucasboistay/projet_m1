import random
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from time import time

def map_rw_single(args):
    n, x0 = args
    positionL = [x0]
    position = x0
    for i in range(n):
        step = random.choice([-1, 1])
        positionL.append(positionL[i] + step)
        position += step
    return positionL

def simulate_random_walks_parallel(n_iterations: int, nb_run: int):
    with Pool() as pool:
        results = pool.map(map_rw_single, [(n_iterations, 0) for _ in range(nb_run)])
    return [result[-1] for result in results]

def simulate_random_walks_serial(n_iterations: int, nb_run: int):
    results = [map_rw_single((n_iterations, 0)) for _ in range(nb_run)]
    return [result[-1] for result in results]

print("Running simulations...")

# Example usage:
n_iterations = 2000
nb_run = 20000

# Without multiprocessing
print("Serial execution...")
start_time = time()
List_df_serial = simulate_random_walks_serial(n_iterations, nb_run)
end_time = time()
print(f"Serial execution time: {end_time - start_time:.4f} seconds")

# With multiprocessing
print("Parallel execution...")
start_time = time()
List_df_parallel = simulate_random_walks_parallel(n_iterations, nb_run)
end_time = time()
print(f"Parallel execution time: {end_time - start_time:.4f} seconds")

# Plotting the histogram using serial results
plt.hist(List_df_serial, range=(-60, 60), bins=120, alpha=0.5, label='Serial')
# Plotting the histogram using parallel results
plt.hist(List_df_parallel, range=(-60, 60), bins=120, alpha=0.5, label='Parallel')
plt.xlabel('Number of occurrences')
plt.ylabel('Final Positions')
plt.title('Comparison of Serial and Parallel Execution Times')
plt.legend()
plt.show()