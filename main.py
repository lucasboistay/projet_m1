## main.py

from isingModel import IsingModel
import matplotlib.pyplot as plt
import time
import numpy as np
from multiprocessing import Pool
from scipy.signal import savgol_filter
import pandas as pd

# Create N ising model and run the simulation to get the final energy and magnetization and plot it

final_energy = []
final_magnetization = []


def Onsager(Tc, T):
    if T < Tc:
        return (1 - 1 / (np.sinh(2 / T) ** 4)) ** (1 / 8)
    else:
        return 0


def run_model(N, M, temperature, iterations):
    """
    Run one Ising model
    :param N: Number of rows
    :param M: Number of columns
    :param temperature: temperature
    :param iterations: number of iterations
    :return: final energy and magnetization
    """
    # Create an Ising model
    ising = IsingModel(N, M, temperature, iterations)

    ising.initialize_lattice(1)  # To get a lattice with all 1's

    # information about the lattice
    starting_lattice = np.copy(ising.get_lattice())

    ising.run_monte_carlo()

    ending_lattice = np.copy(ising.get_lattice())

    return ising.get_total_energy(), ising.magnetization()

def find_critical_temperature(temperature, magnetization):
    """
    Find the critical temperature
    :param temperature: temperature
    :param magnetization: magnetization
    :return: critical temperature
    """
    # Find the critical temperature by finding the temperature where the magnetization is the most changing by derivate
    smooth_magnetization = savgol_filter(magnetization, 20, 3)  # Smooth the magnetization
    derivative = abs(np.gradient(smooth_magnetization, temperature))
    max_derivative_index = np.argmax(derivative)

    plt.figure(figsize=(14, 10))

    plt.plot(temperature, derivative, label="Derivative of magnetization")
    plt.plot(temperature,magnetization, label="Magnetization")
    plt.plot(temperature,smooth_magnetization, label="Smoothed magnetization")
    critical_temperature = temperature[max_derivative_index]
    #plot the critical temperature line
    plt.axvline(x=critical_temperature, color='red', linestyle='--',
                label=r'Computated $T_c$')
    plt.legend()
    plt.xlabel(rf"Temperature /J")
    plt.ylabel(rf"Magnetisation /µ")
    plt.title(f'Critical temperature computation\nCritical temperature = {critical_temperature:.2f}')
    plt.savefig(f"data/critical_temperature.png", dpi=300)
    plt.show()
    plt.close()
    return critical_temperature


# Run the model in parallel

if __name__ == "__main__":
    # Parameters
    number_of_simulations = 100
    number_of_pool_processes = 10 # Number of pool processes, do not set it to more than the number of cores of your CPU
    N = 100
    M = 100
    iterations = 500000
    temperatures = np.linspace(0.1, 4, number_of_simulations)

    # Run the model once for a plot of the magnetization lattice
    # TODO

    ising = IsingModel(N, M, 1, iterations)
    ising.initialize_lattice(-1)  # To get a lattice with all 1's
    ising.run_monte_carlo()
    plt.figure(figsize=(14, 10))
    plt.imshow(ising.get_lattice(), cmap='gray')
    plt.title(f'Ising model lattice\n(Lattice : {N}x{M}, T = 2.269, {iterations} iterations)')
    plt.savefig(f"data/lattice.png", dpi=300)
    plt.show()
    plt.close()

    # Run the model in parallel

    with Pool(number_of_pool_processes) as p:  # Run the model in parallel with a pool of processes
        # v Run the model for each temperature v
        results = p.starmap(run_model, [(N, M, temperature, iterations) for temperature in temperatures])
        final_energy, final_magnetization = zip(*results)  # Unzip the results
        p.close()  # Close the pool
        p.join()  # Join the pool

    # Normalisation
    final_energy = np.array(final_energy) / (N * M)  # Normalise the final energy
    final_magnetization = np.array(final_magnetization) / (N * M)  # Normalise the final magnetization

    # Save the data in a txt file as a table

    data = pd.DataFrame({'Temperature': temperatures, 'Energy': final_energy, 'Magnetization': final_magnetization})
    data.to_csv('data/data.txt', index=False, sep='\t')

    # Find the critical temperature

    critical_temperature = find_critical_temperature(temperatures, final_magnetization)

    # Plot the final energy and magnetization

    onsager = [Onsager(2.269, T) for T in temperatures]

    plt.figure(figsize=(14, 10))
    plt.plot(temperatures, final_magnetization,'bo', label=rf"Simulation data (Monte Carlo)")
    plt.plot(temperatures, onsager, 'g--',label=rf"Onsager solution (Analytical)")
    # plot the critical temperature line
    plt.axvline(x=2.269, color='green', linestyle='-', label=r"Analitical $T_c = 2.269$")
    plt.axvline(x=critical_temperature, color='blue', linestyle='-', label=rf"Computated $T_c = {critical_temperature:.2f}$")
    plt.xlabel(rf"Temperature /J")
    plt.ylabel(rf"Magnetisation /µ")
    plt.legend()
    plt.axhline(c="k", linewidth=1)
    plt.axvline(c="k", linewidth=1)
    plt.title(f'Magnetisation vs Temperature\n(Lattice : {N}x{M}, {iterations} iterations)')
    plt.savefig(f"data/magnetization.png", dpi=300)
    plt.show()
