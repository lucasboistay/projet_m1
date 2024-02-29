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

N = 100
M = 100
iterations = 1000000
t_min = 0.1
t_max = 4


def Onsager(Tc, T):
    """
    Onsager solution for the magnetization
    :param Tc: Critical temperature
    :param T: Temperature array
    :return: Onsager solution
    """
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
    plt.plot(temperature, magnetization, label="Magnetization")
    plt.plot(temperature, smooth_magnetization, label="Smoothed magnetization")
    critical_temperature = temperature[max_derivative_index]
    # plot the critical temperature line
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


def create_gif(temperature, iterations):
    """
    Create a gif of the lattice magnetization for a given temperature
    :param temperature: temperature
    :param iterations: number of iterations
    :return: None
    """
    print(f"------ Creating gif for T={temperature}... ------")
    ising = IsingModel(N, M, temperature, iterations)
    ising.initialize_lattice(-1)  # To get a lattice with all 1's
    ising.run_monte_carlo(save_image=True)
    print("Gif created and saved as ising.gif")


def run_parallel_ising(N_simulation, N_pool_processes, temperatures):
    """
    Run the Ising model in parallel
    :param N_simulation: Number of simulations
    :param N_pool_processes: Number of pool processes (number of cores of your CPU)
    :param temperatures: Temperatures array
    :return: None
    """

    print("------ Running the model in parallel... ------")
    print(f"Number of simulations: {N_simulation}")

    with Pool(number_of_pool_processes) as p:  # Run the model in parallel with a pool of processes
        # v Run the model for each temperature v
        print("Pool of processes created")
        start_time = time.time()
        results = p.starmap(run_model, [(N, M, temperature, iterations) for temperature in temperatures])
        final_energy, final_magnetization = zip(*results)  # Unzip the results
        end_time = time.time()
        p.close()  # Close the pool
        print("Pool of processes closed")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        p.join()  # Join the pool

    # Normalisation
    final_energy = np.array(final_energy) / (N * M)  # Normalise the final energy
    final_magnetization = np.array(final_magnetization) / (N * M)  # Normalise the final magnetization

    # Save the data in a txt file as a table

    data = pd.DataFrame({'Temperature': temperatures, 'Energy': final_energy, 'Magnetization': final_magnetization})
    data.to_csv('data/data.txt', index=False, sep='\t')

    print("Data saved in data/data.txt")


def read_data_from_file(filename):
    """
    Read the data from the file and plot it
    :param filename: Filename
    :return: None
    """
    data = pd.read_csv('data/data.txt', sep='\t')
    temperatures = data['Temperature']
    final_energy = data['Energy']
    final_magnetization = data['Magnetization']

    # Find the critical temperature

    critical_temperature = find_critical_temperature(temperatures, final_magnetization)

    # Plot the final energy and magnetization

    onsager = [Onsager(2.269, T) for T in temperatures]

    plt.figure(figsize=(14, 10))
    plt.plot(temperatures, final_magnetization, 'bo', label=rf"Simulation data (Monte Carlo)")
    plt.plot(temperatures, onsager, 'g--', label=rf"Onsager solution (Analytical)")
    # plot the critical temperature line
    plt.axvline(x=2.269, color='green', linestyle='-', label=r"Analitical $T_c = 2.269$")
    plt.axvline(x=critical_temperature, color='blue', linestyle='-',
                label=rf"Computated $T_c = {critical_temperature:.2f}$")
    plt.xlabel(rf"Temperature /J")
    plt.ylabel(rf"Magnetisation /µ")
    plt.legend()
    plt.axhline(c="k", linewidth=1)
    plt.axvline(c="k", linewidth=1)
    plt.title(f'Magnetisation vs Temperature\n(Lattice : {N}x{M}, {iterations} iterations)')
    plt.savefig(f"data/magnetization.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # Parameters
    number_of_simulations = 100
    number_of_pool_processes = 10  # Number of pool processes, do not set it to more than the number of cores of your
    # CPU
    temperatures = np.linspace(t_min, t_max, number_of_simulations)

    # Run the model once for a gif of the magnetization lattice

    create_gif(2.269, iterations)

    # Run the model in parallel

    # run_parallel_ising(number_of_simulations, number_of_pool_processes, temperatures)

    # Read the data from the file and plot it

    # read_data_from_file('data/data.txt')
