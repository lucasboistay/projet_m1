"""
This file contains the different function to run the Ising model in parallel or even to create a gif of the lattice
magnetization. It also contains the function to find the critical temperature and the Onsager solution for the
magnetization.

@Author: Lucas BOISTAY
@Date: 2024-02-28
"""

from src.isingModel import IsingModel
from constants import N, M, iterations, t_min, t_max, number_of_pool_processes, number_of_simulations, J_values

import numpy as np
import pandas as pd
from multiprocessing import Pool
import time
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

n_parrallel_done = 0

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def find_critical_temperature(temperature: np.ndarray, magnetization: np.ndarray) -> (float, np.ndarray, np.ndarray):
    """
    Find the critical temperature
    :param temperature: (np.ndarray) Temperature array
    :param magnetization: (np.ndarray) Magnetization array
    :return: (float) Critical temperature
    """
    # Find the critical temperature by finding the temperature where the magnetization is the most changing by derivate
    smooth_magnetization = savgol_filter(magnetization, 40, 3)  # Smooth the magnetization
    derivative = abs(np.gradient(smooth_magnetization, temperature))
    max_derivative_index = np.argmax(derivative)
    critical_temperature = temperature[max_derivative_index]

    return critical_temperature, smooth_magnetization, derivative


def Onsager(Tc: float, T: float) -> float:
    """
    Onsager solution for the magnetization
    :param Tc: (float) Critical temperature
    :param T: (np.ndarray) Temperature array
    :return: (np.ndarray) Onsager solution
    """
    if T < Tc:
        return (1 - 1 / (np.sinh(2 / T) ** 4)) ** (1 / 8)
    else:
        return 0

def get_temperature_list(t_min: float, t_max: float, number_of_simulations: int, J: float, sigma: float) \
        -> np.ndarray:
    """
    Get the temperature list (Gaussian distribution around critical temperature)
    :param t_min: (float) Minimum temperature
    :param t_max: (float) Maximum temperature
    :param J: (float) Interaction constant
    :return: (np.ndarray) Temperature list
    """
    # Get the temperature list
    critical_temp = 2.269*J
    number_of_points = 100
    gauss = np.ones(number_of_points)
    i=0

    while i < number_of_points:
        new_value = np.random.normal(critical_temp, sigma, 1)
        if t_min < new_value < t_max:
            gauss[i] = new_value
            i+=1

    return gauss


def run_model(N: int, M: int, temperature: float, iterations: int, J: float) -> (float, float, float, float):
    """
    Run one Ising model
    :param N: (int) Number of rows
    :param M: (int) Number of columns
    :param temperature: (int) temperature
    :param iterations: (int) number of iterations
    :param J: (float) Interaction constant
    :return: (float, float) final energy and magnetization
    """
    global n_parrallel_done

    # Create an Ising model
    ising = IsingModel(N, M, temperature, iterations, J)

    ising.initialize_lattice(1)  # To get a lattice with all 1's

    energy, magnetisation, specific_heat, susceptibility = ising.run_monte_carlo()
    n_parrallel_done += 1
    printProgressBar(n_parrallel_done*3, number_of_simulations, prefix = 'Progress:', suffix = 'Complete', length = 50)
    return energy, magnetisation, specific_heat, susceptibility


def create_gif(temperature: float, iterations: int) -> None:
    """
    Create a gif of the lattice magnetization for a given temperature
    :param temperature: (float) temperature for the ising model
    :param iterations: (int) number of iterations for the ising model
    :return: None
    """
    print(f"------ Creating gif for T={temperature}... ------")
    ising = IsingModel(N, M, temperature, iterations)
    ising.initialize_lattice("random")  # To get a lattice with all 1's
    ising.run_monte_carlo_gif()
    print("Gif created and saved as ising.gif")


def run_parallel_ising(N_simulation: int, N_pool_processes: int, temperatures: np.ndarray, J: float = 1) -> None:
    """
    Run the Ising model in parallel
    :param N_simulation: (int) Number of simulations
    :param N_pool_processes: (int) Number of pool processes (number of cores of your CPU)
    :param temperatures: (np.ndarray) Temperatures array
    :param J: (float) Interaction constant
    :return: None
    """

    print("------ Running the model in parallel... ------")
    print(f"Number of simulations: {N_simulation}")
    # Initial call to print 0% progress
    with Pool(N_pool_processes) as p:  # Run the model in parallel with a pool of processes, if not understood, it's ok
        # v Run the model for each temperature v
        print("Pool of processes created")
        start_time = time.time()
        results = p.starmap(run_model, [(N, M, temperature, iterations, J) for temperature in temperatures])
        final_energy, final_magnetization, specific_heat, susceptibility = zip(*results)  # Unzip the results
        end_time = time.time()
        p.close()  # Close the pool
        print("Pool of processes closed")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        p.join()  # Join the pool

    # Normalisation
    final_energy = np.array(final_energy) / (N*M*J)  # Normalise the final energy
    final_magnetization = np.array(final_magnetization) / (N*M)  # Normalise the final magnetization
    specific_heat = np.array(specific_heat) / (N*M*J)  # Normalise the specific heat
    susceptibility = np.array(susceptibility) / (N*M)  # Normalise the susceptibility

    # Save the data in a txt file as a table

    data = pd.DataFrame({'Temperature': temperatures, 'Energy': final_energy, 'Magnetization': final_magnetization,
                         'Specific heat': specific_heat, 'Susceptibility': susceptibility})
    nom_fichier = f'data/iter_10e{int(np.log10(iterations))}_J_{J}.txt'
    data.to_csv(nom_fichier, index=False, sep='\t')

    print(f"Data saved in {nom_fichier}")


def test_different_J_values(J_values: list[float], temperatures: np.ndarray) -> None:
    """
    Test the Ising model with different J values.
    :param J_values: (array-like) J values
    :param temperatures: (np.ndarray) Temperatures array
    :return: None
    """
    global n_parrallel_done
    print("------ Testing different J values... ------")
    # Different J values

    for J in J_values:
        print(f"J = {J}\n")
        # Run the model in parallel
        run_parallel_ising(number_of_simulations, number_of_pool_processes, temperatures, J)
        n_parrallel_done = 0
    print("------ Testing different J values done ------\n")

def renormalise(J_tab: list[float]):
    """
    Renormalise the data for every file in data/
    :param J_tab:
    :return:
    """
    # get all .txt files in data/
    import os
    data_files = os.listdir('data/')
    data_files = [file for file in data_files if file.endswith('.txt')]
    for (J,file) in zip(J_tab,data_files):
        data = pd.read_csv('data/' + file, sep='\t')
        data[f'Temperature'] = data['Temperature'] * J
        data.to_csv('data/' + file, index=False, sep='\t')
        print(f"Data renormalised for {file}")
