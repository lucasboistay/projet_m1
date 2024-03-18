"""
This is the main file of the project. It will run the Ising model in parallel and plot the results.
@Author: Lucas BOISTAY
@Date: 2024-02-29
"""

from src.utils import create_gif, run_parallel_ising, test_different_J_values, renormalise, get_temperature_list
from src.graphs import plot_data_from_file, plot_different_J_graph
from constants import t_min, t_max, iterations, N, M, number_of_simulations, number_of_pool_processes, J_values

import numpy as np
import matplotlib.pyplot as plt
# Create N ising model and run the simulation to get the final energy and magnetization and plot it

final_energy = []
final_magnetization = []


if __name__ == "__main__":
    # To get the temperature list
    temperatures = np.zeros(1)
    for J in J_values:
        new_temperatures = get_temperature_list(t_min, t_max, number_of_simulations, J, 0.5*J)
        temperatures = np.concatenate((temperatures, new_temperatures))
    # ajout température linéaire
    linear_temperatures = np.linspace(t_min, t_max, num=number_of_simulations-len(temperatures))
    temperatures = np.concatenate((temperatures, linear_temperatures))
    temperatures.sort()
    print(temperatures.size)

    plt.plot(temperatures, np.zeros(len(temperatures)), 'o')
    plt.show()


    #TODO: Mettre plus de points autour de la température critique pour un meilleur résultat

    # Run the model once for a gif of the magnetization lattice

    #create_gif(2.269, iterations)

    # Run the model in parallel

    #run_parallel_ising(number_of_simulations, number_of_pool_processes, temperatures)

    # Read the data from the file and plot it

    #plot_data_from_file('data/data.txt')

    # Run for different J values

    #test_different_J_values(J_values, temperatures)

    #plot_different_J_graph()
