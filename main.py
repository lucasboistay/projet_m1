"""
This is the main file of the project. It will run the Ising model in parallel and plot the results.
@Author: Lucas BOISTAY
@Date: 2024-02-29
"""

import numpy as np

from constants import number_of_simulations, J_values, t_min, t_max, iterations
from src.graphs import (plot_different_J_graph_magnetization, plot_different_J_graph_energy,
                        plot_magnetization_and_energy, plot_critical_temperature_regression,
                        plot_magnetization_and_energy_normalized)
from src.utils import create_gif, test_different_J_values, renormalise

# Create N ising model and run the simulation to get the final energy and magnetization and plot it

final_energy = []
final_magnetization = []


if __name__ == "__main__":

    temperatures = np.linspace(t_min, t_max, number_of_simulations)

    # Run the model once for a gif of the magnetization lattice

    #create_gif(2.269, iterations)

    # Run the model in parallel

    #run_parallel_ising(number_of_simulations, number_of_pool_processes, temperatures)

    # Read the data from the file and plot it

    #plot_data_from_file('data/data.txt')

    # Run for different J values

    #test_different_J_values(J_values, temperatures)

    # Renormalise the data (to do if needed, go change the function in src/utils.py)

    #renormalise(J_values)

    plot_different_J_graph_magnetization()
    plot_different_J_graph_energy()
    plot_magnetization_and_energy()
    plot_critical_temperature_regression()
    plot_magnetization_and_energy_normalized()
