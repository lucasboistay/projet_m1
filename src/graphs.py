"""
To plot the different graphs of the Ising model

@Author: Lucas BOISTAY
@Date: 2024-02-29
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from src.utils import find_critical_temperature
from src.utils import Onsager
from constants import N, M, iterations, t_min, t_max


def plot_critical_temperature(filename: str, temperature: np.ndarray, magnetization: np.ndarray,
                              smoothed_magnetization: np.ndarray, derivative_magnetization: np.ndarray,
                              critical_temperature: float) -> None:
    """
    Plot the critical temperature graph
    :param filename: (str) Filename
    :param temperature: (np.ndarray) Temperature
    :param magnetization: (np.ndarray) Magnetization
    :param smoothed_magnetization: (np.ndarray) Smoothed magnetization
    :param derivative_magnetization: (np.ndarray) Derivative of magnetization
    :param critical_temperature: (float) Critical temperature
    :return: None
    """
    plt.figure(figsize=(14, 10))

    plt.plot(temperature, derivative_magnetization, label="Derivative of magnetization")
    plt.plot(temperature, magnetization, label="Magnetization")
    plt.plot(temperature, smoothed_magnetization, label="Smoothed magnetization")
    # plot the critical temperature line
    plt.axvline(x=critical_temperature, color='red', linestyle='--',
                label=r'Computated $T_c$')
    plt.legend()
    plt.xlabel(rf"Temperature /J")
    plt.ylabel(rf"Magnetisation /µ")
    plt.title(f'Critical temperature computation\nCritical temperature = {critical_temperature:.2f}')
    plt.savefig(f"data/{filename}.png", dpi=300)
    plt.show()
    plt.close()


def plot_data_from_file(filename: str) -> None:
    """
    Read the data from the file and plot it
    :param filename: (str) Filename
    :return: None
    """
    data = pd.read_csv('data/data.txt', sep='\t')
    temperatures = data['Temperature']
    final_energy = data['Energy']
    final_magnetization = data['Magnetization']

    # Find the critical temperature

    critical_temperature, smoothed_magnetization, derivative_magnetization = find_critical_temperature(temperatures,
                                                                                                       final_magnetization)
    plot_critical_temperature("critical_temperature", temperatures, final_magnetization, smoothed_magnetization,
                              derivative_magnetization, critical_temperature)

    # Plot the final energy and magnetization

    onsager = [Onsager(2.269, T) for T in temperatures]

    plt.figure(figsize=(14, 10))
    plt.plot(temperatures, final_magnetization, 'bo', label=rf"Simulation data (Monte Carlo)")
    plt.plot(temperatures, onsager, 'g--', label=rf"Onsager solution (Analytical)")
    # plot the critical temperature line
    plt.axvline(x=2.269, color='green', linestyle='-', label=r"Analitical $T_c = 2.269$")
    plt.axvline(x=critical_temperature, color='blue', linestyle='-',
                label=rf"Computated $T_c = {critical_temperature:.2f}$")
    plt.xlabel(rf"Temperature $\times k_b/J$")
    plt.ylabel(rf"Magnetisation /µ")
    plt.legend()
    plt.axhline(c="k", linewidth=1)
    plt.axvline(c="k", linewidth=1)
    plt.title(f'Magnetisation vs Temperature\n(Lattice : {N}x{M}, {iterations} iterations)')
    plt.savefig(f"data/magnetization.png", dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(14, 10))
    plt.plot(temperatures, final_energy, 'bo', label=rf"Simulation data (Monte Carlo)")
    plt.xlabel(rf"Temperature $\times k_b/J$")
    plt.ylabel(rf"Energy /J")
    plt.legend()
    plt.axhline(c="k", linewidth=1)
    plt.axvline(c="k", linewidth=1)
    plt.title(f'Energy vs Temperature\n(Lattice : {N}x{M}, {iterations} iterations)')
    plt.savefig(f"data/energy.png", dpi=300)
    plt.show()

def plot_different_J_graph() -> None:

    # Read all datafile .txt in the data/ folder
    import os
    data_files = os.listdir('data/')
    data_files = [file for file in data_files if file.endswith('.txt')]

    # Plot the different graphs
    plt.figure(figsize=(14, 10))

    for data_file,color in zip(data_files, ['b', 'g', 'r', 'c', 'm', 'y']):
        #get J from the name of the file (filename type is iter_10e6_J_2.1.txt)
        J = float(data_file.split('_')[-1].split('.txt')[0])
        data = pd.read_csv("data/"+data_file, sep='\t')
        temperatures = data['Temperature']
        final_energy = data['Energy']
        final_magnetization = data['Magnetization']

        # Find the critical temperature

        critical_temperature, smoothed_magnetization, derivative_magnetization = find_critical_temperature(temperatures,
                                                                                                           final_magnetization)

        # Plot the final energy and magnetization

        #onsager = [Onsager(2.269*J, T) for T in temperatures]

        plt.plot(temperatures, final_magnetization, color+'o', label=rf"Simulation data J={J} (Monte Carlo)")
        #plt.plot(temperatures, onsager, '--', color=color, label=rf"Onsager solution J={J}(Analytical)")
        # plot the critical temperature line
        plt.axvline(x=critical_temperature, linestyle='-', color=color,
                    label=rf"Computated $T_c = {critical_temperature:.2f}$")
        analytical_Tc = 2.269*J
        plt.axvline(x=analytical_Tc, linestyle='--', label=rf"Analitical $T_c = {analytical_Tc:.2f}$", color=color)

    plt.xlabel(rf"Temperature $\times k_b/J$")
    plt.ylabel(rf"Magnetisation /µ")
    # to have the legend outside the graph
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.axhline(c="k", linewidth=1)
    plt.axvline(c="k", linewidth=1)
    plt.title(f'Magnetisation vs Temperature\n(Lattice : {N}x{M}, {iterations} iterations\nfor different J values)')
    # To have everything in the graph
    plt.tight_layout()
    plt.savefig(f"data/magnetization.png", dpi=300)
    plt.show()
    plt.close()