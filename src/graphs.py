"""
To plot the different graphs of the Ising model

@Author: Lucas BOISTAY
@Date: 2024-02-29
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
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

def plot_different_J_graph_magnetization() -> None:
    J_values = [0.1, 0.3, 0.7, 1.3, 1.5, 2]
    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    # Read all datafile .txt in the data/ folder
    data_files = os.listdir('data/')
    data_files = [file for file in data_files if file.endswith('.txt')]

    plt.figure(figsize=(10, 5))  # Adjusted the figure size for a single plot
    plt.title('Magnetization vs Temperature\n(Lattice: NxM, iterations\nfor different J values)')

    for data_file, color in zip(data_files, colors):
        J = float(data_file.split('_')[-1].split('.txt')[0])
        data = pd.read_csv("data/" + data_file, sep='\t')
        temperatures = data['Temperature']
        final_magnetization = data['Magnetization']

        critical_temperature, _, _ = find_critical_temperature(temperatures,final_magnetization)

        # Plot the final magnetization
        plt.plot(temperatures, final_magnetization, color + 'o', linestyle=':', label=rf"Simulation data J={J} (Monte Carlo)")
        # plot the critical temperature line
        plt.axvline(x=critical_temperature, linestyle='-', color=color,
                    label=rf"Computated $T_c = {critical_temperature:.2f}$")
        plt.axvline(x=2.269 * J, linestyle='--', color=color, label=rf"Analytical $T_c = {2.269 * J:.2f}$")

    plt.ylabel(rf"Magnetization /$\mu$")
    plt.xlabel(rf"Temperature $\times k_b/J$")

    # Example handles for legend (removed susceptibility)
    dotted_line = mlines.Line2D([], [], color='black', marker='o', linestyle=':', label='Numerical Magnetization')
    full_line = mlines.Line2D([], [], color='black', linestyle='-', label='Numerical Critical Temperature')
    dashed_line = mlines.Line2D([], [], color='black', linestyle='--', label='Analytical Critical Temperature')
    J_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='', label=f'J={J}') for J, color in zip(J_values, colors)]

    # Creating the legend
    plt.legend(handles=[dotted_line, full_line,dashed_line] + J_handles, loc='upper right')
    plt.tight_layout()
    plt.savefig(f"data/magnetization.png", dpi=300)
    plt.show()
    plt.close()

def plot_different_J_graph_energy() -> None:
    J_values = [0.1, 0.3, 0.7, 1.3, 1.5, 2]
    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    # Read all datafile .txt in the data/ folder
    data_files = os.listdir('data/')
    data_files = [file for file in data_files if file.endswith('.txt')]

    plt.figure(figsize=(10, 5))  # Adjusted the figure size for a single plot
    plt.title('Energy vs Temperature\n(Lattice: NxM, iterations\nfor different J values)')

    for data_file, color in zip(data_files, colors):
        J = float(data_file.split('_')[-1].split('.txt')[0])
        data = pd.read_csv("data/" + data_file, sep='\t')
        temperatures = data['Temperature']
        final_energy = data['Energy']

        # Plot the final magnetization
        plt.plot(temperatures, final_energy/8, color + 'o', linestyle=':',
                 label=rf"Simulation data J={J} (Monte Carlo)") #/8 to renormalize the energy because of 4 neighbors *2
        plt.axvline(x=2.269 * J, linestyle='--', color=color, label=rf"Analytical $T_c = {2.269 * J:.2f}$")

    plt.ylabel(rf"Energy /J")
    plt.xlabel(rf"Temperature $\times k_b/J$")

    # Example handles for legend (removed susceptibility)
    dotted_line = mlines.Line2D([], [], color='black', marker='o', linestyle=':', label='Numerical Energy')
    dashed_line = mlines.Line2D([], [], color='black', linestyle='--', label='Analytical Critical Temperature')
    J_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='', label=f'J={J}') for J, color in zip(J_values, colors)]

    # Creating the legend
    plt.legend(handles=[dotted_line,dashed_line] + J_handles, loc='upper right')
    plt.tight_layout()
    plt.savefig(f"data/energy.png", dpi=300)
    plt.show()
    plt.close()

def plot_magnetization_and_energy() -> None:
    J_values = [0.1, 0.3, 0.7, 1.3, 1.5, 2]
    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    # Read all data files in the data/ folder that end with .txt
    data_files = os.listdir('data/')
    data_files = [file for file in data_files if file.endswith('.txt')]

    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    axs[0].set_title(f'Magnetization and Energy vs Temperature\n(Lattice: {N}x{M}, Iterations: {iterations:.0e})')

    for data_file, color in zip(data_files, colors):
        J = float(data_file.split('_')[-1].split('.txt')[0])
        data = pd.read_csv(f"data/{data_file}", sep='\t')
        temperatures = data['Temperature']
        final_magnetization = data['Magnetization']
        final_energy = data['Energy']

        critical_temperature, _, _ = find_critical_temperature(temperatures, final_magnetization)

        # Plot magnetization
        axs[0].plot(temperatures, final_magnetization, color + 'o', linestyle=':', label=f"J={J} (MC)", alpha=0.5)
        axs[0].axvline(x=2.269 * J, linestyle='--', color=color, alpha=0.5, label=f"Analytical Tc J={J}")
        axs[0].axvline(x=critical_temperature, linestyle='-', color=color, alpha=0.5, label=f"Computated Tc J={J}")

        # Plot energy, assuming the energy is already normalized or adjust as needed
        axs[1].plot(temperatures, final_energy/8, color + 's', linestyle=':', label=f"J={J} (MC)", alpha=0.5)

    axs[0].set_ylabel(rf"Magnetization /$\mu$")
    axs[1].set_ylabel(rf"Energy /J")
    axs[1].set_xlabel(rf"Temperature $\times k_b/J$")

    # Example handles for legend (removed susceptibility)
    dotted_line = mlines.Line2D([], [], color='black', marker='o', linestyle=':', label='Numerical Magnetization', alpha=0.5)
    dotted_line_energy = mlines.Line2D([], [], color='black', marker='s', linestyle=':', label='Numerical Energy', alpha=0.5)
    full_line = mlines.Line2D([], [], color='black', linestyle='-', label='Numerical Critical Temperature')
    dashed_line = mlines.Line2D([], [], color='black', linestyle='--', label='Analytical Critical Temperature')
    J_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='', label=f'J={J}') for J, color in
                 zip(J_values, colors)]

    # Creating the legend outside the graph
    plt.legend(handles=[dotted_line, dotted_line_energy, full_line, dashed_line] + J_handles, loc='center left',
               bbox_to_anchor=(1.0, 1))
    # Adding grids to the x-axis
    axs[0].grid(axis='x')
    axs[1].grid(axis='x')
    # Adding grids to the y-axis
    axs[0].grid(axis='y')
    axs[1].grid(axis='y')
    # Adjust subplot parameters manually. You might need to tweak these values.
    plt.subplots_adjust(right=0.80, top=0.90)
    plt.subplots_adjust(hspace=0)
    plt.savefig(f"data/combined_magnetization_and_energy.png", dpi=300)
    plt.show()
    plt.close()