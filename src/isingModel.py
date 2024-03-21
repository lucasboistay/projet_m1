"""
Ising model class
author: 	Lucas BOISTAY
date:		2024-02-28
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Physical constants

K = 1
MU = 1  # Magnetic moment unit

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

# Print iterations progress

class IsingModel:
    """
    Ising model class

    Methods:
    initialize_lattice: Initialize lattice with different types
    print_lattice: Print lattice
    energy: Calculate energy of a site
    get_total_energy: Calculate total energy
    magnetization: Calculate total magnetization
    monte_carlo_step: Monte Carlo step
    run_monte_carlo: Run Monte Carlo simulation
    """

    def __init__(self, M: int = 10, N: int = 10, T: float = 300, iteration: int = 1000, J: float = 1):
        """
        Initialize Ising model
        :param M: (int) Number of rows
        :param N: (int) Number of columns
        :param T: (float) Temperature
        :param iteration: (int) Number of iterations
        """
        self.M = M  # Number of rows
        self.N = N  # Number of columns
        self.T = T  # Temperature
        self.iteration = iteration  # Number of iterations
        self.beta = 1 / (K * T)  # Beta
        self.lattice = np.zeros((M, N))
        self.J = J  # Interaction constant

        # self.energy_history = []
        # self.magnetization_history = []

    def initialize_lattice(self, lattice) -> np.ndarray:
        """
        Initialize lattice with different types
        :param lattice: (int or str) "random", 1, -1
        :return: (np.ndarray) The initialized lattice
        """
        # Initialize lattice for different types
        if lattice == "random":
            self.lattice = np.random.choice([1, -1], (self.M, self.N))
        elif lattice == 1:
            self.lattice = np.ones((self.M, self.N))
        elif lattice == -1:
            self.lattice = -np.ones((self.M, self.N))
        else:
            raise ValueError("Invalid lattice type")
        return self.lattice

    def get_lattice(self) -> np.ndarray:
        """
        Get lattice
        :return: (np.ndarray) lattice
        """
        return self.lattice

    def print_lattice(self) -> None:
        """
        Print lattice
        :return: None
        """
        print(self.lattice)

    def energy(self, i: int, j: int) -> float:
        """
        Calculate energy of a site
        :param i: (int) Row index
        :param j: (int) Column index
        :return: (float) Energy of the site
        """
        return 2 * self.J * self.lattice[i, j] * (
                self.lattice[(i + 1) % self.M, j] +
                self.lattice[i, (j + 1) % self.N] +
                self.lattice[(i - 1) % self.M, j] +
                self.lattice[i, (j - 1) % self.N]
        )

    def get_total_energy(self) -> float:
        """
        Calculate the total energy of the lattice
        :return: Total energy
        """
        energy = 0
        for i in range(self.M):
            for j in range(self.N):
                energy += self.energy(i, j)
        return energy

    def magnetization(self) -> float:
        """
        Calculate total magnetization
        :return: (float) Total magnetization
        """
        return abs(MU * np.sum(np.sum(self.lattice)))

    def flip_spin(self, i: int, j: int) -> None:
        """
        Flip the spin at site (i, j)
        :param i: (int) Row index
        :param j: (int) Column index
        :return: None
        """
        self.lattice[i, j] *= -1

    def metropolis_step(self) -> None:
        """
        One Monte Carlo step
        :return: None
        """
        i, j = np.random.randint(0, self.M), np.random.randint(0, self.N)  # Randomly select a site

        e = self.energy(i, j)  # Calculate energy at the site

        if e < 0 or np.random.random() < np.exp(-e * self.beta):  # Accept the flip with a certain probability
            self.flip_spin(i, j)
        else:  # If the flip is not accepted, flip the spin back
            pass

    def run_monte_carlo_gif(self) -> (None):
        """
        Run Monte Carlo simulation to save a gif of the run.
        :return: (None)
        """

        images = []
        printProgressBar(0, self.iteration, prefix='Progress:', suffix='Complete', length=50)
        for i in range(self.iteration):
            self.metropolis_step()

            # save an image to have 1000 images at the end
            if i % (self.iteration // 200) == 0:
                images.append(np.copy(self.lattice))
            printProgressBar(i, self.iteration, prefix='GIF progress:', suffix='Complete',
                             length=50)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_axis_off()

        ims = []
        for i in range(len(images)):
            im = ax.imshow(images[i], cmap='gray', interpolation='nearest', animated=True, vmin=-1, vmax=1)
            ax.set_title(rf"Ising Model Simulation ($\beta = {self.beta:.2f}$, iteration = {self.iteration:.0e})")
            ims.append([im])

        print("Creating animation...")

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

        print("Saving animation...")

        # Enregistrez l'animation au format GIF
        ani.save('ising.gif', writer='pillow', fps=60, dpi=150)
        # Enregistrez l'animation au format MP4
        # ani.save('ising.mp4', writer='ffmpeg', fps=60, dpi=150)
        plt.show()
        plt.close()

    def run_monte_carlo(self) -> (float, float, float, float):
        """
        Run Monte Carlo simulation, possibility to save a gif of the run.
        :return: (float, float) mean energy and mean magnetization
        """

        energy = []
        magnetization = []

        for i in range(self.iteration):
            self.metropolis_step()

            # To get the mean energy and magnetization of 100 values in the last 10000 iterations
            if i >= self.iteration - 10000 and i % 100 == 0:
                energy.append(self.get_total_energy())
                magnetization.append(self.magnetization())

        mean_energy = np.mean(energy)
        mean_magnetization = np.mean(magnetization)
        specific_heat = np.var(energy)
        susceptibility = np.var(magnetization)

        return mean_energy, mean_magnetization, specific_heat, susceptibility
