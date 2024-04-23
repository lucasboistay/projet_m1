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

    def __init__(self, N: int = 10, T: float = 300, iteration: int = 1000, J: float = 1):
        """
        Initialize Ising model
        :param M: (int) Number of rows
        :param N: (int) Number of columns
        :param T: (float) Temperature
        :param iteration: (int) Number of iterations
        """
        self.M = N  # Number of rows
        self.N = N  # Number of columns
        self.T = T  # Temperature
        self.iteration = iteration  # Number of iterations
        self.beta = 1 / (K * T)  # Beta
        self.lattice = np.zeros((N, N))
        self.J = J  # Interaction constant

        self.J_lattice = np.ones((N, N, 4))

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

    def get_positive_normal_distribution(self, mean, sigma):
        """
        Get a positive normal distribution
        :param mean: (float) mean
        :param sigma: (float) sigma
        :return: (float) positive normal distribution
        """
        value = np.random.normal(mean, sigma)
        while value < 0:
            value = np.random.normal(mean, sigma)
        return value

    def initialize_random_J_lattice(self, mean, sigma) -> np.ndarray:
        """
        Initialize lattice with random J values, following a normal distribution with mean and sigma parameters.
        :return: (np.ndarray) The initialized lattice
        """

        print("Initializing random J lattice...")

        # First value
        self.J_lattice[0, 0, 1] = self.get_positive_normal_distribution(mean, sigma)
        self.J_lattice[0, 0, 2] = self.get_positive_normal_distribution(mean, sigma)

        # First row

        for i in range(1, self.N):
            self.J_lattice[0, i, 1] = self.get_positive_normal_distribution(mean, sigma)
            self.J_lattice[0, i, 2] = self.get_positive_normal_distribution(mean, sigma)
            self.J_lattice[0, i, 3] = self.J_lattice[0, i-1, 1]  # Same value as the previous one (bonded)

        # Rest of the rows
        for i in range(1, self.N):
            for j in range(0, self.N):
                self.J_lattice[i, j, 0] = self.J_lattice[i-1, j, 2]  # Same value as the previous one (bonded)
                self.J_lattice[i, j, 1] = self.get_positive_normal_distribution(mean, sigma)
                self.J_lattice[i, j, 2] = self.get_positive_normal_distribution(mean, sigma)
                self.J_lattice[i, j, 3] = self.J_lattice[i, j-1, 1]  # Same value as the previous one (bonded)

        # Top border
        self.J_lattice[0, :, 0] = 0
        # Bottom border
        self.J_lattice[-1, :, 2] = 0
        # Left border
        self.J_lattice[:, 0, 3] = 0
        # Right border
        self.J_lattice[:, -1, 1] = 0

        print("Random J lattice initialized.")

        return self.J_lattice


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
        return 2 * self.J_lattice[i, j, 1] * self.lattice[i, j] * (
                self.lattice[(i + 1) % self.M, j] * self.J_lattice[i, j, 2] +
                self.lattice[i, (j + 1) % self.N] * self.J_lattice[i, j, 1] +
                self.lattice[(i - 1) % self.M, j] * self.J_lattice[i, j, 0] +
                self.lattice[i, (j - 1) % self.N] * self.J_lattice[i, j, 3]
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

        # border conditions
        if i == 0:
            self.lattice[self.M-1, j] = self.lattice[i, j]
        if i == self.M-1:
            self.lattice[0, j] = self.lattice[i, j]
        if j == 0:
            self.lattice[i, self.N-1] = self.lattice[i, j]
        if j == self.N-1:
            self.lattice[i, 0] = self.lattice[i, j]

    def metropolis_step(self) -> None:
        """
        One Monte Carlo step
        :return: None
        """
        i, j = np.random.randint(0, self.M), np.random.randint(0, self.N)  # Randomly select a site

        e = self.energy(i, j)  # Calculate energy at the site

        self.flip_spin(i, j)  # Flip the spin

        e_new = self.energy(i, j)  # Calculate new energy at the site

        de = e_new - e  # Calculate energy difference

        if de < 0 or np.random.random() < np.exp(-de * self.beta):  # Accept the flip with a certain probability
            pass
        else:  # If the flip is not accepted, flip the spin back
            self.flip_spin(i, j)  # Flip the spin

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
