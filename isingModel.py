"""
author: 	Lucas BOISTAY
date:		2024-02-28
"""

from scipy import constants
import numpy as np
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.utils import printProgressBar

# Physical constants

K = 1
J = 1  # Energy unit
MU = 1  # Magnetic moment unit

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

    def __init__(self, M=10, N=10, T=300, iteration=1000):
        """
        Initialize Ising model
        :param M: Number of rows
        :param N: Number of columns
        :param T: Temperature
        :param iteration: Number of iterations
        """
        self.M = M  # Number of rows
        self.N = N  # Number of columns
        self.T = T  # Temperature
        self.iteration = iteration  # Number of iterations
        self.beta = 1 / (K * T)  # Beta
        self.lattice = np.zeros((M, N))

        #self.energy_history = []
        #self.magnetization_history = []

    def initialize_lattice(self, lattice):
        """
        Initialize lattice with different types
        :param lattice: "random", 1, -1, 0
        :return: The initialized lattice
        """
        # Initialize lattice for different types
        if lattice == "random":
            self.lattice = np.random.choice([1, -1], (self.M, self.N))
        elif lattice == 1:
            self.lattice = np.ones((self.M, self.N))
        elif lattice == -1:
            self.lattice = -np.ones((self.M, self.N))
        elif lattice == 0:
            self.lattice = np.zeros((self.M, self.N))
        else:
            raise ValueError("Invalid lattice type")
        return self.lattice

    def get_lattice(self):
        """
        Get lattice
        :return: lattice
        """
        return self.lattice

    def print_lattice(self):
        """
        Print lattice
        :return: None
        """
        print(self.lattice)

    def energy(self, i, j):
        """
        Calculate energy of a site
        :param i: Row index
        :param j: Column index
        :return: Energy of the site
        """
        return 2 * J * self.lattice[i, j] * (
                self.lattice[(i + 1) % self.M, j] +
                self.lattice[i, (j + 1) % self.N] +
                self.lattice[(i - 1) % self.M, j] +
                self.lattice[i, (j - 1) % self.N]
        )

    def get_total_energy(self):
        """
        Calculate total energy
        :return: Total energy
        """
        energy = 0
        for i in range(self.M):
            for j in range(self.N):
                energy += self.energy(i, j)
        return energy

    def magnetization(self):
        """
        Calculate total magnetization
        :return: Total magnetization
        """
        return abs(MU*np.sum(np.sum(self.lattice)))

    def flip_spin(self, i, j):
        """
        Flip the spin at site (i, j)
        :param i: Row index
        :param j: Column index
        :return: None
        """
        self.lattice[i, j] *= -1

    def monte_carlo_step(self):
        """
        One Monte Carlo step
        :return: None
        """
        i, j = np.random.randint(0, self.M), np.random.randint(0, self.N)  # Randomly select a site

        e = self.energy(i, j)  # Calculate energy after flipping the spin

        if e < 0 or np.random.random() < np.exp(-e * self.beta):  # Accept the flip with a certain probability
            self.flip_spin(i, j)
        else:  # If the flip is not accepted, flip the spin back
            pass

    def run_monte_carlo(self, save_image=False):
        """
        Run Monte Carlo simulation, possibility to save a gif of the run.
        :param save_image: Boolean to save the gif (default: False)
        :return: final lattice
        """
        # TODO: Enregistrer une image toute les X itérations pour faire une vidéo (modulo itération)
        images = []
        for i in range(self.iteration):
            self.monte_carlo_step()

            # save an image to have 200 images at the end
            if save_image and i % (self.iteration // 200) == 0:
                images.append(np.copy(self.lattice))

        if save_image: # Save the animation
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_axis_off()

            ims = []
            for i in range(len(images)):
                im = ax.imshow(images[i], cmap='gray', interpolation='nearest', animated=True, vmin=-1, vmax=1)
                ax.set_title(rf"Ising Model Simulation ($\beta = {self.beta:.2f}$, iteration = {self.iteration})")
                ims.append([im])
            print("Creating animation...")
            ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

            print("Saving animation...")
            # Enregistrez l'animation au format GIF
            ani.save('ising.gif', writer='pillow', fps=30, dpi=100)

            plt.close()

        return self.lattice


    def parallel_monte_carlo_step(self, M, N, beta, iteration):
        # TODO: Implement parallel monte carlo step
        pass

    def run_parallel_monte_carlo(self):
        # TODO Implement parallel monte carlo
        pass
