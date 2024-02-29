"""
author: 	Lucas BOISTAY
date:		2024-02-28
"""

from scipy import constants
import numpy as np
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
# Physical constants

K = 1
J = 1  # Energy unit
MU = 1  # Magnetic moment unit


# Ising model class

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
        return self.lattice

    def print_lattice(self):
        print(self.lattice)

    def energy(self, i, j):
        return 2 * J * self.lattice[i, j] * (
                self.lattice[(i + 1) % self.M, j] +
                self.lattice[i, (j + 1) % self.N] +
                self.lattice[(i - 1) % self.M, j] +
                self.lattice[i, (j - 1) % self.N]
        )

    def get_total_energy(self):
        energy = 0
        for i in range(self.M):
            for j in range(self.N):
                energy += self.energy(i, j)
        return energy

    def magnetization(self):
        return abs(MU*np.sum(np.sum(self.lattice)))

    def flip_spin(self, i, j):
        self.lattice[i, j] *= -1

    def monte_carlo_step(self):
        i, j = np.random.randint(0, self.M), np.random.randint(0, self.N)  # Randomly select a site

        e = self.energy(i, j)  # Calculate energy after flipping the spin

        if e < 0 or np.random.random() < np.exp(-e * self.beta):  # Accept the flip with a certain probability
            self.flip_spin(i, j)
        else:  # If the flip is not accepted, flip the spin back
            pass

    def run_monte_carlo(self):
        # TODO: Enregistrer une image toute les X itérations pour faire une vidéo (modulo itération)
        for i in range(self.iteration):
            self.monte_carlo_step()
        return self.lattice


    def parallel_monte_carlo_step(self, M, N, beta, iteration):
        # TODO: Implement parallel monte carlo step
        pass

    def run_parallel_monte_carlo(self):
        # TODO Implement parallel monte carlo
        pass
