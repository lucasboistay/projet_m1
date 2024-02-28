from scipy import constants
import numpy as np

# Physical constants

K = constants.k
J = 1  # Energy unit

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



    def __init__(self, M=10, N=10, T=300, iteration=5000):
        self.M = M # Number of rows
        self.N = N # Number of columns
        self.T = T # Temperature
        self.iteration = iteration # Number of iterations
        self.beta = 1 / (K * T) # Beta
        self.lattice = np.zeros((M, N))

        self.energy_history = []
        self.magnetization_history = []

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

    def print_lattice(self):
        print(self.lattice)

    def energy(self, i, j):
        return -J * self.lattice[i, j] * (
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
        return np.sum(self.lattice)

    def monte_carlo_step(self):
        i, j = np.random.randint(0, self.M), np.random.randint(0, self.N) # Randomly select a site

        energy_before = 2*self.energy(i,j) # Calculate energy before flipping the spin
        self.lattice[i, j] *= -1 # Flip the spin
        energy_after = 2*self.energy(i,j) # Calculate energy after flipping the spin
        dE = energy_after - energy_before # Calculate the change in energy

        if dE < 0 or np.random.random() < np.exp(-dE * self.beta): # Accept the flip with a certain probability
            pass
        else: # Reject the flip
            self.lattice[i, j] *= -1 # Flip the spin back

        self.energy_history.append(self.get_total_energy())
        self.magnetization_history.append(self.magnetization())

    def run_monte_carlo(self):
        for _ in range(self.iteration):
            self.monte_carlo_step()
        return self.lattice

    def get_energy_history(self):
        return self.energy_history

    def get_magnetization_history(self):
        return self.magnetization_history